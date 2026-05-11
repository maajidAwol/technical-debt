"""
Temporal within-project cross-validation (T1 -> T2).

Stratified K-fold (Stage 7) shuffles snapshot rows uniformly, which
is fine for the headline within-project benchmark but does not
directly answer the question "if we trained on this project's state
at *time T1*, how well would the resulting model generalise to *time
T2 > T1*?". This module implements that experiment per project:

1. For each project, pull commit dates and pick

   * ``T1`` = ``TEMPORAL_T1_PERCENTILE``-th percentile of pre-snapshot
     commit dates (training time),
   * ``T2`` = ``TEMPORAL_T2_PERCENTILE``-th percentile (testing time).

2. Build features and consequence-variant labels at each of T1 and T2,
   reusing the same builder functions as Stage 5 / 6.

3. Train at T1, predict at T2, record the full metric battery per
   project. Aggregate across projects.

Restricted to the consequence variant by default because (a) it is
the proposal's primary target and (b) running three variants x N
projects x feature rebuild starts to dominate the Colab budget.

References
----------
- Falessi, D., et al. (2020). On the need for time-aware evaluation
  of defect prediction models. Empir. Softw. Eng. 25(2).
- Tantithamthavorn, C., Hassan, A. E. (2018). An experience report on
  defect modelling in practice. ICSE 2018.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    HIGH_RISK_PERCENTILE,
    MIN_PRE_SNAPSHOT_COMMITS,
    OBSERVATION_WINDOW_MONTHS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RISK_SCORE_WEIGHTS,
    SZZ_LEAKY_FEATURES,
    TEMPORAL_N_JOBS,
    TEMPORAL_T1_PERCENTILE,
    TEMPORAL_T2_PERCENTILE,
)
from src.data.labeling import compute_consequence_labels  # noqa: E402
from src.features.graph_features import build_graph_features_for_project  # noqa: E402
from src.features.historical_features import (  # noqa: E402
    build_historical_features_for_project,
)
from src.features.priordefect_features import (  # noqa: E402
    build_priordefect_features_for_project,
)
from src.features.static_features import build_static_features_for_project  # noqa: E402
from src.models.train import _make_model, _metric_row  # noqa: E402


KEY_COLS = ("project_id", "basename")
LABEL_COL = "is_high_risk"
DROP_FOR_CONSEQUENCE = ("risk_score",)


def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts


def _project_temporal_snapshots(
    commits: pd.DataFrame,
    project_id: str,
    p1: float,
    p2: float,
    window_months: int,
) -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return (T1, T2) commit-date percentiles, or None if a project lacks
    enough history (need T2 + window <= last commit and >= MIN_PRE
    commits before T1).
    """
    c = commits[commits["PROJECT_ID"] == project_id]["AUTHOR_DATE"].dropna()
    if len(c) < MIN_PRE_SNAPSHOT_COMMITS * 2:
        return None
    t1 = c.quantile(p1 / 100.0, interpolation="nearest")
    t2 = c.quantile(p2 / 100.0, interpolation="nearest")
    t1 = _ensure_utc(pd.Timestamp(t1))
    t2 = _ensure_utc(pd.Timestamp(t2))

    pre_t1 = (c <= t1).sum()
    if pre_t1 < MIN_PRE_SNAPSHOT_COMMITS:
        return None

    # T2 + W must fit before the project's last commit, otherwise the
    # T2 label window is truncated and "future" leaks beyond the data.
    last_commit = _ensure_utc(c.max())
    if t2 + pd.DateOffset(months=window_months) > last_commit:
        return None
    if t2 <= t1:
        return None

    return t1, t2


def _build_dataset_at(
    project_id: str,
    snapshot: pd.Timestamp,
    data: dict[str, pd.DataFrame],
    window_months: int,
) -> pd.DataFrame:
    """Build features + consequence labels for a single project at one
    snapshot. Returns an X-with-y frame ready for ``_to_xy``.
    """
    static_df = build_static_features_for_project(
        project_id,
        snapshot,
        data["sonar_issues"],
        data["sonar_measures"],
        data["changes"],
    )
    hist_df = build_historical_features_for_project(
        project_id, snapshot, data["commits"], data["changes"]
    )
    graph_df = build_graph_features_for_project(project_id, snapshot, data["changes"])
    prior_df = build_priordefect_features_for_project(
        project_id, snapshot, data["commits"], data["changes"], data["szz"], data["jira"]
    )

    if "snapshot_date" in hist_df.columns:
        hist_df = hist_df.drop(columns=["snapshot_date"])
    if "snapshot_date" in graph_df.columns:
        graph_df = graph_df.drop(columns=["snapshot_date"])
    if "snapshot_date" in prior_df.columns:
        prior_df = prior_df.drop(columns=["snapshot_date"])

    feats = static_df.merge(hist_df, on=list(KEY_COLS), how="outer")
    feats = feats.merge(graph_df, on=list(KEY_COLS), how="left")
    feats = feats.merge(prior_df, on=list(KEY_COLS), how="left")

    labels = compute_consequence_labels(
        project_id,
        snapshot,
        data["commits"],
        data["changes"],
        data["szz"],
        data["jira"],
        window_months=window_months,
        percentile=HIGH_RISK_PERCENTILE,
        weights=RISK_SCORE_WEIGHTS,
    )
    if labels.empty:
        return pd.DataFrame()

    df = feats.merge(
        labels[list(KEY_COLS) + ["risk_score", LABEL_COL]],
        on=list(KEY_COLS),
        how="inner",
    )
    return df


def _to_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    drop = list(KEY_COLS) + [LABEL_COL] + list(DROP_FOR_CONSEQUENCE)
    drop = [c for c in drop if c in df.columns]
    X = df.drop(columns=drop)
    X = X.select_dtypes(include="number")
    y = df[LABEL_COL].astype(int)
    return X, y


def _align_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the *training* schema as canonical: drop test-only columns,
    add zero-filled placeholders for missing ones. Avoids feature drift
    between snapshots (e.g. SonarQube rules unseen at T1).
    """
    train_cols = list(X_train.columns)
    for col in train_cols:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[train_cols]
    return X_train, X_test


def _slice_for_project(
    data: dict[str, pd.DataFrame], pid: str
) -> dict[str, pd.DataFrame]:
    """Per-project slice of every cleaned frame.

    Used to pickle as little as possible into each parallel worker:
    feature builders filter on ``PROJECT_ID`` internally, so passing
    pre-filtered slices is purely an optimisation.
    """
    out: dict[str, pd.DataFrame] = {}
    for name, df in data.items():
        if "PROJECT_ID" in df.columns:
            out[name] = df[df["PROJECT_ID"] == pid].copy()
        else:
            out[name] = df
    return out


def _one_temporal_project(
    pid: str,
    model_name: str,
    sliced: dict[str, pd.DataFrame],
    p1: float,
    p2: float,
    window_months: int,
) -> Optional[dict]:
    """Run the T1 -> T2 experiment for a single project. Returns the
    per-project metric row, or None if the project is skipped.

    Pure function: no shared state, safe under joblib loky workers.
    """
    snaps = _project_temporal_snapshots(
        sliced["commits"], pid, p1, p2, window_months
    )
    if snaps is None:
        return None
    t1, t2 = snaps

    df_t1 = _build_dataset_at(pid, t1, sliced, window_months)
    df_t2 = _build_dataset_at(pid, t2, sliced, window_months)
    if df_t1.empty or df_t2.empty:
        return None
    X_tr, y_tr = _to_xy(df_t1)
    X_te, y_te = _to_xy(df_t2)
    if X_tr.empty or X_te.empty or y_tr.sum() == 0 or y_te.sum() == 0:
        return None
    X_tr, X_te = _align_columns(X_tr, X_te)
    for col in SZZ_LEAKY_FEATURES:
        if col in X_tr.columns:
            X_tr = X_tr.drop(columns=[col])
        if col in X_te.columns:
            X_te = X_te.drop(columns=[col])
    X_tr, X_te = _align_columns(X_tr, X_te)

    est = _make_model(model_name)
    if est is None:
        return None
    # Cap intra-fit parallelism: the outer joblib loop already uses
    # TEMPORAL_N_JOBS workers, and tree ensembles default to
    # ``n_jobs=-1``. Without this cap we would oversubscribe the CPU
    # by a factor of N x cores.
    if hasattr(est, "n_jobs"):
        try:
            est.n_jobs = 1
        except (AttributeError, ValueError):
            pass
    est.fit(X_tr, y_tr.values)
    if hasattr(est, "predict_proba"):
        proba = est.predict_proba(X_te)[:, 1]
    else:
        proba = est.decision_function(X_te)
    pred = (proba >= 0.5).astype(int)
    metrics = _metric_row(y_te.values, pred, proba)

    return {
        "project_id": pid,
        "model": model_name,
        "t1": t1,
        "t2": t2,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_pos_train": int(y_tr.sum()),
        "n_pos_test": int(y_te.sum()),
        **metrics,
    }


def temporal_split_results(
    model_name: str,
    data: dict[str, pd.DataFrame],
    eligible_projects: list[str],
    p1: float = TEMPORAL_T1_PERCENTILE,
    p2: float = TEMPORAL_T2_PERCENTILE,
    window_months: int = OBSERVATION_WINDOW_MONTHS,
    n_jobs: int = TEMPORAL_N_JOBS,
) -> pd.DataFrame:
    """Run T1 -> T2 temporal CV for one model across all eligible projects.

    Returns a DataFrame with one row per project containing snapshots,
    sample sizes, and the full metric battery. Projects with
    insufficient history are skipped silently.

    Projects are processed in parallel with ``n_jobs`` workers
    (default :data:`TEMPORAL_N_JOBS`); per-project work is independent
    so this is a pure speedup. Per-fit ``n_jobs`` is capped to 1 in
    each worker to avoid CPU oversubscription.
    """
    sliced_data = {pid: _slice_for_project(data, pid) for pid in eligible_projects}

    raw = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_one_temporal_project)(
            pid, model_name, sliced_data[pid], p1, p2, window_months
        )
        for pid in eligible_projects
    )
    rows = [r for r in raw if r is not None]
    rows.sort(key=lambda r: r["project_id"])
    return pd.DataFrame(rows)


def load_clean_for_temporal() -> dict[str, pd.DataFrame]:
    """Load the cleaned parquets the temporal pipeline needs (with UTC-aware
    timestamps).
    """
    commits = pd.read_parquet(PROCESSED_DATA_DIR / "clean_git_commits.parquet")
    changes = pd.read_parquet(PROCESSED_DATA_DIR / "clean_git_commits_changes.parquet")
    sonar_issues = pd.read_parquet(PROCESSED_DATA_DIR / "clean_sonar_issues.parquet")
    sonar_measures = pd.read_parquet(PROCESSED_DATA_DIR / "clean_sonar_measures.parquet")
    szz = pd.read_parquet(PROCESSED_DATA_DIR / "clean_szz.parquet")
    jira = pd.read_parquet(PROCESSED_DATA_DIR / "clean_jira_issues.parquet")

    for df, cols in (
        (commits, ["AUTHOR_DATE", "COMMITTER_DATE"]),
        (changes, ["DATE"]),
        (sonar_issues, ["CREATION_DATE", "CLOSE_DATE"]),
        (sonar_measures, ["analysis_date"]),
        (szz, ["fix_date", "induce_date"]),
        (jira, ["CREATION_DATE", "RESOLUTION_DATE", "UPDATE_DATE", "COMMIT_DATE"]),
    ):
        for col in cols:
            if col in df.columns and df[col].dtype.kind == "M" and df[col].dt.tz is None:
                df[col] = df[col].dt.tz_localize("UTC")

    return {
        "commits": commits,
        "changes": changes,
        "sonar_issues": sonar_issues,
        "sonar_measures": sonar_measures,
        "szz": szz,
        "jira": jira,
    }
