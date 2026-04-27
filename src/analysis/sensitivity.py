"""
Sensitivity analysis for the consequence-oriented labeling parameters.

Sweeps a grid of ``(observation_window_months, high_risk_percentile)``
and, for each combination, re-derives consequence labels, re-merges
them with the (fixed) feature matrix and runs a standard 10-fold
stratified CV with the best-performing model family (LightGBM).

The resulting table answers the question: **"Are our headline
predictive-power numbers driven by the specific choice of 6-month
window + top-20-percent threshold, or do they hold across a range of
reasonable parameter settings?"** A robust methodology should show
stable or gracefully-degrading metrics across the grid.

Notes
-----
- Features are snapshot-aware and do NOT depend on the observation
  window, so they are computed once.
- Every label rebuild goes through the same ``compute_consequence_labels``
  function as Stage 4, guaranteeing consistent semantics.
- LightGBM is chosen because it won the LOPO comparison in Stage 8;
  using a single strong model keeps this grid tractable
  (9 configurations x 10 folds = 90 fits, approximately 2 minutes).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DATA_DIR, TABLES_DIR  # noqa: E402
from src.data.labeling import compute_consequence_labels  # noqa: E402
from src.data.snapshot import load_snapshots  # noqa: E402
from src.models.train import (  # noqa: E402
    KEY_COLS,
    fold_results_to_frame,
    stratified_kfold_cv,
)


METRIC_COLS = ["precision", "recall", "f1", "roc_auc", "pr_auc", "mcc", "ce_at_20"]


def _load_raw_for_labeling() -> dict[str, pd.DataFrame]:
    commits = pd.read_parquet(PROCESSED_DATA_DIR / "clean_git_commits.parquet")
    changes = pd.read_parquet(PROCESSED_DATA_DIR / "clean_git_commits_changes.parquet")
    szz = pd.read_parquet(PROCESSED_DATA_DIR / "clean_szz.parquet")
    jira = pd.read_parquet(PROCESSED_DATA_DIR / "clean_jira_issues.parquet")
    for df, cols in (
        (commits, ["AUTHOR_DATE", "COMMITTER_DATE"]),
        (changes, ["DATE"]),
        (szz, ["fix_date", "induce_date"]),
        (jira, ["CREATION_DATE", "RESOLUTION_DATE", "UPDATE_DATE", "COMMIT_DATE"]),
    ):
        for col in cols:
            if col in df.columns and df[col].dtype.kind == "M" and df[col].dt.tz is None:
                df[col] = df[col].dt.tz_localize("UTC")
    return {"commits": commits, "changes": changes, "szz": szz, "jira": jira}


def _recompute_consequence_labels(
    window_months: int,
    percentile: float,
    snapshots: pd.DataFrame,
    raw: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    parts = []
    for _, row in snapshots[snapshots["eligible"]].sort_values("project_id").iterrows():
        df = compute_consequence_labels(
            project_id=row["project_id"],
            snapshot=row["snapshot_date"],
            commits=raw["commits"],
            changes=raw["changes"],
            szz=raw["szz"],
            jira=raw["jira"],
            window_months=window_months,
            percentile=percentile,
        )
        parts.append(df[["project_id", "basename", "is_high_risk"]])
    return pd.concat(parts, ignore_index=True)


def run_sensitivity_grid(
    windows: Iterable[int] = (3, 6, 12),
    percentiles: Iterable[float] = (10.0, 20.0, 30.0),
    model_name: str = "lightgbm",
) -> pd.DataFrame:
    """Run the sensitivity grid and return aggregated metrics per cell."""
    raw = _load_raw_for_labeling()
    snapshots = load_snapshots(PROCESSED_DATA_DIR / "project_snapshots.parquet")
    base = pd.read_parquet(PROCESSED_DATA_DIR / "dataset_consequence.parquet")
    feature_cols = [
        c for c in base.columns
        if c not in set(KEY_COLS) | {"is_high_risk", "risk_score"}
    ]
    features_only = base[list(KEY_COLS) + feature_cols].copy()
    features_numeric = features_only.select_dtypes(include="number").copy()
    features_numeric[list(KEY_COLS)] = features_only[list(KEY_COLS)]

    rows = []
    for w in windows:
        for p in percentiles:
            t0 = time.time()
            labels = _recompute_consequence_labels(w, p, snapshots, raw)
            merged = features_only.merge(labels, on=list(KEY_COLS), how="inner")
            X = merged.drop(columns=list(KEY_COLS) + ["is_high_risk"]).select_dtypes(include="number")
            y = merged["is_high_risk"].astype(int)
            fold_results = stratified_kfold_cv("consequence", model_name, X, y, n_splits=10)
            df = fold_results_to_frame(fold_results)
            means = df[METRIC_COLS].mean().to_dict()
            stds = df[METRIC_COLS].std().to_dict()
            n_pos = int(y.sum())
            rows.append(
                {
                    "window_months": w,
                    "percentile": p,
                    "positives": n_pos,
                    "positive_rate_pct": round(100.0 * y.mean(), 2),
                    **{f"{k}_mean": round(v, 4) for k, v in means.items()},
                    **{f"{k}_std": round(v, 4) for k, v in stds.items()},
                    "elapsed_s": round(time.time() - t0, 2),
                }
            )
    return pd.DataFrame(rows)
