"""
Leave-One-Project-Out cross-project validation core.

For each held-out project ``p``:
  1. Compute project-level feature vectors ``[log(n_files),
     log(pre_commits), positive_rate]`` for every training project
     (positive_rate uses ONLY training labels - the held-out project
     never contributes to weight estimation).
  2. Cosine similarity between the held-out project's feature vector
     and each training project's vector yields a per-project weight in
     [0, 1]. Negative cosines are clipped to a small positive floor so
     dissimilar projects still contribute but with reduced influence.
  3. Broadcast the per-project weight to every training row of that
     project. The resulting ``sample_weight`` is one float per training
     row (shape == (n_train,)) and is passed to ``model.fit``.
  4. Classification uses the standard 0.5 threshold (see train.py
     docstring for rationale).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    RANDOM_STATE,  # noqa: F401  - kept for symmetric imports
)
from src.models.train import (  # noqa: E402
    LABEL_COL,
    _make_model,
    compute_metrics,
    load_dataset,
)


SIMILARITY_FLOOR = 0.05  # weight given to dissimilar projects (no zero-weight rows)


@dataclass
class LopoResult:
    model: str
    held_out_project: str
    n_train: int
    n_test: int
    n_pos_test: int
    metrics: dict[str, float] = field(default_factory=dict)
    note: str = ""  # e.g. "training_only" for daemon


# ---------------------------------------------------------------------------
# Project-level similarity weighting
# ---------------------------------------------------------------------------
def _project_level_features(
    X: pd.DataFrame,
    y: pd.Series,
    proj: pd.Series,
    file_group: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Per-project features ``[log_n_files, log_pre_commits, positive_rate]``.

    For multi-snapshot data, ``file_group`` (``project@basename``) is
    used to count distinct files per project. ``total_commits_pre`` is
    averaged over rows then exponentiated; ``positive_rate`` is the
    pooled mean of ``y``. Each value is **invariant** to how many
    snapshots a file appears in, so similarity vectors are not
    inflated by snapshot count.
    """
    df = X[["total_commits_pre"]].copy()
    df["project_id"] = proj.values
    df["y"] = np.asarray(y, dtype=int)
    df["raw_commits"] = np.expm1(df["total_commits_pre"])
    if file_group is not None:
        df["file_group"] = file_group.values
    grp = df.groupby("project_id")

    if file_group is not None:
        # Distinct files per project (independent of snapshot count)
        n_files = grp["file_group"].nunique().astype(float)
        # Mean commits per row, scaled back to a representative count
        mean_raw_commits = grp["raw_commits"].mean()
        log_pre_commits = np.log1p(mean_raw_commits * n_files)
    else:
        n_files = grp.size().astype(float)
        log_pre_commits = np.log1p(grp["raw_commits"].sum())

    out = pd.DataFrame(
        {
            "log_n_files": np.log1p(n_files),
            "log_pre_commits": log_pre_commits,
            "positive_rate": grp["y"].mean(),
        }
    )
    return out


def _similarity_weights(
    train_proj_features: pd.DataFrame,
    test_features: np.ndarray,
) -> dict[str, float]:
    """Cosine similarity between test_features and each training project."""
    train_mat = train_proj_features.values
    sims = cosine_similarity(test_features.reshape(1, -1), train_mat)[0]
    sims = np.clip(sims, SIMILARITY_FLOOR, 1.0)
    return {pid: float(s) for pid, s in zip(train_proj_features.index, sims)}


def _build_sample_weights(
    train_proj: pd.Series,
    weights_by_project: dict[str, float],
) -> np.ndarray:
    """Broadcast per-project weight to one float per training row.

    Returns an ndarray of shape (n_train,). The caller asserts the
    shape before passing to ``model.fit(sample_weight=...)``.
    """
    return np.array([weights_by_project[pid] for pid in train_proj], dtype=float)


# ---------------------------------------------------------------------------
# Per-fold fit & score
# ---------------------------------------------------------------------------
def _fit_lopo_fold(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    proj_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    test_pid: str,
    proj_features_all: pd.DataFrame,
    params: Optional[dict[str, Any]],
) -> dict[str, float]:
    """Fit on training rows with similarity-weighted sample_weight; score test at 0.5."""
    # Project-level features for the test project, derived from the data
    # itself (not from labels), so no leakage even though we use it to
    # weight training rows.
    if test_pid in proj_features_all.index:
        test_feats = proj_features_all.loc[test_pid].values.astype(float)
    else:
        test_feats = proj_features_all.mean().values.astype(float)

    train_pids = sorted(proj_train.unique())
    train_proj_features = proj_features_all.loc[train_pids]
    weights_by_project = _similarity_weights(train_proj_features, test_feats)

    sample_weight = _build_sample_weights(proj_train, weights_by_project)

    # ITEM 5: row-level shape assertion. sample_weight MUST be one float per
    # training row, NOT one float per project. The assertion below catches
    # an entire class of bugs (passing project-level weights by mistake).
    assert sample_weight.shape == (len(X_train),), (
        f"sample_weight shape {sample_weight.shape} != (n_train,) ({len(X_train)},). "
        "Weights must be one float per training row, not per project."
    )
    assert np.all(np.isfinite(sample_weight)) and (sample_weight > 0).all(), (
        "sample_weight contains non-positive or non-finite values"
    )

    est = _make_model(model_name, params, y_train=y_train)
    if est is None:
        raise RuntimeError(f"Model {model_name!r} not available (import failed)")

    # Pipelines need the sample_weight to be routed to the final step
    # via the named-step convention "<step>__sample_weight". Bare
    # estimators take it directly.
    fit_kwargs: dict[str, Any] = {}
    if hasattr(est, "named_steps"):
        clf_step = list(est.named_steps.keys())[-1]
        fit_kwargs[f"{clf_step}__sample_weight"] = sample_weight
    else:
        fit_kwargs["sample_weight"] = sample_weight

    est.fit(X_train, y_train, **fit_kwargs)
    if hasattr(est, "predict_proba"):
        test_probs = est.predict_proba(X_test)[:, 1]
    else:
        test_probs = est.decision_function(X_test)
    return compute_metrics(y_test, test_probs)


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------
def lopo_cv(
    model_name: str,
    *,
    params: Optional[dict[str, Any]] = None,
    skip_test_projects: Iterable[str] = (),
) -> list[LopoResult]:
    """Leave-One-Project-Out validation for a single model.

    Parameters
    ----------
    skip_test_projects :
        Project IDs that should NEVER be held out for evaluation. They
        still appear in training folds for the other projects. Used to
        keep small projects (e.g. daemon with 4 positives) as training-
        only data per item 1 of the May 13 enhancements.
    """
    X, y, proj, file_group = load_dataset()
    proj_features_all = _project_level_features(X, y, proj, file_group=file_group)

    projects = sorted(proj.unique())
    skip = set(skip_test_projects)
    results: list[LopoResult] = []

    for test_pid in projects:
        train_mask = proj.values != test_pid
        test_mask = proj.values == test_pid

        X_train = X.iloc[train_mask].reset_index(drop=True)
        y_train = np.asarray(y.iloc[train_mask], dtype=int)
        proj_train = proj.iloc[train_mask].reset_index(drop=True)
        X_test = X.iloc[test_mask].reset_index(drop=True)
        y_test = np.asarray(y.iloc[test_mask], dtype=int)

        if test_pid in skip:
            # daemon (or similar tiny project) - included in TRAINING for
            # every other fold but never used as the held-out test set.
            results.append(
                LopoResult(
                    model=model_name,
                    held_out_project=test_pid,
                    n_train=int(train_mask.sum()),
                    n_test=int(test_mask.sum()),
                    n_pos_test=int(y_test.sum()),
                    metrics={"f1": float("nan"), "roc_auc": float("nan"),
                             "pr_auc": float("nan"), "ce_at_20": float("nan")},
                    note="training_only",
                )
            )
            continue

        try:
            metrics = _fit_lopo_fold(
                model_name, X_train, y_train, proj_train,
                X_test, y_test, test_pid, proj_features_all, params,
            )
        except RuntimeError:
            return []

        results.append(
            LopoResult(
                model=model_name,
                held_out_project=test_pid,
                n_train=int(train_mask.sum()),
                n_test=int(test_mask.sum()),
                n_pos_test=int(y_test.sum()),
                metrics=metrics,
                note="",
            )
        )
    return results


def lopo_results_to_frame(results: Iterable[LopoResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "model": r.model,
                "project_id": r.held_out_project,
                "n_train": r.n_train,
                "n_test": r.n_test,
                "n_pos_test": r.n_pos_test,
                "f1": r.metrics["f1"],
                "roc_auc": r.metrics["roc_auc"],
                "pr_auc": r.metrics["pr_auc"],
                "ce_at_20": r.metrics["ce_at_20"],
                "note": r.note,
            }
        )
    return pd.DataFrame(rows)


def lopo_summary(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-project rows into mean +/- std per model.

    Rows with ``note == "training_only"`` (e.g. daemon) are excluded
    from the aggregate so they do not contaminate the mean.
    """
    eligible = fold_df[fold_df["note"] != "training_only"].copy()
    metric_cols = ["f1", "roc_auc", "pr_auc", "ce_at_20"]
    grp = eligible.groupby("model")
    means = grp[metric_cols].mean().add_suffix("_mean")
    stds = grp[metric_cols].std().add_suffix("_std")
    n_eff = grp.size().rename("n_projects_scored")
    return pd.concat([means, stds, n_eff], axis=1).reset_index()
