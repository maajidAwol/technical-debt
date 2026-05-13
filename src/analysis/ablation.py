"""
Feature-family ablation on the best model.

Five families x two modes plus the all-features baseline = 11 rows:

  - all_features                         (baseline)
  - only_<family> for each of 5 families (isolated contribution)
  - leave_out_<family> for each of 5     (marginal drop)

Within-project 10-fold CV is used because it is fast and the metric
of interest (F1) has the same direction as the LOPO metric for the
ablation purpose. Reports F1, PR-AUC, CE@20 means across folds.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import ALL_FEATURES, CV_FOLDS, FEATURE_FAMILIES  # noqa: E402
from src.models.train import (  # noqa: E402
    fold_results_to_frame,
    load_dataset,
    stratified_kfold_cv,
)


def _run(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None,
    groups: pd.Series | None = None,
) -> dict:
    results = stratified_kfold_cv(
        model_name, X, y, n_splits=CV_FOLDS, params=params, groups=groups
    )
    df = fold_results_to_frame(results)
    return {
        "f1": float(df["f1"].mean()),
        "pr_auc": float(df["pr_auc"].mean()),
        "ce_at_20": float(df["ce_at_20"].mean()),
        "f1_std": float(df["f1"].std()),
    }


def run_family_ablation(
    model_name: str,
    params: dict | None,
    families: dict[str, list[str]] = FEATURE_FAMILIES,
) -> pd.DataFrame:
    """Return the 11-row ablation table for ``model_name``."""
    X, y, _, file_group = load_dataset()
    rows: list[dict] = []

    # Baseline
    base = _run(model_name, X[list(ALL_FEATURES)], y, params, groups=file_group)
    rows.append({"mode": "all_features", "family": "(all)", "n_features": len(ALL_FEATURES), **base})

    # Per-family only / leave-out
    for fam, cols in families.items():
        cols_in = [c for c in cols if c in X.columns]
        only_X = X[cols_in]
        leave_X = X[[c for c in ALL_FEATURES if c not in cols_in]]

        only = _run(model_name, only_X, y, params, groups=file_group)
        rows.append({"mode": "only_this_family", "family": fam, "n_features": len(cols_in), **only})

        leave = _run(model_name, leave_X, y, params, groups=file_group)
        rows.append({"mode": "leave_out_family", "family": fam, "n_features": leave_X.shape[1], **leave})

    out = pd.DataFrame(rows)
    return out
