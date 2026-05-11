"""
Stage 7d - Resampling comparison: SMOTE vs class_weight="balanced".

Re-runs within-project 10-fold CV on every (variant, model) pair using
SMOTE oversampling on the training fold (test fold untouched), and
contrasts against the class-weighted baseline already produced by
:mod:`scripts.07_train`. The output ``resampling_comparison.csv``
provides the side-by-side mean metric battery the thesis discussion
section references when arguing whether synthetic minority generation
adds value beyond cost-sensitive learning.

Outputs
-------
- ``results/tables/within_project_folds_smote.csv``
- ``results/tables/within_project_summary_smote.csv``
- ``results/tables/resampling_comparison.csv`` - one row per
  ``(variant, model)`` with ``f1_cw`` / ``f1_smote`` / ``f1_delta``,
  same for PR-AUC, MCC, CE@20.

Run
---
.. code-block:: bash

    .\\venv\\Scripts\\python.exe scripts/07d_resample.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*convergence.*")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CV_FOLDS, LABEL_VARIANTS, TABLES_DIR  # noqa: E402
from src.models.train import (  # noqa: E402
    fold_results_to_frame,
    load_variant_matrix,
    stratified_kfold_cv,
    summarize,
)


# Same model list as Stage 7. SVM with SMOTE on the full feature
# matrix is workable here because we only run once per (variant,
# model), not 22 times as in LOPO.
MODELS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
    "svm",
]
KEY_METRICS = ("f1", "pr_auc", "mcc", "ce_at_20")


def _build_comparison(
    cw_summary: pd.DataFrame, smote_summary: pd.DataFrame
) -> pd.DataFrame:
    """Side-by-side table: class_weight vs SMOTE on the headline metrics."""
    cw = cw_summary.rename(
        columns={f"{m}_mean": f"{m}_cw" for m in KEY_METRICS}
    )[["variant", "model"] + [f"{m}_cw" for m in KEY_METRICS]]
    sm = smote_summary.rename(
        columns={f"{m}_mean": f"{m}_smote" for m in KEY_METRICS}
    )[["variant", "model"] + [f"{m}_smote" for m in KEY_METRICS]]
    out = cw.merge(sm, on=["variant", "model"], how="outer")
    for m in KEY_METRICS:
        out[f"{m}_delta"] = (out[f"{m}_smote"] - out[f"{m}_cw"]).round(3)
        out[f"{m}_cw"] = out[f"{m}_cw"].round(3)
        out[f"{m}_smote"] = out[f"{m}_smote"].round(3)
    return out


def main() -> None:
    t0 = time.time()
    print(f"[Stage 7d] Models   : {MODELS}")
    print(f"[Stage 7d] Variants : {LABEL_VARIANTS}")
    print(f"[Stage 7d] CV folds : {CV_FOLDS}")

    all_folds = []
    for variant in LABEL_VARIANTS:
        X, y, _ = load_variant_matrix(variant)
        print(
            f"\n[Stage 7d] variant={variant:<11}  rows={len(X):,}  feats={X.shape[1]}  "
            f"positives={int(y.sum())} ({100 * y.mean():.2f}%)"
        )
        for model_name in MODELS:
            t1 = time.time()
            results = stratified_kfold_cv(
                variant, model_name, X, y, n_splits=CV_FOLDS, use_smote=True
            )
            if not results:
                print(f"   {model_name:<20}  SKIPPED (estimator unavailable)")
                continue
            df = fold_results_to_frame(results)
            df["resampler"] = "smote"
            means = df[list(KEY_METRICS)].mean()
            print(
                f"   {model_name:<20}  SMOTE  "
                f"F1={means['f1']:.3f}  PR_AUC={means['pr_auc']:.3f}  "
                f"MCC={means['mcc']:.3f}  CE@20={means['ce_at_20']:.3f}  "
                f"({time.time() - t1:.1f}s)"
            )
            all_folds.append(df)

    if not all_folds:
        print("[Stage 7d] No results.")
        return

    fold_df = pd.concat(all_folds, ignore_index=True)
    fold_df.to_csv(TABLES_DIR / "within_project_folds_smote.csv", index=False)
    smote_summary = summarize(fold_df)
    smote_summary.to_csv(TABLES_DIR / "within_project_summary_smote.csv", index=False)

    cw_path = TABLES_DIR / "within_project_summary.csv"
    if cw_path.exists():
        cw_summary = pd.read_csv(cw_path)
        comparison = _build_comparison(cw_summary, smote_summary)
        comparison.to_csv(TABLES_DIR / "resampling_comparison.csv", index=False)
        print("\n[Stage 7d] Class-weight vs SMOTE comparison (means):")
        with pd.option_context("display.width", 240, "display.max_rows", None):
            print(comparison.to_string(index=False))
    else:
        print(
            "[Stage 7d] Note: class-weighted baseline not found "
            f"({cw_path}); skipping comparison table. Run scripts/07_train.py first."
        )

    print(f"\n[Stage 7d] Elapsed: {time.time() - t0:.1f}s")
    print("[Stage 7d] Complete.")


if __name__ == "__main__":
    main()
