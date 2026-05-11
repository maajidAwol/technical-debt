"""
Stage 7c - Probability calibration sweep.

Runs the canonical 10-fold within-project CV with two calibration
methods (Platt, isotonic) wrapped via
:class:`sklearn.calibration.CalibratedClassifierCV`. Compares against
an uncalibrated baseline (default ``predict_proba`` from the base
estimator). Writes per-fold metrics, aggregated calibration scores
(Brier, NLL, Expected Calibration Error), and the long-form data
needed to draw reliability diagrams via
:func:`src.reporting.figures.figure_calibration_diagrams`.

Outputs
-------
- ``results/tables/calibration_folds.csv`` - per-fold metrics with
  calibration columns ``brier``, ``nll``, ``ece``.
- ``results/tables/calibration_summary.csv`` - mean +/- std per
  ``(variant, model, method)``.
- ``results/tables/calibration_predictions.parquet`` - long-form
  fold-level (y_true, y_score) pairs for reliability diagrams.

Run
---
.. code-block:: bash

    .\\venv\\Scripts\\python.exe scripts/07c_calibrate.py
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
from config import (  # noqa: E402
    CALIBRATION_METHODS,
    CV_FOLDS,
    LABEL_VARIANTS,
    TABLES_DIR,
)
from src.models.train import (  # noqa: E402
    calibrated_kfold_cv,
    fold_results_to_frame,
    load_variant_matrix,
    stratified_kfold_cv,
    summarize,
)


# Calibration adds a 5-fold inner pass on top of the outer 10-fold, so
# the cost is ~5x the Stage-7 budget. We restrict to the four models
# whose tree-ensemble probabilities are known to need calibration
# (Niculescu-Mizil & Caruana 2005). Logistic regression already
# emits well-calibrated outputs, but we include it as a sanity check.
CALIBRATED_MODELS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
]


def main() -> None:
    t0 = time.time()
    print(f"[Stage 7c] Calibration methods : {CALIBRATION_METHODS}")
    print(f"[Stage 7c] Models              : {CALIBRATED_MODELS}")
    print(f"[Stage 7c] Outer CV folds      : {CV_FOLDS}")

    all_folds = []
    all_long = []  # for reliability diagrams

    for variant in LABEL_VARIANTS:
        X, y, _ = load_variant_matrix(variant)
        print(
            f"\n[Stage 7c] variant={variant:<11}  rows={len(X):,}  feats={X.shape[1]}  "
            f"positives={int(y.sum())} ({100 * y.mean():.2f}%)"
        )
        for model_name in CALIBRATED_MODELS:
            # Uncalibrated baseline (re-running for honest like-for-like
            # comparison; cheap relative to calibrated runs).
            t1 = time.time()
            base_results = stratified_kfold_cv(variant, model_name, X, y, n_splits=CV_FOLDS)
            if not base_results:
                print(f"   {model_name:<20}  baseline SKIPPED (estimator unavailable)")
                continue
            base_df = fold_results_to_frame(base_results)
            base_df["method"] = "uncalibrated"
            base_df["brier"] = float("nan")
            base_df["nll"] = float("nan")
            base_df["ece"] = float("nan")
            all_folds.append(base_df)
            base_means = base_df[["pr_auc", "f1"]].mean()
            print(
                f"   {model_name:<20}  uncalibrated  PR_AUC={base_means['pr_auc']:.3f}  "
                f"F1={base_means['f1']:.3f}  ({time.time() - t1:.1f}s)"
            )

            for method in CALIBRATION_METHODS:
                t2 = time.time()
                cal_results, cal_long = calibrated_kfold_cv(
                    variant, model_name, X, y, n_splits=CV_FOLDS, method=method
                )
                if not cal_results:
                    continue
                cal_df = fold_results_to_frame(cal_results)
                cal_df["method"] = method
                # ``calibrated_kfold_cv`` already attaches brier/nll/ece
                # to each fold's metrics dict; ``fold_results_to_frame``
                # passes them through.
                all_folds.append(cal_df)
                all_long.append(cal_long)
                means = cal_df[["pr_auc", "f1", "brier", "ece"]].mean()
                print(
                    f"   {model_name:<20}  {method:<9}     PR_AUC={means['pr_auc']:.3f}  "
                    f"F1={means['f1']:.3f}  Brier={means['brier']:.4f}  ECE={means['ece']:.4f}  "
                    f"({time.time() - t2:.1f}s)"
                )

    if not all_folds:
        print("[Stage 7c] No results produced.")
        return

    fold_df = pd.concat(all_folds, ignore_index=True)
    fold_df.to_csv(TABLES_DIR / "calibration_folds.csv", index=False)

    summary = summarize(fold_df, group_cols=("variant", "model", "method"))
    summary.to_csv(TABLES_DIR / "calibration_summary.csv", index=False)

    if all_long:
        long_df = pd.concat(all_long, ignore_index=True)
        long_df.to_parquet(TABLES_DIR / "calibration_predictions.parquet", index=False)

    print("\n[Stage 7c] Calibration summary (mean across folds):")
    keep = [c for c in summary.columns if c.endswith("_mean") and c.split("_mean")[0] in {"pr_auc", "f1", "brier", "nll", "ece"}]
    pretty = summary[["variant", "model", "method"] + keep].copy()
    pretty[keep] = pretty[keep].round(4)
    with pd.option_context("display.width", 240, "display.max_rows", None):
        print(pretty.to_string(index=False))

    print(f"\n[Stage 7c] Elapsed: {time.time() - t0:.1f}s")
    print("[Stage 7c] Complete.")


if __name__ == "__main__":
    main()
