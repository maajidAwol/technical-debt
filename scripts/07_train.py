"""
Stage 7 - Within-project stratified 10-fold cross-validation across all
three label variants and all configured models.

Writes:

- ``results/tables/within_project_folds.csv`` - one row per
  ``(variant, model, fold)`` with the full metric battery.
- ``results/tables/within_project_summary.csv`` - mean +/- std per
  ``(variant, model)``.

Run
---
.. code-block:: bash

    .\\venv\\Scripts\\python.exe scripts/07_train.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CV_FOLDS, LABEL_VARIANTS, TABLES_DIR  # noqa: E402
from src.analysis.significance import (  # noqa: E402
    attach_ci_to_summary,
    bootstrap_confidence_intervals,
    pairwise_wilcoxon,
)
from src.models.train import (  # noqa: E402
    fold_results_to_frame,
    load_variant_matrix,
    stratified_kfold_cv,
    summarize,
)


MODELS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
    "svm",
    # SVM activated 2026-04-27 to satisfy the proposal Section 3.5
    # commitment to compare DT/RF/SVM/GBM. RBF-SVM with class_weight
    # balancing on the within-project 10-fold CV completes in ~95 s
    # per variant - manageable here. Note: SVM is *deliberately
    # omitted* from scripts/08_lopo.py because 22-fold LOPO would
    # cost ~65 min just for SVM (audit refinement #3).
]


def main() -> None:
    t0 = time.time()
    all_folds = []

    print(f"[Stage 7] CV folds: {CV_FOLDS}")
    print(f"[Stage 7] Variants: {LABEL_VARIANTS}")
    print(f"[Stage 7] Models:   {MODELS}")

    # Wipe stale fold-level predictions so a re-run doesn't accumulate
    # rows from earlier configurations.
    pred_path = TABLES_DIR / "within_project_predictions.parquet"
    if pred_path.exists():
        pred_path.unlink()

    for variant in LABEL_VARIANTS:
        X, y, proj = load_variant_matrix(variant)
        pos_rate = 100 * y.mean()
        print(
            f"\n[Stage 7] variant={variant:<11}  rows={len(X):,}  "
            f"feats={X.shape[1]}  positives={int(y.sum())} ({pos_rate:.2f}%)"
        )
        for model_name in MODELS:
            t1 = time.time()
            results = stratified_kfold_cv(
                variant,
                model_name,
                X,
                y,
                n_splits=CV_FOLDS,
                persist_predictions=True,
                project_id=proj,
            )
            if not results:
                print(f"   {model_name:<20}  SKIPPED (not installed)")
                continue
            df = fold_results_to_frame(results)
            means = df[["precision", "recall", "f1", "roc_auc", "pr_auc", "mcc", "ce_at_20"]].mean()
            print(
                f"   {model_name:<20}  "
                f"F1={means['f1']:.3f}  "
                f"ROC_AUC={means['roc_auc']:.3f}  "
                f"PR_AUC={means['pr_auc']:.3f}  "
                f"MCC={means['mcc']:.3f}  "
                f"CE@20={means['ce_at_20']:.3f}  "
                f"({time.time() - t1:.1f}s)"
            )
            all_folds.append(df)

    fold_df = pd.concat(all_folds, ignore_index=True)
    fold_df.to_csv(TABLES_DIR / "within_project_folds.csv", index=False)
    summary = summarize(fold_df)
    summary.to_csv(TABLES_DIR / "within_project_summary.csv", index=False)

    # Bootstrap 95% CIs (10000 resamples) and a side-by-side
    # _with_ci.csv variant for convenient pivoting downstream.
    ci_df = bootstrap_confidence_intervals(fold_df)
    ci_df.to_csv(TABLES_DIR / "within_project_ci.csv", index=False)
    summary_with_ci = attach_ci_to_summary(summary, ci_df)
    summary_with_ci.to_csv(TABLES_DIR / "within_project_summary_with_ci.csv", index=False)

    # Wilcoxon paired tests on per-fold F1 + PR-AUC, Bonferroni
    # corrected per variant (15 pairs at 6 models).
    sig_rows = []
    for metric in ("f1", "pr_auc"):
        sig_rows.append(
            pairwise_wilcoxon(
                fold_df,
                pair_within="variant",
                contrast_col="model",
                metric=metric,
                paired_on=("fold",),
            )
        )
    pairwise_df = pd.concat(sig_rows, ignore_index=True)
    pairwise_df.to_csv(TABLES_DIR / "pairwise_significance.csv", index=False)

    print("\n[Stage 7] Within-project summary (mean across folds):")
    metric_mean_cols = [c for c in summary.columns if c.endswith("_mean")]
    pretty = summary[["variant", "model"] + metric_mean_cols].copy()
    pretty[metric_mean_cols] = pretty[metric_mean_cols].round(3)
    with pd.option_context("display.width", 220, "display.max_rows", None):
        print(pretty.to_string(index=False))

    print(f"\n[Stage 7] Elapsed: {time.time() - t0:.1f}s")
    print("[Stage 7] Complete.")


if __name__ == "__main__":
    main()
