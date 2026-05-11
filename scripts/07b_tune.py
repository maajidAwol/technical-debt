"""
Stage 7b - Hyperparameter tuning + retrained within-project CV.

For every ``(variant, model)`` from Stage 7 we run an Optuna search
(``TUNING_TRIALS`` trials, inner stratified ``TUNING_INNER_CV_FOLDS``)
optimising mean PR-AUC, then re-evaluate the tuned configuration on
the canonical 10-fold outer CV.

Outputs
-------
- ``results/tables/tuned_params.json`` - the chosen hyperparameter
  set per (variant, model), trial counts, and search elapsed time.
- ``results/tables/within_project_summary_tuned.csv`` - mean / std
  metric battery on the outer CV using the tuned params, in the same
  shape as ``within_project_summary.csv``.
- ``results/tables/within_project_folds_tuned.csv`` - per-fold rows.

Run
---
.. code-block:: bash

    .\\venv\\Scripts\\python.exe scripts/07b_tune.py
"""
from __future__ import annotations

import json
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
    CV_FOLDS,
    LABEL_VARIANTS,
    TABLES_DIR,
    TUNING_INNER_CV_FOLDS,
    TUNING_TRIALS,
)
from src.models.train import (  # noqa: E402
    fold_results_to_frame,
    load_variant_matrix,
    stratified_kfold_cv,
    summarize,
)
from src.models.tuning import tune_model  # noqa: E402


# SVM is excluded - 30 inner folds * (~95 s / fold) ~= 47 min per
# variant for SVM alone, which exceeds our Colab budget. The proposal
# commitment for SVM is satisfied at default hyperparameters by
# scripts/07_train.py (audit refinement #3).
TUNED_MODELS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
]


def main(n_trials: int = TUNING_TRIALS) -> None:
    t0 = time.time()
    print(f"[Stage 7b] Trials per (variant, model): {n_trials}")
    print(f"[Stage 7b] Inner CV folds              : {TUNING_INNER_CV_FOLDS}")
    print(f"[Stage 7b] Outer CV folds              : {CV_FOLDS}")
    print(f"[Stage 7b] Variants                    : {LABEL_VARIANTS}")
    print(f"[Stage 7b] Models                      : {TUNED_MODELS}")

    tuned_records: list[dict] = []
    all_folds = []

    for variant in LABEL_VARIANTS:
        X, y, _ = load_variant_matrix(variant)
        print(
            f"\n[Stage 7b] variant={variant:<11}  rows={len(X):,}  "
            f"feats={X.shape[1]}  positives={int(y.sum())} ({100 * y.mean():.2f}%)"
        )
        for model_name in TUNED_MODELS:
            est_check_unused = None  # keep imports silent
            t1 = time.time()
            try:
                rec = tune_model(variant, model_name, X, y, n_trials=n_trials)
            except Exception as exc:
                print(f"   {model_name:<20}  TUNE-FAILED ({type(exc).__name__}: {exc})")
                continue
            tuned_records.append(rec)
            print(
                f"   {model_name:<20}  "
                f"best_PR_AUC={rec['best_pr_auc']:.3f}  "
                f"trials={rec['n_trials_completed']}/{rec['n_trials_requested']}  "
                f"({rec['elapsed_s']}s tune)"
            )

            # Re-evaluate tuned model on the canonical outer CV.
            results = stratified_kfold_cv(
                variant,
                model_name,
                X,
                y,
                n_splits=CV_FOLDS,
                params=rec["best_params"],
            )
            if not results:
                continue
            df = fold_results_to_frame(results)
            means = df[["precision", "recall", "f1", "roc_auc", "pr_auc", "mcc", "ce_at_20"]].mean()
            print(
                f"   {model_name:<20}  -> outer CV  "
                f"F1={means['f1']:.3f}  PR_AUC={means['pr_auc']:.3f}  "
                f"MCC={means['mcc']:.3f}  CE@20={means['ce_at_20']:.3f}  "
                f"({time.time() - t1:.1f}s total)"
            )
            all_folds.append(df)

    # Persist tuned params (JSON) and outer-CV results (CSV).
    out_json = TABLES_DIR / "tuned_params.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(tuned_records, fh, indent=2)
    print(f"\n[Stage 7b] Tuned params -> {out_json}")

    if all_folds:
        fold_df = pd.concat(all_folds, ignore_index=True)
        fold_df.to_csv(TABLES_DIR / "within_project_folds_tuned.csv", index=False)
        summary = summarize(fold_df)
        summary.to_csv(TABLES_DIR / "within_project_summary_tuned.csv", index=False)

        metric_mean_cols = [c for c in summary.columns if c.endswith("_mean")]
        pretty = summary[["variant", "model"] + metric_mean_cols].copy()
        pretty[metric_mean_cols] = pretty[metric_mean_cols].round(3)
        print("\n[Stage 7b] Within-project summary with tuned hyperparameters:")
        with pd.option_context("display.width", 220, "display.max_rows", None):
            print(pretty.to_string(index=False))

    print(f"\n[Stage 7b] Elapsed: {time.time() - t0:.1f}s")
    print("[Stage 7b] Complete.")


if __name__ == "__main__":
    main()
