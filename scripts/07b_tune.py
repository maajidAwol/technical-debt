"""
Stage 7b - Hyperparameter tuning for the 4 models.

For each model in config.MODEL_ORDER, run an inner-CV PR-AUC tuning
search (Optuna TPE for tree ensembles; GridSearchCV for logistic
regression). Writes one row per model to
``results/tables/tuned_hyperparameters.csv`` with the JSON-encoded
best params.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    MODEL_ORDER,
    TABLES_DIR,
    TUNING_INNER_CV_FOLDS,
    TUNING_TRIALS,
)
from src.models.train import load_dataset  # noqa: E402
from src.models.tuning import tune_model  # noqa: E402


def main() -> None:
    t0 = time.time()
    print(f"[Stage 7b] Models       : {list(MODEL_ORDER)}")
    print(f"[Stage 7b] Trials/model : {TUNING_TRIALS} (LR uses grid search)")
    print(f"[Stage 7b] Inner folds  : {TUNING_INNER_CV_FOLDS}")
    print(f"[Stage 7b] Objective    : PR-AUC (Saito & Rehmsmeier 2015)")

    X, y, _, file_group = load_dataset()
    print(f"[Stage 7b] Dataset      : rows={len(X):,}  feats={X.shape[1]}  positives={int(y.sum())}")

    rows: list[dict] = []
    for model_name in MODEL_ORDER:
        t1 = time.time()
        print(f"\n[Stage 7b] Tuning {model_name} ...")
        try:
            result = tune_model(model_name, X, y, groups=file_group)
        except Exception as e:
            print(f"   FAILED: {type(e).__name__}: {e}")
            continue
        print(
            f"   best_pr_auc={result['best_pr_auc']:.4f}  "
            f"trials={result['n_trials_completed']}  "
            f"method={result['method']}  "
            f"elapsed={result['elapsed_s']}s"
        )
        print(f"   best_params: {result['best_params']}")
        rows.append(
            {
                "model": result["model"],
                "method": result["method"],
                "best_pr_auc": round(result["best_pr_auc"], 6),
                "best_params_json": json.dumps(result["best_params"]),
                "n_trials_completed": result["n_trials_completed"],
                "n_trials_requested": result["n_trials_requested"],
                "inner_splits": result["inner_splits"],
                "elapsed_s": result["elapsed_s"],
            }
        )

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "tuned_hyperparameters.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[Stage 7b] Wrote {out_path}")
    print(f"[Stage 7b] Total elapsed: {time.time() - t0:.1f}s")
    print("[Stage 7b] Complete.")


if __name__ == "__main__":
    main()
