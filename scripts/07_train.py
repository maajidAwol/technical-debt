"""
Stage 7 - Within-project stratified 10-fold CV with threshold optimisation.

Two modes:

- ``python scripts/07_train.py``            -> Stage A (default hyperparams)
    Writes ``results/tables/default_cv_results.csv``.

- ``python scripts/07_train.py --tuned``    -> Stage C (tuned hyperparams)
    Reads tuned params from ``results/tables/tuned_hyperparameters.csv``
    and rewrites ``results/tables/within_project_results.csv``.

Both modes use threshold optimisation per fold (see Part 6 of the spec).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CV_FOLDS, MODEL_ORDER, TABLES_DIR  # noqa: E402
from src.models.train import (  # noqa: E402
    fold_results_to_frame,
    load_dataset,
    stratified_kfold_cv,
    summarize_by_model,
)


def _load_tuned_params() -> dict[str, dict]:
    """Return {model_name -> best_params dict} from the tuning CSV."""
    path = TABLES_DIR / "tuned_hyperparameters.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/07b_tune.py before --tuned mode."
        )
    df = pd.read_csv(path)
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        out[row["model"]] = json.loads(row["best_params_json"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuned", action="store_true", help="Use tuned hyperparams from 07b_tune.py")
    args = parser.parse_args()

    t0 = time.time()
    print(f"[Stage 7] Mode      : {'tuned' if args.tuned else 'default'}")
    print(f"[Stage 7] CV folds  : {CV_FOLDS}")
    print(f"[Stage 7] Models    : {list(MODEL_ORDER)}")

    X, y, _, file_group = load_dataset()
    pos_rate = 100 * float(y.mean())
    print(f"[Stage 7] Dataset   : rows={len(X):,}  feats={X.shape[1]}  positives={int(y.sum())} ({pos_rate:.2f}%)")

    tuned_params = _load_tuned_params() if args.tuned else {}
    all_folds: list[pd.DataFrame] = []
    for model_name in MODEL_ORDER:
        t1 = time.time()
        params = tuned_params.get(model_name) if args.tuned else None
        results = stratified_kfold_cv(
            model_name, X, y, n_splits=CV_FOLDS, params=params, groups=file_group
        )
        if not results:
            print(f"   {model_name:<22}  SKIPPED (estimator unavailable)")
            continue
        df = fold_results_to_frame(results)
        means = df[["f1", "roc_auc", "pr_auc", "ce_at_20"]].mean()
        print(
            f"   {model_name:<22}  "
            f"F1={means['f1']:.3f}  "
            f"ROC={means['roc_auc']:.3f}  "
            f"PR={means['pr_auc']:.3f}  "
            f"CE@20={means['ce_at_20']:.3f}  "
            f"({time.time() - t1:.1f}s)"
        )
        all_folds.append(df)

    fold_df = pd.concat(all_folds, ignore_index=True)
    out_name = "within_project_results.csv" if args.tuned else "default_cv_results.csv"
    fold_df.to_csv(TABLES_DIR / out_name, index=False)
    print(f"\n[Stage 7] Wrote {TABLES_DIR / out_name}  ({len(fold_df)} rows)")

    summary = summarize_by_model(fold_df)
    print("\n[Stage 7] Per-model summary (mean across folds):")
    with pd.option_context("display.width", 220, "display.max_rows", None):
        cols_show = ["model", "f1_mean", "roc_auc_mean", "pr_auc_mean", "ce_at_20_mean"]
        cols_show = [c for c in cols_show if c in summary.columns]
        print(summary[cols_show].round(3).to_string(index=False))

    print(f"\n[Stage 7] Elapsed: {time.time() - t0:.1f}s")
    print("[Stage 7] Complete.")


if __name__ == "__main__":
    main()
