"""
Stage 8 - Leave-One-Project-Out cross-project validation.

Runs LOPO for every configured model on every label variant, writing:

- ``results/tables/lopo_folds.csv`` - one row per
  ``(variant, model, held_out_project)``.
- ``results/tables/lopo_summary.csv`` - mean +/- std per
  ``(variant, model)`` across projects.
- ``results/tables/lopo_vs_within.csv`` - side-by-side comparison of
  LOPO and within-project means (the "generalization gap" table).

Run
---
.. code-block:: bash

    .\\venv\\Scripts\\python.exe scripts/08_lopo.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

# Silence convergence + feature-name warnings that are cosmetic here.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*convergence.*")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import LABEL_VARIANTS, TABLES_DIR  # noqa: E402
from src.models.cross_project import (  # noqa: E402
    lopo_cv,
    lopo_results_to_frame,
    lopo_summary,
)


MODELS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
]
METRIC_COLS = ["precision", "recall", "f1", "roc_auc", "pr_auc", "mcc", "ce_at_20"]


def _build_gap_table(lopo_summary_df: pd.DataFrame) -> pd.DataFrame:
    within_path = TABLES_DIR / "within_project_summary.csv"
    if not within_path.exists():
        return pd.DataFrame()
    wp = pd.read_csv(within_path)
    keep_cols = ["variant", "model"] + [f"{m}_mean" for m in METRIC_COLS]
    wp = wp[keep_cols].rename(columns={f"{m}_mean": f"{m}_within" for m in METRIC_COLS})
    lp = lopo_summary_df[["variant", "model"] + [f"{m}_mean" for m in METRIC_COLS]].rename(
        columns={f"{m}_mean": f"{m}_lopo" for m in METRIC_COLS}
    )
    merged = wp.merge(lp, on=["variant", "model"], how="outer")
    for m in METRIC_COLS:
        merged[f"{m}_gap"] = (merged[f"{m}_within"] - merged[f"{m}_lopo"]).round(3)
        merged[f"{m}_within"] = merged[f"{m}_within"].round(3)
        merged[f"{m}_lopo"] = merged[f"{m}_lopo"].round(3)
    return merged


def main() -> None:
    t0 = time.time()
    print(f"[Stage 8] Variants : {LABEL_VARIANTS}")
    print(f"[Stage 8] Models   : {MODELS}")
    print("[Stage 8] Leave-one-project-out validation running ...\n")

    all_folds = []
    for variant in LABEL_VARIANTS:
        print(f"[Stage 8] variant={variant}")
        for model_name in MODELS:
            t1 = time.time()
            results = lopo_cv(variant, model_name)
            if not results:
                print(f"   {model_name:<20}  SKIPPED")
                continue
            df = lopo_results_to_frame(results)
            means = df[METRIC_COLS].mean()
            print(
                f"   {model_name:<20}  "
                f"projs={len(df):>2}  "
                f"F1={means['f1']:.3f}  "
                f"ROC={means['roc_auc']:.3f}  "
                f"PR_AUC={means['pr_auc']:.3f}  "
                f"MCC={means['mcc']:.3f}  "
                f"CE@20={means['ce_at_20']:.3f}  "
                f"({time.time() - t1:.1f}s)"
            )
            all_folds.append(df)
        print("")

    fold_df = pd.concat(all_folds, ignore_index=True)
    fold_df.to_csv(TABLES_DIR / "lopo_folds.csv", index=False)

    summary = lopo_summary(fold_df)
    summary.to_csv(TABLES_DIR / "lopo_summary.csv", index=False)

    gap = _build_gap_table(summary)
    if not gap.empty:
        gap.to_csv(TABLES_DIR / "lopo_vs_within.csv", index=False)

    mean_cols = [f"{m}_mean" for m in METRIC_COLS]
    pretty = summary[["variant", "model", "n_projects"] + mean_cols].copy()
    pretty[mean_cols] = pretty[mean_cols].round(3)

    print("[Stage 8] LOPO summary (mean across held-out projects):")
    with pd.option_context("display.width", 240, "display.max_rows", None):
        print(pretty.to_string(index=False))

    if not gap.empty:
        print("\n[Stage 8] Generalization gap (within - LOPO) on key metrics:")
        gap_display_cols = [
            "variant",
            "model",
            "f1_within",
            "f1_lopo",
            "f1_gap",
            "ce_at_20_within",
            "ce_at_20_lopo",
            "ce_at_20_gap",
            "pr_auc_within",
            "pr_auc_lopo",
            "pr_auc_gap",
        ]
        gap_display_cols = [c for c in gap_display_cols if c in gap.columns]
        with pd.option_context("display.width", 240, "display.max_rows", None):
            print(gap[gap_display_cols].to_string(index=False))

    print(f"\n[Stage 8] Elapsed: {time.time() - t0:.1f}s")
    print("[Stage 8] Complete.")


if __name__ == "__main__":
    main()
