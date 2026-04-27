"""
Stage 9 - Sensitivity analyses.

Runs two complementary robustness checks for the claims established in
Stages 7-8:

1. **Sensitivity grid** (consequence variant only) - 3 observation
   windows (3, 6, 12 months) x 3 high-risk percentiles (10, 20, 30) = 9
   configurations, each evaluated with 10-fold stratified CV using
   LightGBM (the strongest LOPO performer from Stage 8).

2. **Feature-group ablation** (all three variants) - for each variant,
   trains with only one group of features at a time, and with each
   group removed. Reveals which feature families carry the predictive
   signal.

Writes
------
- ``results/tables/sensitivity_consequence.csv``
- ``results/tables/feature_ablation.csv``

Run
---
.. code-block:: bash

    .\\venv\\Scripts\\python.exe scripts/09_sensitivity.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import LABEL_VARIANTS, TABLES_DIR  # noqa: E402
from src.analysis.ablation import run_ablation  # noqa: E402
from src.analysis.sensitivity import run_sensitivity_grid  # noqa: E402


def main() -> None:
    t0 = time.time()

    # -----------------------------------------------------------------
    # 1. Sensitivity grid
    # -----------------------------------------------------------------
    print("[Stage 9] Sensitivity grid (consequence variant, LightGBM)")
    print("[Stage 9] windows = {3, 6, 12} months x percentiles = {10, 20, 30}")
    grid = run_sensitivity_grid(windows=(3, 6, 12), percentiles=(10.0, 20.0, 30.0))
    grid.to_csv(TABLES_DIR / "sensitivity_consequence.csv", index=False)

    show = grid[
        [
            "window_months",
            "percentile",
            "positive_rate_pct",
            "f1_mean",
            "roc_auc_mean",
            "pr_auc_mean",
            "mcc_mean",
            "ce_at_20_mean",
            "elapsed_s",
        ]
    ].copy()
    print("\n[Stage 9] Consequence sensitivity grid:")
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(show.to_string(index=False))

    # -----------------------------------------------------------------
    # 2. Feature-group ablation
    # -----------------------------------------------------------------
    ablation_parts = []
    for variant in LABEL_VARIANTS:
        print(f"\n[Stage 9] Feature-group ablation: variant={variant} (LightGBM)")
        t1 = time.time()
        df = run_ablation(variant, model_name="lightgbm")
        ablation_parts.append(df)
        base = df[df["mode"] == "all_features"].iloc[0]
        print(
            f"   baseline   feats={int(base['n_features']):>3}  "
            f"F1={base['f1_mean']:.3f}  "
            f"ROC={base['roc_auc_mean']:.3f}  "
            f"CE@20={base['ce_at_20_mean']:.3f}"
        )
        for _, row in df[df["mode"] != "all_features"].iterrows():
            tag = f"{row['mode']}:{row['group']}"
            print(
                f"   {tag:<40}  feats={int(row['n_features']):>3}  "
                f"F1={row['f1_mean']:.3f}  "
                f"ROC={row['roc_auc_mean']:.3f}  "
                f"CE@20={row['ce_at_20_mean']:.3f}"
            )
        print(f"   ({time.time() - t1:.1f}s)")

    ablation = pd.concat(ablation_parts, ignore_index=True)
    ablation.to_csv(TABLES_DIR / "feature_ablation.csv", index=False)

    print(f"\n[Stage 9] Elapsed total: {time.time() - t0:.1f}s")
    print("[Stage 9] Complete.")


if __name__ == "__main__":
    main()
