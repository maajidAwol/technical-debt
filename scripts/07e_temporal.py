"""
Stage 7e - Temporal within-project cross-validation (T1 -> T2).

For each project with sufficient pre-history we re-build features and
consequence-variant labels at two project-specific snapshots
(``T1=40th-percentile`` commit date for training, ``T2=70th-percentile``
for testing), then train at T1 and test at T2. This is the "future
data" sanity check Falessi et al. (2020) recommend for defect-prediction
benchmarks, and the missing piece in the original pipeline that the
2026-04-27 audit flagged.

Outputs
-------
- ``results/tables/temporal_per_project.csv`` - one row per
  (project, model) with T1, T2, sample sizes, and the metric battery.
- ``results/tables/temporal_summary.csv`` - mean +/- std per model
  across projects.

Run
---
.. code-block:: bash

    .\\venv\\Scripts\\python.exe scripts/07e_temporal.py
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
    OBSERVATION_WINDOW_MONTHS,
    TABLES_DIR,
    TEMPORAL_T1_PERCENTILE,
    TEMPORAL_T2_PERCENTILE,
)
from src.models.temporal import (  # noqa: E402
    load_clean_for_temporal,
    temporal_split_results,
)


# Subset of Stage-7 models. SVM is excluded because it must rebuild
# the kernel matrix per project, which compounds the per-project rebuild
# cost. DT is excluded because its high variance is already
# characterised by the within-project CV (and adds noise here).
TEMPORAL_MODELS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
]
METRIC_COLS = ["precision", "recall", "f1", "roc_auc", "pr_auc", "mcc", "ce_at_20"]


def main() -> None:
    t0 = time.time()
    print(f"[Stage 7e] T1 percentile : {TEMPORAL_T1_PERCENTILE}")
    print(f"[Stage 7e] T2 percentile : {TEMPORAL_T2_PERCENTILE}")
    print(f"[Stage 7e] Window months : {OBSERVATION_WINDOW_MONTHS}")
    print(f"[Stage 7e] Models        : {TEMPORAL_MODELS}")
    print("[Stage 7e] Loading cleaned parquets ...")
    data = load_clean_for_temporal()
    projects = sorted(data["commits"]["PROJECT_ID"].dropna().unique().tolist())
    print(f"[Stage 7e] Projects in data: {len(projects)}")

    all_rows = []
    for model in TEMPORAL_MODELS:
        t1_ = time.time()
        df = temporal_split_results(
            model,
            data,
            projects,
            p1=TEMPORAL_T1_PERCENTILE,
            p2=TEMPORAL_T2_PERCENTILE,
            window_months=OBSERVATION_WINDOW_MONTHS,
        )
        if df.empty:
            print(f"   {model:<20}  no eligible projects (insufficient history)")
            continue
        means = df[METRIC_COLS].mean()
        print(
            f"   {model:<20}  projs={len(df):>2}  "
            f"F1={means['f1']:.3f}  PR_AUC={means['pr_auc']:.3f}  "
            f"MCC={means['mcc']:.3f}  CE@20={means['ce_at_20']:.3f}  "
            f"({time.time() - t1_:.1f}s)"
        )
        all_rows.append(df)

    if not all_rows:
        print("[Stage 7e] No results produced (no eligible projects).")
        return

    per_project = pd.concat(all_rows, ignore_index=True)
    per_project.to_csv(TABLES_DIR / "temporal_per_project.csv", index=False)

    summary = (
        per_project.groupby("model")[METRIC_COLS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in summary.columns
    ]
    summary.to_csv(TABLES_DIR / "temporal_summary.csv", index=False)

    print("\n[Stage 7e] Temporal split summary (mean across projects, T1 -> T2):")
    keep = ["model"] + [f"{m}_mean" for m in METRIC_COLS]
    pretty = summary[keep].copy()
    for c in keep[1:]:
        pretty[c] = pretty[c].round(3)
    with pd.option_context("display.width", 240, "display.max_rows", None):
        print(pretty.to_string(index=False))

    print(f"\n[Stage 7e] Elapsed: {time.time() - t0:.1f}s")
    print("[Stage 7e] Complete.")


if __name__ == "__main__":
    main()
