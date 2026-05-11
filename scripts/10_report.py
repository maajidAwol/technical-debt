"""
Stage 10 - Final reporting.

Generates:
  1. SHAP + permutation importance per variant (LightGBM, 80/20 split)
  2. Seven publication-ready figures (300 DPI PNG + PDF)
  3. ``docs/06_results.md`` with every table and figure reference
  4. ``docs/07_discussion.md`` scaffold
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import TABLES_DIR  # noqa: E402
from src.analysis.importance import compute_importance  # noqa: E402
from src.reporting import figures as fig  # noqa: E402
from src.reporting.render import render_discussion_scaffold, render_results  # noqa: E402


VARIANTS = ("consequence", "severity", "szz")


def run_importance() -> None:
    print("\n>>> Stage 10.1 - SHAP + permutation importance")
    for v in VARIANTS:
        print(f"  - variant: {v}")
        out = compute_importance(v, model_name="lightgbm")

        top15_shap = out["shap_summary"].head(15)
        top15_perm = out["permutation"].head(15)

        top15_shap.to_csv(TABLES_DIR / f"shap_top15_{v}.csv", index=False)
        top15_perm.to_csv(TABLES_DIR / f"perm_top15_{v}.csv", index=False)
        out["shap_summary"].to_csv(TABLES_DIR / f"shap_full_{v}.csv", index=False)
        out["permutation"].to_csv(TABLES_DIR / f"perm_full_{v}.csv", index=False)

        print(f"    top SHAP: {', '.join(top15_shap['feature'].head(5).tolist())}")
        print(f"    top perm: {', '.join(top15_perm['feature'].head(5).tolist())}")

        fig.fig_shap_summary(v, out["shap_values"], out["X_sample"], top_n=15)
        print(f"    -> fig_shap_{v}.png / .pdf")


def run_figures() -> None:
    print("\n>>> Stage 10.2 - Global figures")
    fig.fig_label_agreement_venn();       print("  - fig_label_agreement_venn")
    fig.fig_per_project_positive_rates(); print("  - fig_per_project_positive_rates")
    fig.fig_within_vs_lopo();             print("  - fig_within_vs_lopo")
    fig.fig_sensitivity_heatmap();        print("  - fig_sensitivity_heatmap")
    fig.fig_feature_ablation();           print("  - fig_feature_ablation")
    fig.fig_lopo_per_project();           print("  - fig_lopo_per_project")

    # Confusion matrices from Stage 7's persisted predictions.
    pred_path = TABLES_DIR / "within_project_predictions.parquet"
    if pred_path.exists():
        preds = pd.read_parquet(pred_path)
        per_proj = fig.figure_confusion_matrices(preds)
        if not per_proj.empty:
            print(f"  - fig_confusion_matrices  ({len(per_proj):,} per-project rows)")
        else:
            print("  - fig_confusion_matrices  (no rows)")
    else:
        print(
            f"  - fig_confusion_matrices  SKIPPED (predictions parquet missing: {pred_path})"
        )

    # Calibration reliability diagrams (only if Stage 7c has been run).
    calib_path = TABLES_DIR / "calibration_predictions.parquet"
    if calib_path.exists():
        calib_long = pd.read_parquet(calib_path)
        fig.figure_calibration_diagrams(calib_long)
        print("  - fig_calibration_reliability")
    else:
        print("  - fig_calibration_reliability  SKIPPED (run scripts/07c_calibrate.py first)")


def run_docs() -> None:
    print("\n>>> Stage 10.3 - Render docs/06_results.md + docs/07_discussion.md")
    r = render_results()
    print(f"  -> {r}")
    d = render_discussion_scaffold()
    print(f"  -> {d}")


def main() -> None:
    run_importance()
    run_figures()
    run_docs()
    print("\nAll Stage 10 artefacts written to results/ and docs/.")


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 40)
    main()
