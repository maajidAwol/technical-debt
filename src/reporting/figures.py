"""
Publication-ready figures for the TD prediction thesis.

Every public function writes a 300-DPI PNG and a PDF to ``FIGURES_DIR``
using a consistent greyscale-friendly palette. Figures are designed to
be included directly in the thesis; no further editing is required.

Figures generated:

- ``fig_label_agreement_venn.(png|pdf)`` - 3-way Venn of high-risk sets
- ``fig_per_project_positive_rates.(png|pdf)`` - bar chart per project
- ``fig_within_vs_lopo.(png|pdf)`` - dot-plot of generalization gap
- ``fig_sensitivity_heatmap.(png|pdf)`` - sensitivity ROC / CE@20 grid
- ``fig_feature_ablation.(png|pdf)`` - grouped bar chart
- ``fig_shap_{variant}.(png|pdf)`` - per-variant SHAP summary
- ``fig_lopo_per_project.(png|pdf)`` - LOPO F1 distribution by project
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import FIGURES_DIR, PROCESSED_DATA_DIR, TABLES_DIR  # noqa: E402


PALETTE = {
    "consequence": "#1f77b4",
    "severity": "#d62728",
    "szz": "#2ca02c",
    "all": "#7f7f7f",
    "static_sonar": "#1f77b4",
    "historical": "#ff7f0e",
    "project_context": "#2ca02c",
    "within": "#1f77b4",
    "lopo": "#d62728",
}


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Label agreement Venn
# ---------------------------------------------------------------------------
def fig_label_agreement_venn() -> None:
    from matplotlib_venn import venn3

    cons = pd.read_parquet(PROCESSED_DATA_DIR / "labels_consequence.parquet")
    sev = pd.read_parquet(PROCESSED_DATA_DIR / "labels_severity.parquet")
    szz = pd.read_parquet(PROCESSED_DATA_DIR / "labels_szz.parquet")

    def _key(df):
        return set(
            zip(
                df.loc[df["is_high_risk"], "project_id"],
                df.loc[df["is_high_risk"], "basename"],
            )
        )

    s_cons, s_sev, s_szz = _key(cons), _key(sev), _key(szz)

    fig, ax = plt.subplots(figsize=(6, 5))
    v = venn3(
        [s_cons, s_sev, s_szz],
        set_labels=("Consequence\n(top 20%)", "Severity\n(BLOCKER/CRIT)", "SZZ\n(fault-fix in window)"),
        set_colors=(PALETTE["consequence"], PALETTE["severity"], PALETTE["szz"]),
        alpha=0.55,
        ax=ax,
    )
    if v is not None:
        for label in v.set_labels or []:
            if label:
                label.set_fontsize(10)
        for label in v.subset_labels or []:
            if label:
                label.set_fontsize(9)
    ax.set_title("High-Risk Technical Debt: Three-Label Agreement\n(22 projects, 23,911 files)", fontsize=11)
    _save(fig, "fig_label_agreement_venn")


# ---------------------------------------------------------------------------
# 2. Per-project positive rates
# ---------------------------------------------------------------------------
def fig_per_project_positive_rates() -> None:
    df = pd.read_csv(TABLES_DIR / "label_summary.csv")
    df = df.sort_values("n_basenames", ascending=False)

    x = np.arange(len(df))
    w = 0.28

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, df["consequence_rate_pct"], w, label="Consequence", color=PALETTE["consequence"])
    ax.bar(x, df["severity_rate_pct"], w, label="Severity", color=PALETTE["severity"])
    ax.bar(x + w, df["szz_rate_pct"], w, label="SZZ", color=PALETTE["szz"])

    short = [p.replace("org.apache:", "") for p in df["project_id"]]
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Positive rate (%)")
    ax.set_title("Per-project high-risk positive rates by label variant")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    _save(fig, "fig_per_project_positive_rates")


# ---------------------------------------------------------------------------
# 3. Within vs LOPO (dot-plot)
# ---------------------------------------------------------------------------
def fig_within_vs_lopo() -> None:
    gap = pd.read_csv(TABLES_DIR / "lopo_vs_within.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

    for ax, metric, title in [
        (axes[0], "f1", "F1 score"),
        (axes[1], "ce_at_20", "CE @ top 20%"),
    ]:
        d = gap.dropna(subset=[f"{metric}_within", f"{metric}_lopo"]).copy()
        d["label"] = d["variant"] + " / " + d["model"]
        d = d.sort_values(f"{metric}_lopo")
        y = np.arange(len(d))
        ax.hlines(y, d[f"{metric}_lopo"], d[f"{metric}_within"], color="#888", linewidth=1.2, alpha=0.6)
        ax.scatter(d[f"{metric}_within"], y, color=PALETTE["within"], s=38, label="Within-project", zorder=3)
        ax.scatter(d[f"{metric}_lopo"], y, color=PALETTE["lopo"], s=38, label="LOPO", zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(d["label"], fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel(title)
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        ax.set_title(f"Generalization gap: {title}")
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("Within-project CV vs Leave-One-Project-Out", y=1.02, fontsize=12)
    _save(fig, "fig_within_vs_lopo")


# ---------------------------------------------------------------------------
# 4. Sensitivity heatmap
# ---------------------------------------------------------------------------
def fig_sensitivity_heatmap() -> None:
    df = pd.read_csv(TABLES_DIR / "sensitivity_consequence.csv")
    pivots = {
        "roc_auc_mean": "ROC-AUC",
        "f1_mean": "F1",
        "ce_at_20_mean": "CE @ top 20%",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (col, title) in zip(axes, pivots.items()):
        piv = df.pivot(index="window_months", columns="percentile", values=col)
        sns.heatmap(
            piv,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            ax=ax,
            cbar_kws={"shrink": 0.8},
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(title)
        ax.set_xlabel("Percentile threshold (%)")
        ax.set_ylabel("Observation window (months)")

    fig.suptitle("Consequence-variant sensitivity (LightGBM, 10-fold CV)", y=1.04, fontsize=12)
    _save(fig, "fig_sensitivity_heatmap")


# ---------------------------------------------------------------------------
# 5. Feature ablation
# ---------------------------------------------------------------------------
def fig_feature_ablation() -> None:
    df = pd.read_csv(TABLES_DIR / "feature_ablation.csv")
    only = df[df["mode"] == "only_this_group"].copy()

    groups = ["static_sonar", "historical", "project_context"]
    variants = ["consequence", "severity", "szz"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=False)

    for ax, metric, title in [(axes[0], "f1_mean", "F1"), (axes[1], "ce_at_20_mean", "CE @ top 20%")]:
        x = np.arange(len(variants))
        w = 0.26
        for i, g in enumerate(groups):
            vals = []
            for v in variants:
                r = only[(only["variant"] == v) & (only["group"] == g)]
                vals.append(float(r[metric].iloc[0]) if len(r) else np.nan)
            ax.bar(x + (i - 1) * w, vals, w, label=g.replace("_", " "), color=PALETTE.get(g, None))

        # baseline = all features (dashed line per variant)
        base_vals = []
        for v in variants:
            r = df[(df["variant"] == v) & (df["mode"] == "all_features")]
            base_vals.append(float(r[metric].iloc[0]) if len(r) else np.nan)
        for xi, bv in zip(x, base_vals):
            ax.hlines(bv, xi - 1.5 * w, xi + 1.5 * w, linestyles="--", colors="black", linewidth=1.1)

        ax.set_xticks(x)
        ax.set_xticklabels(variants)
        ax.set_ylabel(title)
        ax.set_title(f"Only-this-group performance ({title})")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9, title="Feature group")

    fig.suptitle("Feature-group ablation (LightGBM). Dashed = all-features baseline.", y=1.02, fontsize=12)
    _save(fig, "fig_feature_ablation")


# ---------------------------------------------------------------------------
# 6. SHAP summary per variant
# ---------------------------------------------------------------------------
def fig_shap_summary(variant: str, shap_values: np.ndarray, X_sample: pd.DataFrame, top_n: int = 15) -> None:
    import shap

    fig = plt.figure(figsize=(8, 0.35 * top_n + 1.5))
    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=top_n,
        show=False,
        plot_size=None,
    )
    plt.title(f"SHAP feature importance - variant: {variant} (LightGBM, top {top_n})", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"fig_shap_{variant}.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / f"fig_shap_{variant}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. LOPO per-project F1 distribution
# ---------------------------------------------------------------------------
def fig_lopo_per_project() -> None:
    df = pd.read_csv(TABLES_DIR / "lopo_folds.csv")
    order = ["consequence", "severity", "szz"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    data = [df.loc[df["variant"] == v, "f1"].values for v in order]
    parts = ax.boxplot(
        data,
        tick_labels=order,
        showmeans=True,
        patch_artist=True,
        medianprops={"color": "black"},
    )
    for patch, v in zip(parts["boxes"], order):
        patch.set_facecolor(PALETTE.get(v, "#ccc"))
        patch.set_alpha(0.7)

    # overlay individual projects
    for i, v in enumerate(order, start=1):
        sub = df[df["variant"] == v]
        ax.scatter(
            np.random.normal(i, 0.06, size=len(sub)),
            sub["f1"],
            alpha=0.5,
            s=22,
            color="black",
        )

    ax.set_ylabel("F1 per held-out project")
    ax.set_title("LOPO F1 distribution per variant (best model per variant, across 22 projects)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    _save(fig, "fig_lopo_per_project")


# ---------------------------------------------------------------------------
# 8. Calibration reliability diagrams (Stage 7c)
# ---------------------------------------------------------------------------
def figure_calibration_diagrams(calib_long_df: pd.DataFrame, n_bins: int = 10) -> None:
    """Reliability diagrams comparing pre- vs post-calibration predictions.

    ``calib_long_df`` is the per-fold long-form table emitted by
    :func:`src.models.train.calibrated_kfold_cv` augmented with the
    uncalibrated baseline rows (typically prepended in the Stage 7c
    driver). Expected columns: ``variant``, ``model``, ``method``
    (one of ``"uncalibrated" | "platt" | "isotonic"``),
    ``y_true`` (list), ``y_score`` (list).

    One panel per (variant, model). All three methods are overlaid
    on the same axes.
    """
    df = calib_long_df.copy()

    # Concatenate per-fold lists so we plot a single reliability curve
    # per (variant, model, method) using all 10 folds combined.
    grouped = df.groupby(["variant", "model", "method"], as_index=False).agg(
        y_true=("y_true", lambda s: np.concatenate([np.asarray(x) for x in s])),
        y_score=("y_score", lambda s: np.concatenate([np.asarray(x) for x in s])),
    )

    method_palette = {
        "uncalibrated": "#7f7f7f",
        "platt": "#1f77b4",
        "isotonic": "#d62728",
    }
    method_marker = {"uncalibrated": "o", "platt": "s", "isotonic": "^"}

    pairs = grouped[["variant", "model"]].drop_duplicates().reset_index(drop=True)
    n_pairs = len(pairs)
    if n_pairs == 0:
        return

    cols = min(3, n_pairs)
    rows = int(np.ceil(n_pairs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.8 * rows), squeeze=False)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for idx, (_, pair) in enumerate(pairs.iterrows()):
        ax = axes[idx // cols][idx % cols]
        sub = grouped[(grouped["variant"] == pair["variant"]) & (grouped["model"] == pair["model"])]

        for _, row in sub.iterrows():
            y_t = np.asarray(row["y_true"])
            y_s = np.asarray(row["y_score"])
            if len(y_t) == 0:
                continue
            bin_ids = np.digitize(y_s, bins[1:-1], right=False)
            xs, ys = [], []
            for b in range(n_bins):
                mask = bin_ids == b
                if not mask.any():
                    continue
                xs.append(float(np.mean(y_s[mask])))
                ys.append(float(np.mean(y_t[mask])))
            ax.plot(
                xs,
                ys,
                marker=method_marker.get(row["method"], "o"),
                color=method_palette.get(row["method"], "#444"),
                label=row["method"],
                linewidth=1.5,
                markersize=5,
            )
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.8, label="perfect")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Empirical positive rate")
        ax.set_title(f"{pair['variant']} / {pair['model']}", fontsize=10)
        ax.grid(linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", fontsize=7)

    # Hide unused axes
    for k in range(n_pairs, rows * cols):
        axes[k // cols][k % cols].axis("off")

    fig.suptitle("Probability calibration reliability diagrams (per (variant, model), 10 outer folds combined)", y=1.02, fontsize=11)
    _save(fig, "fig_calibration_reliability")


# ---------------------------------------------------------------------------
# 9. Confusion matrices (Stage 10 reporting)
# ---------------------------------------------------------------------------
def figure_confusion_matrices(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """3-variant x 6-model grid of within-project confusion matrices.

    Aggregates fold-level predictions over all folds for each
    (variant, model) cell. Returns the per-project errors DataFrame
    that the Stage-10 driver writes to ``per_project_errors.csv``.

    ``predictions_df`` is expected to be the output of
    :func:`src.models.train.stratified_kfold_cv` with
    ``persist_predictions=True``, i.e. columns
    ``variant, model, fold, project_id, y_true, y_score, y_pred``.
    """
    df = predictions_df.copy()
    if df.empty:
        return pd.DataFrame()

    variants_order = ["consequence", "severity", "szz"]
    variants_present = [v for v in variants_order if v in df["variant"].unique()]
    models_present = sorted(df["model"].unique().tolist())

    rows = len(variants_present)
    cols = len(models_present)
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols + 1.0, 2.4 * rows + 0.5), squeeze=False)

    for i, variant in enumerate(variants_present):
        for j, model in enumerate(models_present):
            ax = axes[i][j]
            sub = df[(df["variant"] == variant) & (df["model"] == model)]
            if sub.empty:
                ax.axis("off")
                continue
            y_t = sub["y_true"].astype(int).values
            y_p = sub["y_pred"].astype(int).values
            tn = int(((y_t == 0) & (y_p == 0)).sum())
            fp = int(((y_t == 0) & (y_p == 1)).sum())
            fn = int(((y_t == 1) & (y_p == 0)).sum())
            tp = int(((y_t == 1) & (y_p == 1)).sum())
            cm = np.array([[tn, fp], [fn, tp]])
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                cbar=False,
                xticklabels=["pred 0", "pred 1"],
                yticklabels=["true 0", "true 1"],
                annot_kws={"fontsize": 8},
            )
            ax.set_title(f"{variant}\n{model}", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        "Confusion matrices (within-project 10-fold CV, predictions concatenated)", y=1.02, fontsize=11
    )
    _save(fig, "fig_confusion_matrices")

    # Per-project errors table (FP, FN counts per project x variant x model)
    per_proj = (
        df.groupby(["variant", "model", "project_id"])
        .apply(
            lambda s: pd.Series(
                {
                    "n": len(s),
                    "tp": int(((s["y_true"] == 1) & (s["y_pred"] == 1)).sum()),
                    "fp": int(((s["y_true"] == 0) & (s["y_pred"] == 1)).sum()),
                    "fn": int(((s["y_true"] == 1) & (s["y_pred"] == 0)).sum()),
                    "tn": int(((s["y_true"] == 0) & (s["y_pred"] == 0)).sum()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    per_proj["error_rate"] = ((per_proj["fp"] + per_proj["fn"]) / per_proj["n"]).round(4)
    per_proj.to_csv(TABLES_DIR / "per_project_errors.csv", index=False)
    return per_proj
