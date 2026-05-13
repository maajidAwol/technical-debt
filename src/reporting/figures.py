"""
12 publication-ready figures for the TD prediction thesis.

Every function writes one 300-DPI PNG into ``results/figures/``.
Consistent style: ``seaborn-v0_8-whitegrid``; sizes vary by content
(bar charts 10x6, heatmaps 8x8, multi-subplot 12x5).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless backend for CI / server runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    ALL_FEATURES,
    FEATURE_FAMILIES,
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    TABLES_DIR,
)


_STYLE = "seaborn-v0_8-whitegrid"
_FAMILY_COLORS = {
    "size_complexity": "#1f77b4",
    "static_debt": "#ff7f0e",
    "historical": "#2ca02c",
    "graph": "#d62728",
    "prior_defect": "#9467bd",
}


def _set_style() -> None:
    plt.style.use(_STYLE)


def _feature_to_family() -> dict[str, str]:
    out: dict[str, str] = {}
    for fam, cols in FEATURE_FAMILIES.items():
        for c in cols:
            out[c] = fam
    return out


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 01 - positive rates per project
# ---------------------------------------------------------------------------
def fig_01_positive_rates() -> Path:
    _set_style()
    stats = pd.read_csv(PROCESSED_DATA_DIR / "label_statistics.csv")
    stats = stats.sort_values("positive_rate_pct", ascending=True)
    colors = ["#2ca02c" if r < 15 else "#ff7f0e" for r in stats["positive_rate_pct"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(stats["project_id"], stats["positive_rate_pct"], color=colors)
    mean_rate = stats["positive_rate_pct"].mean()
    ax.axvline(mean_rate, ls="--", color="black", alpha=0.5, label=f"mean = {mean_rate:.1f}%")
    ax.set_xlabel("Positive rate (%)")
    ax.set_title("Per-Project Positive Rate of High-Risk TD Label")
    ax.legend()
    out = FIGURES_DIR / "fig_01_positive_rates.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Fig 02 - label signal breakdown
# ---------------------------------------------------------------------------
def fig_02_label_signal_breakdown() -> Path:
    _set_style()
    labels = pd.read_parquet(PROCESSED_DATA_DIR / "labels.parquet")
    static_any = (labels[["S1_severity", "S2_debt", "S3_smells"]].sum(axis=1) >= 1).astype(int)
    history_any = (labels[["S4_bugfix", "S5_churn", "S6_contributors"]].sum(axis=1) >= 1).astype(int)
    labels = labels.assign(_static=static_any, _history=history_any)

    rows = []
    for pid, sub in labels.groupby("project_id"):
        s_only = int(((sub["_static"] == 1) & (sub["_history"] == 0)).sum())
        h_only = int(((sub["_static"] == 0) & (sub["_history"] == 1)).sum())
        both = int(((sub["_static"] == 1) & (sub["_history"] == 1)).sum())
        neither = int(((sub["_static"] == 0) & (sub["_history"] == 0)).sum())
        rows.append({"project_id": pid, "static_only": s_only, "history_only": h_only,
                     "both": both, "neither": neither})
    df = pd.DataFrame(rows).sort_values("project_id").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    bot = np.zeros(len(df))
    for col, color in (("static_only", "#1f77b4"), ("history_only", "#2ca02c"),
                       ("both", "#d62728"), ("neither", "#cccccc")):
        ax.bar(x, df[col], bottom=bot, label=col, color=color)
        bot += df[col].values
    ax.set_xticks(x)
    ax.set_xticklabels(df["project_id"], rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("File count")
    ax.set_title("Dual-Signal Label Composition per Project")
    ax.legend(loc="upper right")
    out = FIGURES_DIR / "fig_02_label_signal_breakdown.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Fig 03 - risk score distribution
# ---------------------------------------------------------------------------
def fig_03_risk_score_distribution() -> Path:
    _set_style()
    labels = pd.read_parquet(PROCESSED_DATA_DIR / "labels.parquet")
    pos = labels.loc[labels["is_high_risk"] == 1, "risk_score"]
    neg = labels.loc[labels["is_high_risk"] == 0, "risk_score"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 41)
    ax.hist(neg, bins=bins, alpha=0.6, color="#1f77b4", label="is_high_risk = 0", density=True)
    ax.hist(pos, bins=bins, alpha=0.6, color="#d62728", label="is_high_risk = 1", density=True)
    ax.axvline(0.5, ls="--", color="black", alpha=0.6, label="threshold = 0.50")
    ax.set_xlabel("risk_score")
    ax.set_ylabel("density")
    ax.set_title("Risk Score Distribution by Label Class")
    ax.legend()
    out = FIGURES_DIR / "fig_03_risk_score_distribution.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Fig 04 - feature correlation heatmap
# ---------------------------------------------------------------------------
def fig_04_feature_correlation_heatmap() -> Path:
    _set_style()
    ds = pd.read_parquet(PROCESSED_DATA_DIR / "dataset_final.parquet")
    ordered = list(ALL_FEATURES)
    corr = ds[ordered].corr(method="spearman").values
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ordered)))
    ax.set_yticks(range(len(ordered)))
    ax.set_xticklabels(ordered, rotation=80, fontsize=7)
    ax.set_yticklabels(ordered, fontsize=7)

    # Family boundary lines
    boundary = 0
    for fam, cols in FEATURE_FAMILIES.items():
        boundary += len(cols)
        if boundary < len(ordered):
            ax.axhline(boundary - 0.5, color="black", lw=1)
            ax.axvline(boundary - 0.5, color="black", lw=1)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Feature Spearman Correlation Matrix (family-clustered)")
    out = FIGURES_DIR / "fig_04_feature_correlation_heatmap.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Fig 05 - collision resolution per project
# ---------------------------------------------------------------------------
def fig_05_collision_resolution() -> Path:
    _set_style()
    report = pd.read_csv(PROCESSED_DATA_DIR / "collision_report.csv").sort_values("project_id")
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(report))
    ax.bar(x, report["n_kept"], color="#2ca02c", label="kept")
    ax.bar(x, report["n_dropped"], bottom=report["n_kept"], color="#d62728", label="dropped")
    ax.set_xticks(x)
    ax.set_xticklabels(report["project_id"], rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Basename count")
    ax.set_title("Basename Resolution Status per Project")
    ax.legend()
    out = FIGURES_DIR / "fig_05_collision_resolution.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Fig 06 - model comparison F1
# ---------------------------------------------------------------------------
def fig_06_model_comparison_f1() -> Path:
    _set_style()
    within = pd.read_csv(TABLES_DIR / "within_project_results.csv")
    lopo_pp = pd.read_csv(TABLES_DIR / "lopo_per_project.csv")
    lopo_pp = lopo_pp[lopo_pp["note"] != "training_only"]

    models = sorted(within["model"].unique())
    w_mean = within.groupby("model")["f1"].mean().reindex(models)
    w_std = within.groupby("model")["f1"].std().reindex(models)
    l_mean = lopo_pp.groupby("model")["f1"].mean().reindex(models)
    l_std = lopo_pp.groupby("model")["f1"].std().reindex(models)

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w/2, w_mean.values, w, yerr=w_std.values, color="#1f77b4", label="Within-project (fold std)")
    ax.bar(x + w/2, l_mean.values, w, yerr=l_std.values, color="#d62728", label="LOPO (project std)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("F1")
    ax.set_title("Model F1 Comparison: Within-Project vs LOPO")
    ax.legend()
    ax.set_ylim(0, 1)
    out = FIGURES_DIR / "fig_06_model_comparison_f1.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Fig 07 - per-project LOPO F1 for best model
# ---------------------------------------------------------------------------
def fig_07_lopo_per_project_f1(best_model: str) -> Path:
    _set_style()
    lopo_pp = pd.read_csv(TABLES_DIR / "lopo_per_project.csv")
    sub = lopo_pp[(lopo_pp["model"] == best_model) & (lopo_pp["note"] != "training_only")].copy()
    sub = sub.sort_values("f1", ascending=False)
    mean_f1 = sub["f1"].mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(sub))
    ax.bar(x, sub["f1"], color="#1f77b4")
    ax.axhline(mean_f1, ls="--", color="black", alpha=0.6, label=f"mean = {mean_f1:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["project_id"], rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("LOPO F1")
    ax.set_title(f"Per-Project LOPO F1 (Best Model: {best_model})")
    ax.legend()
    ax.set_ylim(0, 1)
    out = FIGURES_DIR / "fig_07_lopo_per_project_f1.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Fig 08 - CE@20 distribution per model
# ---------------------------------------------------------------------------
def fig_08_lopo_ce20_distribution() -> Path:
    _set_style()
    lopo_pp = pd.read_csv(TABLES_DIR / "lopo_per_project.csv")
    lopo_pp = lopo_pp[lopo_pp["note"] != "training_only"]
    models = sorted(lopo_pp["model"].unique())
    data = [lopo_pp.loc[lopo_pp["model"] == m, "ce_at_20"].values for m in models]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, labels=models, patch_artist=True)
    ax.axhline(0.5, ls="--", color="black", alpha=0.6, label="random baseline = 0.5")
    ax.set_ylabel("CE@20")
    ax.set_title("Cross-Project CE@20 Distribution")
    ax.legend()
    ax.set_ylim(0, 1)
    out = FIGURES_DIR / "fig_08_lopo_ce20_distribution.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Fig 09 - F1 vs threshold curve for best model
# ---------------------------------------------------------------------------
def fig_09_threshold_curve(best_model: str, best_params: dict) -> Path:
    _set_style()
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    from src.models.train import _make_model, load_dataset

    X, y, _, _ = load_dataset()
    yv = np.asarray(y, dtype=int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    thresholds = np.arange(0.05, 0.80, 0.01)
    f1s = []
    for t in thresholds:
        fold_f1 = []
        for tr, te in skf.split(X, yv):
            est = _make_model(best_model, best_params, y_train=yv[tr])
            est.fit(X.iloc[tr], yv[tr])
            proba = est.predict_proba(X.iloc[te])[:, 1]
            pred = (proba >= t).astype(int)
            if pred.sum() == 0:
                fold_f1.append(0.0)
                continue
            fold_f1.append(f1_score(yv[te], pred, zero_division=0))
        f1s.append(float(np.mean(fold_f1)))
    f1s = np.array(f1s)
    optimal_idx = int(np.argmax(f1s))
    optimal_t = float(thresholds[optimal_idx])
    f1_at_05 = float(f1s[np.argmin(np.abs(thresholds - 0.5))])
    f1_at_opt = float(f1s[optimal_idx])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, f1s, color="#1f77b4", lw=2)
    ax.axvline(optimal_t, ls="-", color="#d62728", alpha=0.7,
               label=f"optimal = {optimal_t:.2f}  (F1 = {f1_at_opt:.3f})")
    ax.axvline(0.5, ls="--", color="black", alpha=0.6,
               label=f"default 0.5  (F1 = {f1_at_05:.3f})")
    delta = f1_at_opt - f1_at_05
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Mean 5-fold CV F1")
    ax.set_title(f"F1 vs Threshold ({best_model})  -  delta(opt - 0.5) = {delta:+.4f}")
    ax.legend()
    out = FIGURES_DIR / "fig_09_threshold_curve.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Fig 10 - feature family ablation
# ---------------------------------------------------------------------------
def fig_10_ablation() -> Path:
    _set_style()
    abl = pd.read_csv(TABLES_DIR / "ablation_results.csv")
    families = list(FEATURE_FAMILIES.keys())
    baseline = abl[abl["mode"] == "all_features"].iloc[0]

    only = abl[abl["mode"] == "only_this_family"].set_index("family").reindex(families)
    leave = abl[abl["mode"] == "leave_out_family"].set_index("family").reindex(families)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(families))
    w = 0.27

    # F1 subplot
    ax1.bar(x - w, only["f1"], w, label="only_this_family", color="#1f77b4")
    ax1.bar(x, leave["f1"], w, label="leave_out_family", color="#d62728")
    ax1.bar(x + w, [baseline["f1"]] * len(families), w, label="all_features", color="#2ca02c")
    ax1.set_xticks(x)
    ax1.set_xticklabels(families, rotation=20, ha="right")
    ax1.set_ylabel("F1")
    ax1.set_title("F1 by family ablation")
    ax1.set_ylim(0, 1)
    ax1.legend()

    # CE@20 subplot
    ax2.bar(x - w, only["ce_at_20"], w, label="only_this_family", color="#1f77b4")
    ax2.bar(x, leave["ce_at_20"], w, label="leave_out_family", color="#d62728")
    ax2.bar(x + w, [baseline["ce_at_20"]] * len(families), w, label="all_features", color="#2ca02c")
    ax2.set_xticks(x)
    ax2.set_xticklabels(families, rotation=20, ha="right")
    ax2.set_ylabel("CE@20")
    ax2.set_title("CE@20 by family ablation")
    ax2.set_ylim(0, 1)
    ax2.legend()

    fig.suptitle(f"Feature Family Ablation (best model)")
    fig.tight_layout()
    out = FIGURES_DIR / "fig_10_ablation.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Fig 11 - SHAP top-15
# ---------------------------------------------------------------------------
def fig_11_shap_best_model(best_model: str, best_params: dict, sample_size: int = 2000) -> Path:
    _set_style()
    import shap
    from src.models.train import _make_model, load_dataset

    X, y, _, _ = load_dataset()
    yv = np.asarray(y, dtype=int)
    est = _make_model(best_model, best_params, y_train=yv)
    est.fit(X, yv)

    # Subsample for SHAP if large
    if len(X) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_shap = X.iloc[idx]
    else:
        X_shap = X

    explainer = shap.TreeExplainer(est)
    shap_values = explainer.shap_values(X_shap)
    if isinstance(shap_values, list):  # legacy multi-class list-of-arrays
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:  # newer SHAP returns (n, n_feat, n_classes)
        shap_values = shap_values[..., 1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:15]
    top_features = [X.columns[i] for i in top_idx]
    top_values = mean_abs[top_idx]
    feat_fam = _feature_to_family()
    colors = [_FAMILY_COLORS.get(feat_fam.get(f, ""), "#888888") for f in top_features]

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(top_features))[::-1]
    ax.barh(y_pos, top_values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature Importance (SHAP, {best_model})  -  top 15")

    # Family legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=col, label=fam) for fam, col in _FAMILY_COLORS.items()]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    out = FIGURES_DIR / "fig_11_shap_best_model.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Fig 12 - ROC and PR curves for 4 models (LOPO)
# ---------------------------------------------------------------------------
def fig_12_roc_pr_curves(per_model_lopo_curves: dict[str, dict]) -> Path:
    """``per_model_lopo_curves[model] = {"fpr": ..., "tpr": ..., "precision": ...,
    "recall": ..., "positive_rate": float}``"""
    _set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"logistic_regression": "#9467bd", "random_forest": "#2ca02c",
              "xgboost": "#d62728", "lightgbm": "#ff7f0e"}

    # ROC
    for model, curves in per_model_lopo_curves.items():
        ax1.plot(curves["fpr"], curves["tpr"], color=colors.get(model, "black"),
                 label=f"{model} (AUC={curves['roc_auc']:.3f})", lw=2)
    ax1.plot([0, 1], [0, 1], ls="--", color="black", alpha=0.5, label="random")
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title("ROC curves (LOPO, pooled)")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.02)

    # PR
    overall_pos_rate = None
    for model, curves in per_model_lopo_curves.items():
        ax2.plot(curves["recall"], curves["precision"], color=colors.get(model, "black"),
                 label=f"{model} (PR-AUC={curves['pr_auc']:.3f})", lw=2)
        overall_pos_rate = curves.get("positive_rate", overall_pos_rate)
    if overall_pos_rate is not None:
        ax2.axhline(overall_pos_rate, ls="--", color="black", alpha=0.5,
                    label=f"random = {overall_pos_rate:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall curves (LOPO, pooled)")
    ax2.legend()
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.02)

    fig.suptitle("ROC and PR Curves (LOPO)")
    fig.tight_layout()
    out = FIGURES_DIR / "fig_12_roc_pr_curves.png"
    _save(fig, out)
    return out
