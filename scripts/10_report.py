"""
Stage 10 - Render the 12 thesis figures + feature importance tables.

Per spec:
  - SHAP for BEST MODEL only (top 15 features) -> fig_11
  - Permutation importance for ALL 4 models (top 10 each)
    -> results/tables/feature_importance.csv
  - Figures 1-12 as defined in src/reporting/figures.py

Pooled LOPO predictions for fig_12: for each of the 4 models, run
LOPO once collecting (y_true, y_prob) across all 22 held-out projects
(excluding daemon), then compute pooled ROC + PR curves.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import ALL_FEATURES, FIGURES_DIR, MODEL_ORDER, RANDOM_STATE, TABLES_DIR  # noqa: E402
from src.models.cross_project import (  # noqa: E402
    _build_sample_weights,
    _project_level_features,
    _similarity_weights,
)
from src.models.train import _make_model, load_dataset  # noqa: E402
from src.reporting import figures as F  # noqa: E402


SKIP_TEST_PROJECTS = ("org.apache:daemon",)


def _load_tuned_params() -> dict[str, dict]:
    df = pd.read_csv(TABLES_DIR / "tuned_hyperparameters.csv")
    return {row["model"]: json.loads(row["best_params_json"]) for _, row in df.iterrows()}


def _best_model_name() -> str:
    mc = pd.read_csv(TABLES_DIR / "model_comparison.csv")
    return mc[mc["is_best"]].iloc[0]["model"]


# ---------------------------------------------------------------------------
# Permutation importance (within-project, single fold for speed)
# ---------------------------------------------------------------------------
def _permutation_importance_all_models(tuned_params: dict[str, dict]) -> pd.DataFrame:
    X, y, _ = load_dataset()
    yv = np.asarray(y, dtype=int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(iter(skf.split(X, yv)))
    rows: list[dict] = []
    for model_name in MODEL_ORDER:
        params = tuned_params.get(model_name)
        est = _make_model(model_name, params, y_train=yv[train_idx])
        if est is None:
            continue
        est.fit(X.iloc[train_idx], yv[train_idx])
        result = permutation_importance(
            est,
            X.iloc[test_idx],
            yv[test_idx],
            n_repeats=5,
            random_state=RANDOM_STATE,
            scoring="average_precision",
            n_jobs=-1,
        )
        order = np.argsort(result.importances_mean)[::-1][:10]
        for rank, idx in enumerate(order, start=1):
            rows.append(
                {
                    "model": model_name,
                    "rank": rank,
                    "feature": X.columns[idx],
                    "importance_mean": float(result.importances_mean[idx]),
                    "importance_std": float(result.importances_std[idx]),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pooled LOPO predictions per model for fig_12
# ---------------------------------------------------------------------------
def _pooled_lopo_curves(tuned_params: dict[str, dict]) -> dict[str, dict]:
    X, y, proj = load_dataset()
    proj_features_all = _project_level_features(X, y, proj)
    projects = sorted(proj.unique())

    out: dict[str, dict] = {}
    for model_name in MODEL_ORDER:
        params = tuned_params.get(model_name)
        all_y_true: list[np.ndarray] = []
        all_y_prob: list[np.ndarray] = []
        for test_pid in projects:
            if test_pid in SKIP_TEST_PROJECTS:
                continue
            train_mask = proj.values != test_pid
            test_mask = proj.values == test_pid

            X_train = X.iloc[train_mask].reset_index(drop=True)
            y_train = np.asarray(y.iloc[train_mask], dtype=int)
            proj_train = proj.iloc[train_mask].reset_index(drop=True)
            X_test = X.iloc[test_mask].reset_index(drop=True)
            y_test = np.asarray(y.iloc[test_mask], dtype=int)

            if test_pid in proj_features_all.index:
                test_feats = proj_features_all.loc[test_pid].values.astype(float)
            else:
                test_feats = proj_features_all.mean().values.astype(float)
            train_pids = sorted(proj_train.unique())
            train_proj_features = proj_features_all.loc[train_pids]
            weights_by_project = _similarity_weights(train_proj_features, test_feats)
            sample_weight = _build_sample_weights(proj_train, weights_by_project)

            est = _make_model(model_name, params, y_train=y_train)
            if est is None:
                break
            fit_kwargs: dict = {}
            if hasattr(est, "named_steps"):
                clf_step = list(est.named_steps.keys())[-1]
                fit_kwargs[f"{clf_step}__sample_weight"] = sample_weight
            else:
                fit_kwargs["sample_weight"] = sample_weight
            est.fit(X_train, y_train, **fit_kwargs)
            if hasattr(est, "predict_proba"):
                probs = est.predict_proba(X_test)[:, 1]
            else:
                probs = est.decision_function(X_test)
            all_y_true.append(y_test)
            all_y_prob.append(probs)

        if not all_y_true:
            continue
        y_true = np.concatenate(all_y_true)
        y_prob = np.concatenate(all_y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        out[model_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "precision": precision,
            "recall": recall,
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
            "positive_rate": float(np.mean(y_true)),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    tuned_params = _load_tuned_params()
    best_model = _best_model_name()
    best_params = tuned_params.get(best_model, {})
    print(f"[Stage 10] Best model: {best_model}")
    print(f"[Stage 10] Best params: {best_params}")

    generated: list[str] = []
    failed: list[tuple[str, str]] = []

    # --- Static figures (no model training needed) ---
    for fn, name in (
        (F.fig_01_positive_rates, "fig_01"),
        (F.fig_02_label_signal_breakdown, "fig_02"),
        (F.fig_03_risk_score_distribution, "fig_03"),
        (F.fig_04_feature_correlation_heatmap, "fig_04"),
        (F.fig_05_collision_resolution, "fig_05"),
        (F.fig_06_model_comparison_f1, "fig_06"),
        (F.fig_08_lopo_ce20_distribution, "fig_08"),
        (F.fig_10_ablation, "fig_10"),
    ):
        try:
            path = fn()
            print(f"   {name} OK -> {path.name}")
            generated.append(name)
        except Exception as e:
            print(f"   {name} FAIL: {type(e).__name__}: {e}")
            failed.append((name, f"{type(e).__name__}: {e}"))

    # --- Best-model dependent figures ---
    try:
        path = F.fig_07_lopo_per_project_f1(best_model)
        print(f"   fig_07 OK -> {path.name}")
        generated.append("fig_07")
    except Exception as e:
        print(f"   fig_07 FAIL: {type(e).__name__}: {e}")
        failed.append(("fig_07", f"{type(e).__name__}: {e}"))

    try:
        t1 = time.time()
        path = F.fig_09_threshold_curve(best_model, best_params)
        print(f"   fig_09 OK -> {path.name}  ({time.time() - t1:.1f}s)")
        generated.append("fig_09")
    except Exception as e:
        print(f"   fig_09 FAIL: {type(e).__name__}: {e}")
        failed.append(("fig_09", f"{type(e).__name__}: {e}"))

    try:
        t1 = time.time()
        path = F.fig_11_shap_best_model(best_model, best_params)
        print(f"   fig_11 OK -> {path.name}  ({time.time() - t1:.1f}s)")
        generated.append("fig_11")
    except Exception as e:
        print(f"   fig_11 FAIL: {type(e).__name__}: {e}")
        failed.append(("fig_11", f"{type(e).__name__}: {e}"))

    # --- Pooled LOPO curves for fig_12 ---
    print("\n[Stage 10] Running pooled LOPO for fig_12 ...")
    try:
        t1 = time.time()
        curves = _pooled_lopo_curves(tuned_params)
        print(f"[Stage 10] Pooled LOPO done ({time.time() - t1:.1f}s, {len(curves)} models)")
        path = F.fig_12_roc_pr_curves(curves)
        print(f"   fig_12 OK -> {path.name}")
        generated.append("fig_12")
    except Exception as e:
        print(f"   fig_12 FAIL: {type(e).__name__}: {e}")
        failed.append(("fig_12", f"{type(e).__name__}: {e}"))

    # --- Permutation importance table ---
    print("\n[Stage 10] Computing permutation importance (4 models, top 10 each) ...")
    try:
        t1 = time.time()
        perm_df = _permutation_importance_all_models(tuned_params)
        perm_path = TABLES_DIR / "feature_importance.csv"
        perm_df.to_csv(perm_path, index=False)
        print(f"[Stage 10] Wrote {perm_path}  ({len(perm_df)} rows, {time.time() - t1:.1f}s)")
    except Exception as e:
        print(f"[Stage 10] permutation importance FAIL: {type(e).__name__}: {e}")

    # --- Summary ---
    print("\n[Stage 10] Figure generation summary:")
    print(f"   Generated: {len(generated)} / 12")
    for n in sorted(generated):
        print(f"      {n}")
    if failed:
        print(f"   Failed: {len(failed)}")
        for n, msg in failed:
            print(f"      {n}: {msg}")
    print(f"\n[Stage 10] Elapsed: {time.time() - t0:.1f}s")
    print("[Stage 10] Complete.")


if __name__ == "__main__":
    main()
