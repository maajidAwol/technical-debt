"""
Within-project stratified K-fold training core.

Builds models, runs K-fold CV, computes the four-metric battery
(F1, ROC-AUC, PR-AUC, CE@20) at a fixed threshold of 0.5. Two
regimes are supported:

- ``stratified_kfold_cv(..., params=None)`` -- default hyperparams.
- ``stratified_kfold_cv(..., params=<dict>)`` -- tuned hyperparams from
  the Optuna driver (scripts/07b_tune.py).

Threshold policy: classification uses the standard 0.5 cutoff on
``predict_proba``. Tree ensembles with ``class_weight='balanced'``
(or ``scale_pos_weight`` for XGBoost) are already well-calibrated at
the 17% positive rate of this dataset; per-fold threshold sweeping
added complexity without measurable F1 gain in our pilot
(within-project F1 deltas were +0.002 / -0.002 / -0.012 / +0.034
for xgboost / lightgbm / random_forest / logistic_regression), and
the negative deltas mean per-fold threshold variance was hurting
the ensembles. We therefore report F1 at the unambiguous 0.5
threshold.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    ALL_FEATURES,
    COST_EFFECTIVENESS_AT,
    CV_FOLDS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)

CLASSIFICATION_THRESHOLD = 0.5


KEY_COLS = ("project_id", "basename")
LABEL_COL = "is_high_risk"

MODEL_NAMES = ("logistic_regression", "random_forest", "xgboost", "lightgbm")


# ---------------------------------------------------------------------------
# Model zoo (4 models)
# ---------------------------------------------------------------------------
def _merged_params(defaults: dict[str, Any], overrides: Optional[dict[str, Any]]) -> dict[str, Any]:
    out = dict(defaults)
    if overrides:
        out.update(overrides)
    return out


def _make_model(name: str, params: Optional[dict[str, Any]] = None, *, y_train: Optional[np.ndarray] = None):
    """Build an sklearn estimator by name.

    Parameters
    ----------
    name : one of MODEL_NAMES.
    params : optional overrides merged over the estimator defaults.
    y_train : optional training labels used to compute ``scale_pos_weight``
        for XGBoost (``n_neg / n_pos``). Only used when ``name == "xgboost"``.

    Logistic regression is wrapped in a Pipeline with StandardScaler so
    per-fold scaling is honest. Tree ensembles don't need scaling but
    are returned bare for speed.
    """
    name = name.lower()
    if name == "logistic_regression":
        defaults = {
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": RANDOM_STATE,
            "solver": "lbfgs",
        }
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(**_merged_params(defaults, params))),
            ]
        )
    if name == "random_forest":
        defaults = {
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
        }
        return RandomForestClassifier(**_merged_params(defaults, params))
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            return None
        defaults = {
            "eval_metric": "aucpr",
            "random_state": RANDOM_STATE,
            "verbosity": 0,
            "n_jobs": -1,
        }
        if y_train is not None:
            n_pos = int(np.sum(y_train == 1))
            n_neg = int(np.sum(y_train == 0))
            if n_pos > 0:
                defaults["scale_pos_weight"] = float(n_neg) / float(n_pos)
        return XGBClassifier(**_merged_params(defaults, params))
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            return None
        defaults = {
            "class_weight": "balanced",
            "verbose": -1,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        return LGBMClassifier(**_merged_params(defaults, params))
    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Four-metric battery
# ---------------------------------------------------------------------------
def _recall_at_top_20(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Recall captured in the top-20% highest-scoring rows (CE@20)."""
    n = len(y_true)
    total_pos = int(np.asarray(y_true).sum())
    if n == 0 or total_pos == 0:
        return 0.0
    top_k = max(1, int(n * COST_EFFECTIVENESS_AT))
    idx = np.argsort(y_prob)[::-1]
    top_labels = np.asarray(y_true)[idx[:top_k]]
    return float(top_labels.sum() / total_pos)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Return F1 (at 0.5), ROC-AUC, PR-AUC, CE@20."""
    y_true = np.asarray(y_true)
    y_pred = (y_prob >= CLASSIFICATION_THRESHOLD).astype(int)
    if y_pred.sum() < 1 or y_true.sum() < 1:
        return {"f1": 0.0, "roc_auc": 0.5, "pr_auc": 0.0, "ce_at_20": 0.0}
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "ce_at_20": _recall_at_top_20(y_true, y_prob),
    }


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
def load_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, project_id) from dataset_final.parquet.

    Uses the 27 feature columns declared in config.ALL_FEATURES in that
    exact order so train and inference paths agree.
    """
    df = pd.read_parquet(PROCESSED_DATA_DIR / "dataset_final.parquet")
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"dataset_final.parquet missing expected features: {missing}")
    X = df[list(ALL_FEATURES)].copy()
    y = df[LABEL_COL].astype(int)
    proj = df["project_id"]
    return X, y, proj


# ---------------------------------------------------------------------------
# K-fold driver with threshold optimisation
# ---------------------------------------------------------------------------
@dataclass
class FoldResult:
    model: str
    fold: int
    n_train: int
    n_test: int
    n_pos_test: int
    metrics: dict[str, float] = field(default_factory=dict)


def _fit_and_score_fold(
    model_name: str,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_te: pd.DataFrame,
    y_te: np.ndarray,
    params: Optional[dict[str, Any]],
) -> dict[str, float]:
    """Fit on (X_tr, y_tr); return metric battery on (X_te, y_te) at threshold 0.5."""
    est = _make_model(model_name, params, y_train=y_tr)
    if est is None:
        raise RuntimeError(f"Model {model_name!r} not available (import failed)")
    est.fit(X_tr, y_tr)
    if hasattr(est, "predict_proba"):
        test_probs = est.predict_proba(X_te)[:, 1]
    else:
        test_probs = est.decision_function(X_te)
    return compute_metrics(y_te, test_probs)


def stratified_kfold_cv(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = CV_FOLDS,
    *,
    params: Optional[dict[str, Any]] = None,
) -> list[FoldResult]:
    """Run stratified K-fold CV for one model. F1 at the standard 0.5 threshold."""
    yv = np.asarray(y, dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    out: list[FoldResult] = []
    for fold, (tr, te) in enumerate(skf.split(X, yv), start=1):
        X_tr = X.iloc[tr]
        X_te = X.iloc[te]
        y_tr = yv[tr]
        y_te = yv[te]
        try:
            metrics = _fit_and_score_fold(model_name, X_tr, y_tr, X_te, y_te, params)
        except RuntimeError:
            return []
        out.append(
            FoldResult(
                model=model_name,
                fold=fold,
                n_train=len(tr),
                n_test=len(te),
                n_pos_test=int(y_te.sum()),
                metrics=metrics,
            )
        )
    return out


def fold_results_to_frame(results: Iterable[FoldResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "model": r.model,
                "fold": r.fold,
                "n_train": r.n_train,
                "n_test": r.n_test,
                "n_pos_test": r.n_pos_test,
                "f1": r.metrics["f1"],
                "roc_auc": r.metrics["roc_auc"],
                "pr_auc": r.metrics["pr_auc"],
                "ce_at_20": r.metrics["ce_at_20"],
            }
        )
    return pd.DataFrame(rows)


def summarize_by_model(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold rows into mean +/- std per model."""
    metric_cols = ["f1", "roc_auc", "pr_auc", "ce_at_20"]
    grp = fold_df.groupby("model")[metric_cols]
    means = grp.mean().add_suffix("_mean")
    stds = grp.std().add_suffix("_std")
    return pd.concat([means, stds], axis=1).reset_index()
