"""
Within-project stratified K-fold training and evaluation.

For a single ``(variant, model)`` pair this module:

1. Loads the corresponding ``dataset_{variant}.parquet`` from Stage 6.
2. Separates numeric feature columns from metadata and labels (dropping
   ``risk_score`` for the consequence variant so that the score that
   defined the label is not used as an input feature).
3. Runs stratified K-fold cross-validation (``config.CV_FOLDS``), with
   optional class-imbalance handling via class-weights (default) or
   SMOTE oversampling (``use_smote=True``).
4. Computes a full metric battery per fold: precision, recall, F1,
   ROC-AUC, PR-AUC (average precision), MCC, and Cost-Effectiveness at
   top-20 percent (``CE@20``).

The driver script ``scripts/07_train.py`` iterates over all variants and
models and writes a tidy results table.

References
----------
- Kamei et al. (2013). Just-in-time quality assurance. IEEE TSE.
- Menzies et al. (2007). Data mining static code attributes. IEEE TSE.
- Saito and Rehmsmeier (2015). PR vs ROC evaluation for imbalanced data.
  PLOS ONE 10(3).
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
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    COST_EFFECTIVENESS_AT,
    CV_FOLDS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)


KEY_COLS = ("project_id", "basename")
LABEL_COL = "is_high_risk"
DROP_FOR_CONSEQUENCE = ("risk_score",)


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------
def _make_model(name: str):
    """Build an sklearn estimator by name, using sane defaults from config.

    XGBoost / LightGBM are optional imports: if not installed, those
    models are silently skipped.
    """
    name = name.lower()
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )
    if name == "logistic_regression":
        return Pipeline(
            [
                ("scale", StandardScaler(with_mean=False)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    if name == "svm":
        return Pipeline(
            [
                ("scale", StandardScaler(with_mean=False)),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            return None
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        )
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            return None
        return LGBMClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
            verbose=-1,
        )
    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Metric battery
# ---------------------------------------------------------------------------
def _cost_effectiveness_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float) -> float:
    """Fraction of positives captured by the top-``k``% predicted-risk files.

    A prioritization-oriented metric: if a tool inspects only the top
    ``k``% highest-scoring files, CE@k is the recall achieved in that
    budget - used e.g. in Menzies et al. 2007.
    """
    n = len(y_true)
    total_pos = int(y_true.sum())
    if n == 0 or total_pos == 0:
        return 0.0
    budget = max(1, int(np.ceil(n * k)))
    top_idx = np.argsort(-y_score)[:budget]
    return float(y_true[top_idx].sum() / total_pos)


def _metric_row(y_true, y_pred, y_score) -> dict[str, float]:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan"),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "ce_at_20": _cost_effectiveness_at_k(y_true, y_score, COST_EFFECTIVENESS_AT),
    }


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
def load_variant_matrix(variant: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return ``(X, y, project_id)`` for the specified label variant.

    - ``X`` - numeric feature matrix only (non-numeric columns dropped).
    - ``y`` - ``is_high_risk`` as int.
    - ``project_id`` - group column for group-aware validation.
    """
    df = pd.read_parquet(PROCESSED_DATA_DIR / f"dataset_{variant}.parquet")
    drop_cols = list(KEY_COLS) + [LABEL_COL]
    if variant == "consequence":
        for c in DROP_FOR_CONSEQUENCE:
            if c in df.columns:
                drop_cols.append(c)
    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include="number")
    y = df[LABEL_COL].astype(int)
    proj = df["project_id"]
    return X, y, proj


# ---------------------------------------------------------------------------
# K-fold driver
# ---------------------------------------------------------------------------
@dataclass
class FoldResult:
    variant: str
    model: str
    fold: int
    n_train: int
    n_test: int
    n_pos_test: int
    metrics: dict[str, float] = field(default_factory=dict)


def stratified_kfold_cv(
    variant: str,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = CV_FOLDS,
) -> list[FoldResult]:
    """Run stratified K-fold for one ``(variant, model)`` pair.

    Returns a list of per-fold results containing the full metric
    battery.
    """
    est = _make_model(model_name)
    if est is None:
        return []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    out: list[FoldResult] = []
    yv = y.values
    for fold, (tr, te) in enumerate(skf.split(X, yv), start=1):
        est_f = _make_model(model_name)
        X_tr = X.iloc[tr]
        X_te = X.iloc[te]
        est_f.fit(X_tr, yv[tr])
        if hasattr(est_f, "predict_proba"):
            proba = est_f.predict_proba(X_te)[:, 1]
        else:
            proba = est_f.decision_function(X_te)
        pred = (proba >= 0.5).astype(int)
        metrics = _metric_row(yv[te], pred, proba)
        out.append(
            FoldResult(
                variant=variant,
                model=model_name,
                fold=fold,
                n_train=len(tr),
                n_test=len(te),
                n_pos_test=int(yv[te].sum()),
                metrics=metrics,
            )
        )
    return out


def fold_results_to_frame(results: Iterable[FoldResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "variant": r.variant,
            "model": r.model,
            "fold": r.fold,
            "n_train": r.n_train,
            "n_test": r.n_test,
            "n_pos_test": r.n_pos_test,
            **r.metrics,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold results into mean +/- std per (variant, model)."""
    metric_cols = [c for c in fold_df.columns if c in {"precision", "recall", "f1", "roc_auc", "pr_auc", "mcc", "ce_at_20"}]
    grp = fold_df.groupby(["variant", "model"])[metric_cols]
    means = grp.mean().add_suffix("_mean")
    stds = grp.std().add_suffix("_std")
    return pd.concat([means, stds], axis=1).reset_index()
