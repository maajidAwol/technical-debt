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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
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
    CALIBRATION_INNER_CV_FOLDS,
    COST_EFFECTIVENESS_AT,
    CV_FOLDS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TABLES_DIR,
)


KEY_COLS = ("project_id", "basename")
LABEL_COL = "is_high_risk"
DROP_FOR_CONSEQUENCE = ("risk_score",)


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------
def _merged_params(defaults: dict[str, Any], overrides: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Merge optional ``overrides`` over ``defaults`` (overrides win).

    Keys in ``overrides`` that are not understood by the estimator are
    passed through unchanged - sklearn will raise a clear error if they
    are invalid, which is a desirable failure mode for a tuning loop.
    """
    out = dict(defaults)
    if overrides:
        out.update(overrides)
    return out


def _make_model(name: str, params: Optional[dict[str, Any]] = None):
    """Build an sklearn estimator by name, using sane defaults from config.

    Optional ``params`` (dict) is merged over the defaults *of the
    underlying estimator*. For pipeline-wrapped models (LR, SVM) the
    overrides apply to the trailing classifier step, not the scaler.

    XGBoost / LightGBM are optional imports: if not installed, those
    models are silently skipped.
    """
    name = name.lower()
    if name == "decision_tree":
        defaults = {"random_state": RANDOM_STATE, "class_weight": "balanced"}
        return DecisionTreeClassifier(**_merged_params(defaults, params))
    if name == "random_forest":
        defaults = {
            "n_estimators": 200,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
            "n_jobs": -1,
        }
        return RandomForestClassifier(**_merged_params(defaults, params))
    if name == "logistic_regression":
        defaults = {
            "max_iter": 5000,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "solver": "lbfgs",
        }
        return Pipeline(
            [
                ("scale", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(**_merged_params(defaults, params))),
            ]
        )
    if name == "svm":
        defaults = {
            "kernel": "rbf",
            "probability": True,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        }
        return Pipeline(
            [
                ("scale", StandardScaler(with_mean=False)),
                ("clf", SVC(**_merged_params(defaults, params))),
            ]
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            return None
        defaults = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": RANDOM_STATE,
            "eval_metric": "logloss",
            "n_jobs": -1,
        }
        return XGBClassifier(**_merged_params(defaults, params))
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            return None
        defaults = {
            "n_estimators": 200,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
            "n_jobs": -1,
            "verbose": -1,
        }
        return LGBMClassifier(**_merged_params(defaults, params))
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


def _expected_calibration_error(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> float:
    """Equal-width-bin Expected Calibration Error (Naeini et al. 2015).

    Lower is better; 0 indicates perfect calibration. Used for
    calibration diagnostics in :func:`calibrated_kfold_cv`.
    """
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_score, bins[1:-1], right=False)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        bin_conf = float(np.mean(y_score[mask]))
        bin_acc = float(np.mean(y_true[mask]))
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


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


def _resample_smote(X_tr: pd.DataFrame, y_tr: np.ndarray, random_state: int):
    """SMOTE over-sample the minority class on the *training fold only*.

    Imported lazily so unit tests / smoke imports don't pay the cost.
    """
    from imblearn.over_sampling import SMOTE  # local import

    n_min = int(np.sum(y_tr == 1))
    # SMOTE requires at least k_neighbors+1 minority samples; drop k
    # automatically for small folds.
    k = min(5, max(1, n_min - 1))
    sm = SMOTE(random_state=random_state, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X_tr, y_tr)
    return X_res, y_res


def stratified_kfold_cv(
    variant: str,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = CV_FOLDS,
    *,
    params: Optional[dict[str, Any]] = None,
    use_smote: bool = False,
    persist_predictions: bool = False,
    project_id: Optional[pd.Series] = None,
) -> list[FoldResult]:
    """Run stratified K-fold for one ``(variant, model)`` pair.

    Parameters
    ----------
    variant, model_name :
        Label variant and model identifier.
    X, y :
        Feature matrix and label vector.
    n_splits :
        Number of stratified folds.
    params :
        Optional hyperparameter overrides forwarded to :func:`_make_model`.
        Used by the Stage 7b tuning driver (``scripts/07b_tune.py``).
    use_smote :
        If True, SMOTE-oversample the *training fold* only (test fold
        is never resampled). Replaces - not augments - the
        ``class_weight="balanced"`` default; SMOTE generates synthetic
        minority samples instead.
    persist_predictions :
        If True, returns of ``stratified_kfold_cv`` additionally append
        per-row predictions to ``TABLES_DIR/within_project_predictions.parquet``.
        Each row carries ``(variant, model, fold, project_id, basename_idx,
        y_true, y_score)``.
    project_id :
        Optional ``project_id`` series aligned with X/y. Required only
        when ``persist_predictions=True``.

    Returns
    -------
    list[FoldResult]
        Per-fold metric battery. Empty list if the estimator is
        unavailable (e.g. xgboost/lightgbm not installed).
    """
    est = _make_model(model_name, params)
    if est is None:
        return []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    out: list[FoldResult] = []
    yv = y.values
    pred_rows: list[dict[str, Any]] = []
    for fold, (tr, te) in enumerate(skf.split(X, yv), start=1):
        est_f = _make_model(model_name, params)
        X_tr = X.iloc[tr]
        X_te = X.iloc[te]
        y_tr = yv[tr]
        if use_smote:
            try:
                X_tr, y_tr = _resample_smote(X_tr, y_tr, random_state=RANDOM_STATE)
            except ValueError:
                # Fold has too few minority samples for SMOTE; fall
                # back to original training fold (still class-weighted).
                pass
        est_f.fit(X_tr, y_tr)
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
        if persist_predictions:
            pid_arr = (
                project_id.iloc[te].values
                if project_id is not None
                else np.array(["__unknown__"] * len(te))
            )
            for k, row_idx in enumerate(te):
                pred_rows.append(
                    {
                        "variant": variant,
                        "model": model_name,
                        "fold": fold,
                        "row_idx": int(row_idx),
                        "project_id": str(pid_arr[k]),
                        "y_true": int(yv[te][k]),
                        "y_score": float(proba[k]),
                        "y_pred": int(pred[k]),
                        "use_smote": bool(use_smote),
                    }
                )

    if persist_predictions and pred_rows:
        _append_predictions_parquet(
            TABLES_DIR / "within_project_predictions.parquet",
            pd.DataFrame(pred_rows),
        )

    return out


def _append_predictions_parquet(path: Path, df_new: pd.DataFrame) -> None:
    """Append ``df_new`` to a parquet file, creating it if needed.

    PyArrow doesn't natively support append, so we read-modify-write.
    Fold-level prediction tables are small enough (~100 KB per
    variant/model) that this is fine.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            df_out = pd.concat([existing, df_new], ignore_index=True)
        except Exception:
            df_out = df_new
    else:
        df_out = df_new
    df_out.to_parquet(path, index=False)


def calibrated_kfold_cv(
    variant: str,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = CV_FOLDS,
    *,
    method: str = "isotonic",
    params: Optional[dict[str, Any]] = None,
) -> tuple[list[FoldResult], pd.DataFrame]:
    """Stratified K-fold with probability calibration.

    Wraps the base estimator in
    :class:`sklearn.calibration.CalibratedClassifierCV` using a
    ``cv=CALIBRATION_INNER_CV_FOLDS`` inner split on the training
    fold. The calibration data is therefore disjoint from the outer
    test fold, so the held-out metric is honest.

    Parameters
    ----------
    method :
        ``"platt"`` (sigmoid) or ``"isotonic"``. Platt assumes a
        sigmoid distortion; isotonic is non-parametric and usually
        better when the base model is a tree ensemble.

    Returns
    -------
    (results, calib_df) :
        ``results`` mirrors :func:`stratified_kfold_cv`. ``calib_df``
        carries per-fold calibration diagnostics (Brier, NLL, ECE)
        plus a long-form dataset of (y_true, y_score) used to draw
        reliability diagrams.
    """
    sk_method = "sigmoid" if method == "platt" else "isotonic"
    base = _make_model(model_name, params)
    if base is None:
        return [], pd.DataFrame()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    out: list[FoldResult] = []
    calib_rows: list[dict[str, Any]] = []
    yv = y.values
    for fold, (tr, te) in enumerate(skf.split(X, yv), start=1):
        base_f = _make_model(model_name, params)
        cal = CalibratedClassifierCV(base_f, method=sk_method, cv=CALIBRATION_INNER_CV_FOLDS)
        X_tr = X.iloc[tr]
        X_te = X.iloc[te]
        cal.fit(X_tr, yv[tr])
        proba = cal.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = _metric_row(yv[te], pred, proba)
        # Calibration-specific diagnostics
        brier = float(brier_score_loss(yv[te], proba))
        try:
            nll = float(log_loss(yv[te], np.clip(proba, 1e-7, 1 - 1e-7)))
        except ValueError:
            nll = float("nan")
        ece = _expected_calibration_error(yv[te], proba)
        metrics_full = {**metrics, "brier": brier, "nll": nll, "ece": ece, "method": method}
        out.append(
            FoldResult(
                variant=variant,
                model=model_name,
                fold=fold,
                n_train=len(tr),
                n_test=len(te),
                n_pos_test=int(yv[te].sum()),
                metrics=metrics_full,
            )
        )
        calib_rows.append(
            {
                "variant": variant,
                "model": model_name,
                "method": method,
                "fold": fold,
                "brier": brier,
                "nll": nll,
                "ece": ece,
                "y_true": yv[te].tolist(),
                "y_score": proba.tolist(),
            }
        )
    return out, pd.DataFrame(calib_rows)


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


def summarize(fold_df: pd.DataFrame, group_cols: Iterable[str] = ("variant", "model")) -> pd.DataFrame:
    """Aggregate per-fold results into mean +/- std per ``group_cols``.

    ``group_cols`` defaults to ``("variant", "model")`` for backwards
    compatibility but can be widened (e.g. include ``"method"`` when
    summarising calibrated runs across Platt/isotonic).
    """
    candidate_cols = {
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "mcc",
        "ce_at_20",
        "brier",
        "nll",
        "ece",
    }
    metric_cols = [c for c in fold_df.columns if c in candidate_cols]
    group_cols = list(group_cols)
    grp = fold_df.groupby(group_cols)[metric_cols]
    means = grp.mean().add_suffix("_mean")
    stds = grp.std().add_suffix("_std")
    return pd.concat([means, stds], axis=1).reset_index()
