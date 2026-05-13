"""
Optuna hyperparameter tuning for the 4-model pipeline (Stage 7b).

Search spaces match Part 5 of the rebuild spec exactly. Objective:
mean PR-AUC over a stratified ``TUNING_INNER_CV_FOLDS``-fold inner CV
disjoint from the outer 10-fold validation in 07_train.py.

PR-AUC is the recommended objective for class-imbalanced binary
classification (Saito & Rehmsmeier 2015); ROC-AUC is optimistic when
positive rate < 25%.

Logistic regression uses a small grid (``C in {0.01, 0.1, 1, 10, 100}``)
per spec; tree ensembles use Optuna TPE with the spec's exact ranges.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    RANDOM_STATE,
    TUNING_INNER_CV_FOLDS,
    TUNING_TRIALS,
)
from src.models.train import _make_model  # noqa: E402


# ---------------------------------------------------------------------------
# Search spaces (verbatim from Part 5)
# ---------------------------------------------------------------------------
def _suggest_params(trial, model_name: str) -> dict[str, Any]:
    name = model_name.lower()
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.3]),
        }
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        }
    raise ValueError(f"No Optuna space defined for model: {model_name}")


# ---------------------------------------------------------------------------
# Inner-CV PR-AUC objective
# ---------------------------------------------------------------------------
def _objective_factory(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int,
    groups: Optional[np.ndarray] = None,
):
    """Build an Optuna PR-AUC objective.

    If ``groups`` is provided and has fewer unique values than rows,
    uses ``StratifiedGroupKFold`` so all rows sharing a group label
    land in the same fold (no file-id leakage for multi-snapshot data).
    """
    if groups is not None and len(np.unique(groups)) < len(y):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        split_iter_fn = lambda: splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        split_iter_fn = lambda: splitter.split(X, y)

    def objective(trial) -> float:
        params = _suggest_params(trial, model_name)
        scores: list[float] = []
        for tr, te in split_iter_fn():
            X_tr = X.iloc[tr]
            X_te = X.iloc[te]
            y_tr = y[tr]
            est = _make_model(model_name, params, y_train=y_tr)
            if est is None:
                return float("-inf")
            est.fit(X_tr, y_tr)
            if hasattr(est, "predict_proba"):
                proba = est.predict_proba(X_te)[:, 1]
            else:
                proba = est.decision_function(X_te)
            try:
                scores.append(float(average_precision_score(y[te], proba)))
            except ValueError:
                scores.append(0.0)
        return float(np.mean(scores))

    return objective


# ---------------------------------------------------------------------------
# Public tuner
# ---------------------------------------------------------------------------
def tune_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = TUNING_TRIALS,
    inner_splits: int = TUNING_INNER_CV_FOLDS,
    timeout_s: Optional[int] = None,
    groups: Optional[pd.Series] = None,
) -> dict[str, Any]:
    """Run an Optuna study (or GridSearchCV for LR); return best params + diagnostics."""
    name = model_name.lower()
    yv = np.asarray(y, dtype=int)
    groups_arr = groups.values if groups is not None else None

    if name == "logistic_regression":
        return _tune_logistic_grid(X, yv, inner_splits, groups=groups_arr)

    import optuna  # local import; cheap cold-start

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    objective = _objective_factory(model_name, X, yv, inner_splits, groups=groups_arr)

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s, show_progress_bar=False)
    elapsed = time.time() - t0

    best = study.best_trial
    return {
        "model": model_name,
        "best_params": dict(best.params),
        "best_pr_auc": float(best.value) if best.value is not None else float("nan"),
        "n_trials_completed": len(study.trials),
        "n_trials_requested": n_trials,
        "inner_splits": inner_splits,
        "elapsed_s": round(elapsed, 2),
        "method": "optuna_tpe",
    }


def _tune_logistic_grid(
    X: pd.DataFrame,
    y: np.ndarray,
    inner_splits: int,
    groups: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Logistic regression via GridSearchCV on C in {0.01, 0.1, 1, 10, 100} per spec."""
    t0 = time.time()
    est = _make_model("logistic_regression")
    # _make_model wraps LR in a Pipeline named ("scaler", "clf"); GridSearchCV
    # needs "clf__C" to address the inner classifier.
    param_grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    if groups is not None and len(np.unique(groups)) < len(y):
        cv = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        est,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        refit=False,
    )
    gs.fit(X, y, groups=groups)
    elapsed = time.time() - t0
    best_c = float(gs.best_params_["clf__C"])
    return {
        "model": "logistic_regression",
        "best_params": {"C": best_c},
        "best_pr_auc": float(gs.best_score_),
        "n_trials_completed": len(gs.cv_results_["params"]),
        "n_trials_requested": len(gs.cv_results_["params"]),
        "inner_splits": inner_splits,
        "elapsed_s": round(elapsed, 2),
        "method": "grid_search",
    }
