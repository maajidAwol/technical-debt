"""
Hyperparameter tuning for the within-project pipeline (Stage 7b).

Implements an Optuna-based random/TPE search optimising mean PR-AUC on
an inner stratified K-fold split of the training data. We use PR-AUC
because all three label variants are imbalanced (positive rate 1-15%);
Saito and Rehmsmeier (2015) show that ROC-AUC is overly optimistic in
that regime.

Search spaces are conservative on purpose - the goal is to validate
that the headline ranking from Stage 7 is *not* an artefact of bad
defaults, not to win a Kaggle leaderboard. Per the proposal Section
2.3 / 3.5 we declare these spaces in advance; results in
``results/tables/tuned_params.json`` make them auditable.

References
----------
- Saito, T., Rehmsmeier, M. (2015). PR vs ROC for imbalanced data.
  PLOS ONE 10(3).
- Bergstra, J., Bengio, Y. (2012). Random search for hyperparameter
  optimization. JMLR.
- Akiba, T., et al. (2019). Optuna. KDD.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    RANDOM_STATE,
    TUNING_INNER_CV_FOLDS,
    TUNING_TRIALS,
)
from src.models.train import _make_model  # noqa: E402


def _suggest_params(trial, model_name: str) -> dict[str, Any]:
    """Per-model search space. Kept small (<=6 hyperparameters) so
    ``TUNING_TRIALS=30`` covers a meaningful slice of each space.
    """
    name = model_name.lower()
    if name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2"]),
            "solver": "lbfgs",
        }
    if name == "decision_tree":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 4, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.3, 0.5]
            ),
        }
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        }
    if name == "svm":
        return {
            "C": trial.suggest_float("C", 1e-2, 100.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
    raise ValueError(f"No tuning space defined for model: {model_name}")


def _objective_factory(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int,
):
    """Return an Optuna objective that mean PR-AUC over inner folds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial) -> float:
        params = _suggest_params(trial, model_name)
        scores = []
        for tr, te in skf.split(X, y):
            est = _make_model(model_name, params)
            if est is None:
                return float("-inf")
            X_tr = X.iloc[tr]
            X_te = X.iloc[te]
            est.fit(X_tr, y[tr])
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


def tune_model(
    variant: str,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = TUNING_TRIALS,
    inner_splits: int = TUNING_INNER_CV_FOLDS,
    timeout_s: Optional[int] = None,
) -> dict[str, Any]:
    """Run an Optuna study; return ``{best_params, best_value, n_trials, elapsed_s}``.

    ``timeout_s`` is forwarded to ``study.optimize`` for hard-stopping
    long searches; ``None`` means no timeout.
    """
    import optuna  # local import - keeps cold-start cheap

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    objective = _objective_factory(model_name, X, y.values, inner_splits)

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s, show_progress_bar=False)
    elapsed = time.time() - t0

    best = study.best_trial
    return {
        "variant": variant,
        "model": model_name,
        "best_params": dict(best.params),
        "best_pr_auc": float(best.value) if best.value is not None else float("nan"),
        "n_trials_completed": len(study.trials),
        "n_trials_requested": n_trials,
        "inner_splits": inner_splits,
        "elapsed_s": round(elapsed, 2),
    }
