"""
Feature-importance analysis via SHAP and permutation importance.

For each label variant, trains a single LightGBM model on the whole
dataset (stratified 80/20 split for hold-out importance scoring) and
computes:

- **TreeSHAP values** - accurate Shapley values for tree ensembles, used
  to produce the SHAP summary plot and per-feature mean absolute SHAP
  value.
- **Permutation importance** - model-agnostic drop in ROC-AUC when the
  column is shuffled; provides a validation metric against SHAP.

Both rankings are saved as CSV tables; the raw SHAP matrix is persisted
for use by the plotting module.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import RANDOM_STATE  # noqa: E402
from src.models.train import _make_model, load_variant_matrix  # noqa: E402


def compute_importance(
    variant: str,
    model_name: str = "lightgbm",
    n_permutation_repeats: int = 5,
    shap_background_size: int = 1000,
) -> dict[str, pd.DataFrame]:
    """Compute SHAP + permutation importance for one variant.

    Returns a dict with:
    - ``shap_summary`` - DataFrame indexed by feature with mean absolute
      SHAP value, mean SHAP value and signed contribution sign.
    - ``permutation`` - DataFrame with mean and std permutation
      importance per feature (drop in ROC-AUC).
    - ``shap_matrix`` - raw SHAP values (numpy array) and corresponding
      X_sample DataFrame for plotting.
    """
    X, y, proj = load_variant_matrix(variant)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    est = _make_model(model_name)
    est.fit(X_train, y_train)

    # ---- SHAP ----
    import shap

    # Use a background sample for expected value computation to bound runtime
    bg_size = min(shap_background_size, len(X_train))
    bg = X_train.sample(n=bg_size, random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(est, data=bg, feature_perturbation="interventional")
    # Limit to the test split for plotting (faster, consistent with other stats)
    shap_values = explainer.shap_values(X_test, check_additivity=False)
    if isinstance(shap_values, list):
        # binary classifier with per-class SHAP - take class-1 contribution
        shap_values = shap_values[1]

    shap_values = np.asarray(shap_values)

    # Legacy TreeExplainer can return (n_samples, n_features, 2) for binary
    if shap_values.ndim == 3:
        shap_values = shap_values[..., 1]

    shap_abs = np.abs(shap_values).mean(axis=0)
    shap_mean = shap_values.mean(axis=0)
    shap_summary = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "mean_abs_shap": shap_abs,
                "mean_shap": shap_mean,
                "sign": np.where(shap_mean >= 0, "+", "-"),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # ---- Permutation importance ----
    pi = permutation_importance(
        est, X_test, y_test,
        n_repeats=n_permutation_repeats,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=1,
    )
    perm = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "perm_importance_mean": pi.importances_mean,
                "perm_importance_std": pi.importances_std,
            }
        )
        .sort_values("perm_importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "shap_summary": shap_summary,
        "permutation": perm,
        "shap_values": shap_values,
        "X_sample": X_test.reset_index(drop=True),
    }
