"""
Leave-One-Project-Out (LOPO) cross-project validation.

For each label variant and each model, this module runs N = 22 folds
where the test set is one held-out project and the training set is the
union of the other 21 projects. This is the strongest generalization
test for software-engineering ML: it answers "if I train on 21 projects
and deploy on a brand-new project, what performance should I expect?"

Public API
----------
- ``lopo_cv(variant, model_name)`` - return a DataFrame with one row
  per held-out project containing the full metric battery.
- ``lopo_summary(folds_df)`` - aggregate per-variant, per-model means
  and standard deviations across held-out projects.

Metrics are identical to the within-project module so results are
directly comparable (difference = generalization gap).

References
----------
- Zimmermann et al. (2009). Cross-project defect prediction. FSE.
- Herbold et al. (2018). A comparative study to benchmark cross-project
  defect prediction approaches. IEEE TSE 44(9).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    COST_EFFECTIVENESS_AT,
    PROCESSED_DATA_DIR,
)
from src.models.train import (  # noqa: E402
    KEY_COLS,
    LABEL_COL,
    _cost_effectiveness_at_k,
    _make_model,
    _metric_row,
    load_variant_matrix,
)


@dataclass
class LopoFoldResult:
    variant: str
    model: str
    held_out_project: str
    n_train: int
    n_test: int
    n_pos_test: int
    metrics: dict[str, float] = field(default_factory=dict)


def lopo_cv(
    variant: str,
    model_name: str,
) -> list[LopoFoldResult]:
    """Run LOPO CV for one ``(variant, model)`` pair.

    Returns one result per held-out project. Projects where the
    training set contains no positive labels are skipped (degenerate).
    """
    X, y, proj = load_variant_matrix(variant)
    projects = sorted(proj.unique())
    out: list[LopoFoldResult] = []

    for held_out in projects:
        te_mask = (proj == held_out).values
        tr_mask = ~te_mask

        y_tr = y.values[tr_mask]
        y_te = y.values[te_mask]
        # Require both classes in training and at least one positive in test
        if len(set(y_tr)) < 2 or y_te.sum() == 0:
            continue

        est = _make_model(model_name)
        if est is None:
            return []

        X_tr = X.iloc[tr_mask]
        X_te = X.iloc[te_mask]
        est.fit(X_tr, y_tr)
        if hasattr(est, "predict_proba"):
            proba = est.predict_proba(X_te)[:, 1]
        else:
            proba = est.decision_function(X_te)
        pred = (proba >= 0.5).astype(int)
        metrics = _metric_row(y_te, pred, proba)

        out.append(
            LopoFoldResult(
                variant=variant,
                model=model_name,
                held_out_project=held_out,
                n_train=int(tr_mask.sum()),
                n_test=int(te_mask.sum()),
                n_pos_test=int(y_te.sum()),
                metrics=metrics,
            )
        )
    return out


def lopo_results_to_frame(results: Iterable[LopoFoldResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "variant": r.variant,
                "model": r.model,
                "held_out_project": r.held_out_project,
                "n_train": r.n_train,
                "n_test": r.n_test,
                "n_pos_test": r.n_pos_test,
                **r.metrics,
            }
        )
    return pd.DataFrame(rows)


def lopo_summary(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate LOPO per-project metrics into mean +/- std per ``(variant, model)``."""
    metric_cols = [
        c for c in fold_df.columns
        if c in {"precision", "recall", "f1", "roc_auc", "pr_auc", "mcc", "ce_at_20"}
    ]
    grp = fold_df.groupby(["variant", "model"])[metric_cols]
    means = grp.mean().add_suffix("_mean")
    stds = grp.std().add_suffix("_std")
    n = grp.count().iloc[:, :1].rename(columns={metric_cols[0]: "n_projects"})
    return pd.concat([n, means, stds], axis=1).reset_index()
