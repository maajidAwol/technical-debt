"""
Feature-group ablation analysis.

Quantifies how much predictive power each family of features contributes
to each label variant. Three training regimes are compared for every
(variant, group) combination:

- **Only this group** - isolates the intrinsic predictive power of the
  group.
- **All features except this group** - reveals the unique marginal
  contribution (a drop from the full-model baseline).
- **All features** (baseline) - printed once per variant for reference.

The groups are defined semantically rather than by column prefix so that
new features added later still map cleanly to a group.

References
----------
- Zimmermann et al. (2007). Predicting defects for Eclipse. PROMISE.
- Rahman, D'Souza, and Devanbu (2013). Sample size vs. bias. MSR.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.train import (  # noqa: E402
    KEY_COLS,
    LABEL_COL,
    DROP_FOR_CONSEQUENCE,
    fold_results_to_frame,
    load_variant_matrix,
    stratified_kfold_cv,
)


METRIC_COLS = ["precision", "recall", "f1", "roc_auc", "pr_auc", "mcc", "ce_at_20"]


# Feature-group definitions. Names are matched against column names in
# ``dataset_{variant}.parquet``. Any feature not matched is grouped as
# ``other`` so the ablation is exhaustive.
GROUPS: dict[str, list[str]] = {
    "static_sonar": [
        "n_issues_open",
        "n_blocker",
        "n_critical",
        "n_major",
        "n_minor",
        "n_info",
        "n_code_smell",
        "n_bug",
        "n_vulnerability",
        "total_debt_minutes",
        "total_effort_minutes",
        "n_distinct_rules",
        "max_severity_rank",
        "issue_density",
        "debt_per_loc",
        "pseudo_ncloc_at_t",
    ],
    "historical": [
        "total_commits_pre",
        "total_contributors_pre",
        "code_added_pre",
        "code_removed_pre",
        "code_churn_pre",
        "avg_change_size_pre",
        "max_single_commit_churn_pre",
        "std_change_size_pre",
        "recent_churn_30d_pre",
        "recent_churn_90d_pre",
        "recent_commits_30d_pre",
        "recent_commits_90d_pre",
        "file_age_days_at_snapshot",
        "days_since_last_change_at_snapshot",
        "ownership_ratio_pre",
    ],
    "project_context": [
        # Populated dynamically below - every column starting with "project_"
    ],
    "cocg": [
        # Populated dynamically below - every column starting with "cocg_"
    ],
    "prior_defect": [
        # Populated dynamically below - every column matching the pre-snapshot
        # bug-fix / SZZ / Jira prefixes (see priordefect_features.py).
    ],
}


def _resolve_groups(available_cols: Iterable[str]) -> dict[str, list[str]]:
    """Expand the dynamic placeholders and drop missing cols."""
    avail = set(available_cols)
    resolved = {
        name: sorted(c for c in cols if c in avail)
        for name, cols in GROUPS.items()
        if name not in {"project_context", "cocg", "prior_defect"}
    }
    resolved["project_context"] = sorted(
        c for c in avail if c.startswith("project_") or c == "has_project_context"
    )
    resolved["cocg"] = sorted(c for c in avail if c.startswith("cocg_"))
    resolved["prior_defect"] = sorted(
        c
        for c in avail
        if (
            c.startswith("bugfix_commits_pre")
            or c.startswith("szz_inducing_pre")
            or c.startswith("linked_jira_")
            or c == "time_since_last_bugfix_days"
            or c == "bug_density_pre"
        )
    )
    assigned = {c for cols in resolved.values() for c in cols}
    other = sorted(c for c in avail if c not in assigned)
    if other:
        resolved["other"] = other
    return resolved


def _evaluate(X: pd.DataFrame, y: pd.Series, variant: str, model_name: str) -> dict[str, float]:
    fold_results = stratified_kfold_cv(variant, model_name, X, y, n_splits=10)
    df = fold_results_to_frame(fold_results)
    return df[METRIC_COLS].mean().to_dict()


def run_ablation(variant: str, model_name: str = "lightgbm") -> pd.DataFrame:
    """Run the group ablation for one variant and return a tidy table."""
    X, y, _ = load_variant_matrix(variant)
    groups = _resolve_groups(X.columns)
    rows = []

    t0 = time.time()
    base_metrics = _evaluate(X, y, variant, model_name)
    rows.append(
        {
            "variant": variant,
            "group": "all",
            "mode": "all_features",
            "n_features": X.shape[1],
            **{f"{k}_mean": round(v, 4) for k, v in base_metrics.items()},
            "elapsed_s": round(time.time() - t0, 2),
        }
    )

    for name, cols in groups.items():
        if not cols:
            continue

        t1 = time.time()
        metrics_only = _evaluate(X[cols], y, variant, model_name)
        rows.append(
            {
                "variant": variant,
                "group": name,
                "mode": "only_this_group",
                "n_features": len(cols),
                **{f"{k}_mean": round(v, 4) for k, v in metrics_only.items()},
                "elapsed_s": round(time.time() - t1, 2),
            }
        )

        t1 = time.time()
        remaining = [c for c in X.columns if c not in cols]
        if remaining:
            metrics_wo = _evaluate(X[remaining], y, variant, model_name)
            rows.append(
                {
                    "variant": variant,
                    "group": name,
                    "mode": "leave_out_this_group",
                    "n_features": len(remaining),
                    **{f"{k}_mean": round(v, 4) for k, v in metrics_wo.items()},
                    "elapsed_s": round(time.time() - t1, 2),
                }
            )
    return pd.DataFrame(rows)
