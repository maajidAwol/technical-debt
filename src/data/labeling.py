"""
Three-variant labeling for High-Risk Technical Debt.

Implements the three mutually-disjoint label definitions compared in the
approved MSc proposal (Section 3.4):

1. **Consequence-oriented (primary)** - top ``P%`` per project by a weighted
   risk score combining (a) future bug-fix commits, (b) future churn and
   (c) future SZZ fault-inducing events in the observation window
   ``(t, t + W]``.
2. **Severity-based baseline** - positive iff the file has any SonarQube
   BLOCKER or CRITICAL issue open at snapshot ``t``. This is the
   conventional static-analysis view of "high-risk" debt.
3. **SZZ defect-oriented baseline** - positive iff the file is touched by
   at least one SZZ fault-inducing commit that is fixed inside the
   observation window (classical SZZ-defect-prediction target).

Each variant returns a DataFrame indexed by ``(project_id, basename)`` with
a binary ``is_high_risk`` column plus the numeric signals used to derive
it (kept for diagnostics and sensitivity analysis).

References
----------
- Lenarduzzi, V., et al. (2019). The Technical Debt Dataset. PROMISE.
- Kamei, Y., et al. (2013). A large-scale empirical study of just-in-time
  quality assurance. IEEE TSE 39(6).
- Tsoukalas, D., et al. (2020). Machine learning for technical debt
  identification. IEEE TSE.
- Jiang, Z., Chen, T., Zhou, Y. (2024). Graph-based technical debt
  prediction. Empir. Softw. Eng. 29.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    HIGH_RISK_PERCENTILE,
    OBSERVATION_WINDOW_MONTHS,
    RISK_SCORE_WEIGHTS,
    SEVERITY_BASELINE_LEVELS,
)
from src.data.szz import (  # noqa: E402
    basename_universe_at_snapshot,
    bugfix_touches_in_window,
    churn_in_window,
    jira_bug_commits_in_window,
    open_issues_at_snapshot,
    szz_events_in_window,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _min_max_norm(s: pd.Series) -> pd.Series:
    """Min-max normalize a numeric Series to ``[0, 1]``; constant series -> 0."""
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def _top_percentile(s: pd.Series, percentile: float) -> pd.Series:
    """Boolean mask for the top ``percentile``% values of ``s`` (ties broken by value).

    ``percentile`` is in ``[0, 100]``. If all values tie, returns all-False.
    """
    if len(s) == 0:
        return pd.Series([], dtype=bool)
    threshold = np.percentile(s, 100 - percentile)
    mask = s > threshold
    if mask.sum() == 0:
        mask = s >= threshold
    return mask


# ---------------------------------------------------------------------------
# Consequence-oriented (primary) labeling
# ---------------------------------------------------------------------------
def compute_consequence_labels(
    project_id: str,
    snapshot: pd.Timestamp,
    commits: pd.DataFrame,
    changes: pd.DataFrame,
    szz: pd.DataFrame,
    jira: Optional[pd.DataFrame] = None,
    window_months: int = OBSERVATION_WINDOW_MONTHS,
    percentile: float = HIGH_RISK_PERCENTILE,
    weights: dict = RISK_SCORE_WEIGHTS,
) -> pd.DataFrame:
    """Compute consequence-oriented labels for one project.

    The risk score is a weighted sum of three min-max-normalized components:
    ``n_bugfix_commits_future`` (weight 0.5), ``future_churn`` (weight 0.3)
    and ``n_szz_fixes_future`` (weight 0.2). Files ranked in the top
    ``percentile`` percent within the project are flagged as high-risk.

    Parameters
    ----------
    project_id :
        Which project to compute for.
    snapshot :
        Snapshot date ``t`` for this project.
    commits, changes, szz :
        Cleaned DataFrames from Stage 3.
    jira :
        Optional Jira DataFrame; if supplied an extra
        ``n_jira_bug_commits_future`` column is added for diagnostics
        (not part of the primary score).
    window_months :
        Observation window length.
    percentile :
        Fraction of files labeled positive, in ``[0, 100]``.
    weights :
        Dictionary of component weights. Keys must be a subset of
        ``{"bugfix_commits_future", "future_churn", "szz_defects_future"}``.

    Returns
    -------
    DataFrame with columns:
    ``project_id``, ``basename``, component counts, normalized components
    (``*_norm``), ``risk_score``, ``is_high_risk``.
    """
    universe = basename_universe_at_snapshot(changes, project_id, snapshot)
    if universe.empty:
        return universe.assign(is_high_risk=False)

    bf = bugfix_touches_in_window(commits, changes, project_id, snapshot, window_months)
    ch = churn_in_window(changes, project_id, snapshot, window_months)
    sz = szz_events_in_window(szz, changes, project_id, snapshot, window_months)

    df = universe.merge(bf, on="basename", how="left")
    df = df.merge(ch, on="basename", how="left")
    df = df.merge(sz, on="basename", how="left")

    if jira is not None:
        jb = jira_bug_commits_in_window(commits, jira, changes, project_id, snapshot, window_months)
        df = df.merge(jb, on="basename", how="left")
        df["n_jira_bug_commits_future"] = df["n_jira_bug_commits_future"].fillna(0).astype("int64")

    count_cols = [
        "n_bugfix_commits_future",
        "bugfix_churn_future",
        "future_churn",
        "future_add",
        "future_removed",
        "future_commits",
        "n_szz_fixes_future",
        "n_szz_inducing_past",
    ]
    for c in count_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype("int64")

    df["bugfix_norm"] = _min_max_norm(df["n_bugfix_commits_future"])
    df["churn_norm"] = _min_max_norm(df["future_churn"])
    df["szz_norm"] = _min_max_norm(df["n_szz_fixes_future"])

    w_b = weights.get("bugfix_commits_future", 0.5)
    w_c = weights.get("future_churn", 0.3)
    w_s = weights.get("szz_defects_future", 0.2)
    total_w = w_b + w_c + w_s
    if total_w == 0:
        raise ValueError("RISK_SCORE_WEIGHTS sum to zero")

    df["risk_score"] = (
        w_b * df["bugfix_norm"] + w_c * df["churn_norm"] + w_s * df["szz_norm"]
    ) / total_w

    df["is_high_risk"] = _top_percentile(df["risk_score"], percentile)
    df["is_high_risk"] = df["is_high_risk"].astype(bool)
    return df


# ---------------------------------------------------------------------------
# Severity-based baseline labeling
# ---------------------------------------------------------------------------
_SEVERITY_ORDER = {"INFO": 0, "MINOR": 1, "MAJOR": 2, "CRITICAL": 3, "BLOCKER": 4}


def compute_severity_labels(
    project_id: str,
    snapshot: pd.Timestamp,
    sonar_issues: pd.DataFrame,
    changes: pd.DataFrame,
    high_risk_levels: Iterable[str] = SEVERITY_BASELINE_LEVELS,
) -> pd.DataFrame:
    """Compute SonarQube severity-based labels for one project.

    A basename is labeled high-risk iff it has at least one SonarQube issue
    open at ``snapshot`` whose ``SEVERITY`` is in ``high_risk_levels``
    (default ``("BLOCKER", "CRITICAL")``).

    Returns a DataFrame with columns:
    ``project_id``, ``basename``, ``n_blocker``, ``n_critical``,
    ``n_major``, ``n_minor``, ``n_info``, ``max_severity_rank``,
    ``is_high_risk``.
    """
    universe = basename_universe_at_snapshot(changes, project_id, snapshot)
    if universe.empty:
        return universe.assign(is_high_risk=False)

    open_iss = open_issues_at_snapshot(sonar_issues, project_id, snapshot)
    if open_iss.empty:
        df = universe.copy()
        for col in ("n_blocker", "n_critical", "n_major", "n_minor", "n_info"):
            df[col] = 0
        df["max_severity_rank"] = 0
        df["is_high_risk"] = False
        return df

    sev_counts = (
        open_iss.pivot_table(
            index="basename", columns="SEVERITY", values="ISSUE_KEY", aggfunc="count", fill_value=0
        )
        .rename(
            columns={
                "BLOCKER": "n_blocker",
                "CRITICAL": "n_critical",
                "MAJOR": "n_major",
                "MINOR": "n_minor",
                "INFO": "n_info",
            }
        )
        .reset_index()
    )
    for col in ("n_blocker", "n_critical", "n_major", "n_minor", "n_info"):
        if col not in sev_counts.columns:
            sev_counts[col] = 0

    df = universe.merge(sev_counts, on="basename", how="left")
    for col in ("n_blocker", "n_critical", "n_major", "n_minor", "n_info"):
        df[col] = df[col].fillna(0).astype("int64")

    def _max_rank(row) -> int:
        rank = 0
        if row["n_blocker"] > 0:
            rank = 4
        elif row["n_critical"] > 0:
            rank = 3
        elif row["n_major"] > 0:
            rank = 2
        elif row["n_minor"] > 0:
            rank = 1
        elif row["n_info"] > 0:
            rank = 0
        return rank

    df["max_severity_rank"] = df.apply(_max_rank, axis=1).astype("int64")

    levels = {s.upper() for s in high_risk_levels}
    flags = (
        (("BLOCKER" in levels) & (df["n_blocker"] > 0))
        | (("CRITICAL" in levels) & (df["n_critical"] > 0))
        | (("MAJOR" in levels) & (df["n_major"] > 0))
        | (("MINOR" in levels) & (df["n_minor"] > 0))
        | (("INFO" in levels) & (df["n_info"] > 0))
    )
    df["is_high_risk"] = flags.astype(bool)
    return df


# ---------------------------------------------------------------------------
# SZZ defect-oriented baseline labeling
# ---------------------------------------------------------------------------
def compute_szz_labels(
    project_id: str,
    snapshot: pd.Timestamp,
    changes: pd.DataFrame,
    szz: pd.DataFrame,
    window_months: int = OBSERVATION_WINDOW_MONTHS,
) -> pd.DataFrame:
    """SZZ defect-oriented labels: basename is positive iff a fault-fixing
    commit within the window modified it.

    This is the classical SZZ-based defect-prediction target, adapted to
    basename granularity.
    """
    universe = basename_universe_at_snapshot(changes, project_id, snapshot)
    if universe.empty:
        return universe.assign(is_high_risk=False)

    sz = szz_events_in_window(szz, changes, project_id, snapshot, window_months)
    df = universe.merge(sz, on="basename", how="left")
    for c in ("n_szz_fixes_future", "n_szz_inducing_past"):
        if c in df.columns:
            df[c] = df[c].fillna(0).astype("int64")
        else:
            df[c] = 0
    df["is_high_risk"] = df["n_szz_fixes_future"] > 0
    return df


# ---------------------------------------------------------------------------
# Label agreement
# ---------------------------------------------------------------------------
def label_agreement(labels_by_variant: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pairwise Cohen's kappa and Jaccard between label variants.

    Each value in ``labels_by_variant`` must be a DataFrame containing
    ``project_id``, ``basename`` and ``is_high_risk`` columns.
    """
    from itertools import combinations

    # Align all variants on the union of (project_id, basename)
    keyed = {
        name: df[["project_id", "basename", "is_high_risk"]]
        .rename(columns={"is_high_risk": name})
        for name, df in labels_by_variant.items()
    }
    merged = None
    for name, df in keyed.items():
        merged = df if merged is None else merged.merge(df, on=["project_id", "basename"], how="outer")
    for name in keyed:
        merged[name] = merged[name].fillna(False).astype(bool)

    rows = []
    names = list(keyed.keys())
    for a, b in combinations(names, 2):
        ya, yb = merged[a].values, merged[b].values
        pa = ya.mean()
        pb = yb.mean()
        p_obs = (ya == yb).mean()
        p_exp = pa * pb + (1 - pa) * (1 - pb)
        kappa = (p_obs - p_exp) / (1 - p_exp) if p_exp < 1 else np.nan
        inter = (ya & yb).sum()
        union = (ya | yb).sum()
        jaccard = inter / union if union > 0 else np.nan
        rows.append(
            {
                "variant_a": a,
                "variant_b": b,
                "positives_a": int(ya.sum()),
                "positives_b": int(yb.sum()),
                "agreement_pct": round(p_obs * 100, 2),
                "cohen_kappa": round(kappa, 4) if np.isfinite(kappa) else np.nan,
                "jaccard": round(jaccard, 4) if np.isfinite(jaccard) else np.nan,
                "intersection": int(inter),
                "union": int(union),
            }
        )
    return pd.DataFrame(rows)
