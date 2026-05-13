"""
Dual-signal combined-weight binary label for technical debt.

Binary output: ``is_high_risk`` in {0, 1}. Built from six binary signals
spanning two independent sources:

  Static (SonarQube, at snapshot t):
    S1_severity     - file has >= 1 OPEN BLOCKER/CRITICAL issue
    S2_debt         - total SonarQube DEBT minutes > project median
    S3_smells       - count of CODE_SMELL issues > project median

  History (Git, AUTHOR_DATE <= t):
    S4_bugfix       - bug-fix commit count > project median
    S5_churn        - lifetime churn > project 75th percentile
    S6_contributors - distinct authors > project median

A weighted sum ``risk_score = sum(w_i * S_i)`` in [0, 1] is thresholded
at 0.50 (relaxed to 0.45 / 0.40 per project if positives < 5).

Weights are derived empirically via point-biserial correlation between
each signal and a 6-month post-snapshot bug-fix surrogate (Kamei TSE
2013 weighting approach). If the empirical weights are within 0.05 of
the theoretically motivated baseline, the theoretical weights are used
and the empirical run is logged as confirmation. The surrogate is used
only for weight derivation and is never persisted alongside features.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

from config import (
    BUGFIX_REGEX,
    LABEL_MIN_POSITIVES_PER_PROJECT,
    LABEL_RISK_THRESHOLD,
    LABEL_SURROGATE_WINDOW_MONTHS,
    LABEL_THRESHOLD_FALLBACKS,
    SEVERITY_LABEL_LEVELS,
    THEORETICAL_WEIGHTS,
)


_BUGFIX_RE = re.compile(BUGFIX_REGEX, re.IGNORECASE)

SIGNAL_COLUMNS: Tuple[str, ...] = (
    "S1_severity",
    "S2_debt",
    "S3_smells",
    "S4_bugfix",
    "S5_churn",
    "S6_contributors",
)

# All six signals use p75 as the within-project
# threshold. This ensures only genuinely elevated
# files are flagged (top 25% on each dimension).
# Using the median would flag ~50% of files per
# signal by construction, which is not a useful
# high-risk indicator. p75 alignment follows the
# design rationale of S5 and is consistent with
# percentile-based prioritization in Kamei 2013.
# (S1 is a boolean presence flag for BLOCKER/CRITICAL
# issues - not percentile-based by construction.)


# ---------------------------------------------------------------------------
# Six per-file binary signals at snapshot t
# ---------------------------------------------------------------------------
def _project_basenames(
    sonar_issues_p: pd.DataFrame,
    changes_p: pd.DataFrame,
) -> pd.Index:
    """Union of basenames seen in either Sonar issues or Git changes for one project."""
    sonar_b = sonar_issues_p["basename"].dropna().unique() if "basename" in sonar_issues_p else []
    git_b = changes_p["basename"].dropna().unique() if "basename" in changes_p else []
    return pd.Index(sorted(set(sonar_b) | set(git_b)), name="basename")


def _static_signals(
    sonar_issues_p: pd.DataFrame,
    t: pd.Timestamp,
    basenames: pd.Index,
) -> pd.DataFrame:
    """Compute S1, S2, S3 over basenames using SONAR_ISSUES rows with CREATION_DATE <= t."""
    df = sonar_issues_p.copy()
    if "CREATION_DATE" in df.columns:
        df = df[df["CREATION_DATE"] <= t]
    sev = df["SEVERITY"].astype(str).str.upper()
    status = df["STATUS"].astype(str).str.upper()
    issue_type = df["TYPE"].astype(str).str.upper()

    # S1 - open BLOCKER/CRITICAL count
    sev_mask = sev.isin(SEVERITY_LABEL_LEVELS) & (status == "OPEN")
    n_sev = df.loc[sev_mask].groupby("basename").size()

    # S2 - total debt
    debt = df.groupby("basename")["DEBT"].sum(min_count=1).fillna(0.0)

    # S3 - smell count
    smell_mask = issue_type == "CODE_SMELL"
    n_smells = df.loc[smell_mask].groupby("basename").size()

    out = pd.DataFrame(index=basenames)
    out["n_severity"] = n_sev.reindex(basenames).fillna(0).astype("int64")
    out["total_debt"] = debt.reindex(basenames).fillna(0.0).astype("float64")
    out["n_smells"] = n_smells.reindex(basenames).fillna(0).astype("int64")

    out["S1_severity"] = (out["n_severity"] >= 1).astype("int64")

    debt_p75 = out["total_debt"].quantile(0.75)
    smell_p75 = out["n_smells"].quantile(0.75)
    out["S2_debt"] = (out["total_debt"] > debt_p75).astype("int64")
    out["S3_smells"] = (out["n_smells"] > smell_p75).astype("int64")
    return out


def _history_signals(
    commits_p: pd.DataFrame,
    changes_p: pd.DataFrame,
    t: pd.Timestamp,
    basenames: pd.Index,
) -> pd.DataFrame:
    """Compute S4, S5, S6 over basenames using commits/changes with AUTHOR_DATE/DATE <= t."""
    if "AUTHOR_DATE" in commits_p.columns:
        commits_pre = commits_p[commits_p["AUTHOR_DATE"] <= t]
    else:
        commits_pre = commits_p.iloc[0:0]

    if "DATE" in changes_p.columns:
        changes_pre = changes_p[changes_p["DATE"] <= t]
    else:
        changes_pre = changes_p.iloc[0:0]

    commit_meta = commits_pre[["COMMIT_HASH", "AUTHOR"]].copy()
    if "is_bugfix" in commits_pre.columns:
        commit_meta["is_bugfix"] = commits_pre["is_bugfix"].astype(bool).values
    else:
        msgs = commits_pre.get("COMMIT_MESSAGE", pd.Series([""] * len(commits_pre)))
        commit_meta["is_bugfix"] = msgs.fillna("").astype(str).str.contains(_BUGFIX_RE, regex=True)

    file_commit_pairs = changes_pre[
        ["COMMIT_HASH", "basename", "LINES_ADDED", "LINES_REMOVED"]
    ].merge(commit_meta, on="COMMIT_HASH", how="left")
    file_commit_pairs["is_bugfix"] = file_commit_pairs["is_bugfix"].fillna(False).astype(bool)

    # S4 - bug-fix commit count per basename
    bf = (
        file_commit_pairs[file_commit_pairs["is_bugfix"]]
        .drop_duplicates(["basename", "COMMIT_HASH"])
        .groupby("basename")
        .size()
    )

    # S5 - total churn (added + removed) per basename
    file_commit_pairs["churn"] = (
        file_commit_pairs["LINES_ADDED"].fillna(0).astype("int64")
        + file_commit_pairs["LINES_REMOVED"].fillna(0).astype("int64")
    )
    churn = file_commit_pairs.groupby("basename")["churn"].sum()

    # S6 - distinct authors per basename
    authors = (
        file_commit_pairs.dropna(subset=["AUTHOR"])
        .groupby("basename")["AUTHOR"]
        .nunique()
    )

    out = pd.DataFrame(index=basenames)
    out["n_bugfix"] = bf.reindex(basenames).fillna(0).astype("int64")
    out["total_churn"] = churn.reindex(basenames).fillna(0).astype("int64")
    out["n_authors"] = authors.reindex(basenames).fillna(0).astype("int64")

    bf_p75 = out["n_bugfix"].quantile(0.75)
    churn_p75 = out["total_churn"].quantile(0.75)
    auth_p75 = out["n_authors"].quantile(0.75)

    out["S4_bugfix"] = (out["n_bugfix"] > bf_p75).astype("int64")
    out["S5_churn"] = (out["total_churn"] > churn_p75).astype("int64")
    out["S6_contributors"] = (out["n_authors"] > auth_p75).astype("int64")
    return out


def _future_bugfix_count(
    commits_p: pd.DataFrame,
    changes_p: pd.DataFrame,
    t: pd.Timestamp,
    window_months: int,
    basenames: pd.Index,
) -> pd.Series:
    """Surrogate: bug-fix commits per basename in (t, t + window]. Weight-derivation only."""
    if "AUTHOR_DATE" not in commits_p.columns or "DATE" not in changes_p.columns:
        return pd.Series(0, index=basenames, dtype="int64", name="future_bugfix_count")

    t_end = t + pd.DateOffset(months=window_months)
    post_commits = commits_p[(commits_p["AUTHOR_DATE"] > t) & (commits_p["AUTHOR_DATE"] <= t_end)]
    if "is_bugfix" in post_commits.columns:
        bf_hashes = post_commits.loc[post_commits["is_bugfix"].astype(bool), "COMMIT_HASH"]
    else:
        msgs = post_commits.get("COMMIT_MESSAGE", pd.Series([""] * len(post_commits)))
        is_bf = msgs.fillna("").astype(str).str.contains(_BUGFIX_RE, regex=True)
        bf_hashes = post_commits.loc[is_bf, "COMMIT_HASH"]

    if len(bf_hashes) == 0:
        return pd.Series(0, index=basenames, dtype="int64", name="future_bugfix_count")

    post_changes = changes_p[
        (changes_p["DATE"] > t)
        & (changes_p["DATE"] <= t_end)
        & changes_p["COMMIT_HASH"].isin(bf_hashes)
    ]
    counts = (
        post_changes.drop_duplicates(["basename", "COMMIT_HASH"])
        .groupby("basename")
        .size()
    )
    return counts.reindex(basenames).fillna(0).astype("int64").rename("future_bugfix_count")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_dual_signal_signals(
    project_id: str,
    t: pd.Timestamp,
    commits: pd.DataFrame,
    changes: pd.DataFrame,
    sonar_issues: pd.DataFrame,
    surrogate_window_months: int = LABEL_SURROGATE_WINDOW_MONTHS,
) -> pd.DataFrame:
    """Return per-basename signals S1..S6 plus the weight-derivation surrogate.

    The surrogate column ``future_bugfix_count`` is returned alongside S1..S6
    so the caller can stack frames across projects to fit the empirical
    weights. It must be dropped before merging with the feature matrix
    (see assertions in scripts/06_build_dataset.py).
    """
    cp = commits[commits["PROJECT_ID"] == project_id] if "PROJECT_ID" in commits.columns else commits
    ch = changes[changes["PROJECT_ID"] == project_id] if "PROJECT_ID" in changes.columns else changes
    si = (
        sonar_issues[sonar_issues["PROJECT_ID"] == project_id]
        if "PROJECT_ID" in sonar_issues.columns
        else sonar_issues
    )

    basenames = _project_basenames(si, ch)
    if len(basenames) == 0:
        cols = ["project_id", "basename", *SIGNAL_COLUMNS, "future_bugfix_count"]
        return pd.DataFrame({c: pd.Series(dtype="int64") for c in cols})

    static_df = _static_signals(si, t, basenames)
    history_df = _history_signals(cp, ch, t, basenames)
    surrogate = _future_bugfix_count(cp, ch, t, surrogate_window_months, basenames)

    out = pd.DataFrame(index=basenames)
    for col in SIGNAL_COLUMNS:
        out[col] = static_df[col] if col in static_df.columns else history_df[col]
    out["future_bugfix_count"] = surrogate
    out.insert(0, "project_id", project_id)
    out.index.name = "basename"
    return out.reset_index()


def derive_weights_empirically(
    df_with_signals: pd.DataFrame,
    df_with_surrogate: pd.DataFrame,
) -> Dict[str, float]:
    """Empirical weights via |point-biserial correlation| with future bug-fix count.

    Both inputs must align row-by-row on ``(project_id, basename)``. Each
    signal's |r| is floored at 0.01, normalised to sum to 1.0, rounded to
    two decimals; rounding drift is absorbed by the largest-weight signal.
    """
    if len(df_with_signals) != len(df_with_surrogate):
        raise ValueError("signals and surrogate frames must have the same length")
    y = df_with_surrogate["future_bugfix_count"].astype(float).values
    corrs: Dict[str, float] = {}
    for s in SIGNAL_COLUMNS:
        x = df_with_signals[s].astype(float).values
        if np.var(x) == 0 or np.var(y) == 0:
            r = 0.0
        else:
            r, _ = pointbiserialr(x, y)
        corrs[s] = max(abs(float(r)), 0.01)

    total = sum(corrs.values())
    weights = {k: round(v / total, 2) for k, v in corrs.items()}
    drift = 1.0 - sum(weights.values())
    top = max(weights, key=weights.get)
    weights[top] = round(weights[top] + drift, 2)
    return weights


def choose_weights(
    derived: Dict[str, float],
    theoretical: Dict[str, float] = THEORETICAL_WEIGHTS,
    tolerance: float = 0.05,
) -> Tuple[Dict[str, float], str]:
    """Pick theoretical weights if every signal is within ``tolerance``, else derived."""
    max_diff = max(abs(derived[s] - theoretical[s]) for s in SIGNAL_COLUMNS)
    if max_diff <= tolerance:
        return dict(theoretical), "theoretical"
    return dict(derived), "empirical"


def apply_label_thresholding(
    risk_scores: pd.Series,
    min_positives: int = LABEL_MIN_POSITIVES_PER_PROJECT,
    primary_threshold: float = LABEL_RISK_THRESHOLD,
    fallback_thresholds: Iterable[float] = LABEL_THRESHOLD_FALLBACKS,
) -> Tuple[pd.Series, float, bool]:
    """Return (is_high_risk, threshold_used, satisfied_min_positives)."""
    thresholds = [primary_threshold, *fallback_thresholds]
    for thr in thresholds:
        labels = (risk_scores >= thr).astype("int64")
        if int(labels.sum()) >= min_positives:
            return labels, thr, True
    labels = (risk_scores >= thresholds[-1]).astype("int64")
    return labels, thresholds[-1], False


def compute_dual_signal_labels(
    signals_df: pd.DataFrame,
    weights: Dict[str, float],
    group_by: Tuple[str, ...] = ("project_id",),
) -> pd.DataFrame:
    """Score and threshold per group. Returns the labels.parquet row set.

    ``signals_df`` must contain ``group_by`` columns plus ``basename`` and
    ``S1_severity..S6_contributors``. The output adds ``risk_score``,
    ``is_high_risk``, ``threshold_used``, ``min_positives_satisfied``.

    Use ``group_by=("project_id",)`` for single-snapshot pipelines and
    ``group_by=("project_id", "snapshot_id")`` for multi-snapshot mode so
    each (project, snapshot) gets its own min-positives fallback.
    """
    score = np.zeros(len(signals_df), dtype="float64")
    for s, w in weights.items():
        score += float(w) * signals_df[s].astype("float64").values

    keep_cols = list(group_by) + ["basename", *SIGNAL_COLUMNS]
    out = signals_df[keep_cols].copy()
    out["risk_score"] = score

    is_high = np.zeros(len(out), dtype="int64")
    threshold_used = np.zeros(len(out), dtype="float64")
    min_positives_ok = np.ones(len(out), dtype="int64")
    out = out.reset_index(drop=True)
    group_keys = list(group_by) if len(group_by) > 1 else group_by[0]
    for _, idx in out.groupby(group_keys).groups.items():
        positions = np.asarray(idx)
        rs = out.loc[positions, "risk_score"]
        labels, thr, ok = apply_label_thresholding(rs)
        is_high[positions] = labels.values
        threshold_used[positions] = thr
        if not ok:
            min_positives_ok[positions] = 0
    out["is_high_risk"] = is_high
    out["threshold_used"] = threshold_used
    out["min_positives_satisfied"] = min_positives_ok
    return out
