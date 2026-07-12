"""
Family 5 - Prior defect history features at (project, basename) granularity.

Five features computed strictly from events with date ``<= t``:

    bugfix_commits_pre   - lifetime bug-fix commit count
    bugfix_commits_90d   - bug-fix commits in last 90 days before t
    bug_density_pre      - bugfix_commits_pre / total_commits_pre
                           (0.0 if total_commits_pre == 0)
    n_jira_bugs_pre      - JIRA Bug-type tickets linked to the file
                           before t
    jira_blocker_flag    - 1 if any linked JIRA Bug has
                           PRIORITY in {Blocker, Critical}, else 0

References: Hassan ICSE 2009, Falessi et al. ESEM 2020.
The same bug-fix regex used for the dual-signal label is reused via
the ``is_bugfix`` column produced in stage 3.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.szz import basename_universe_at_snapshot  # noqa: E402


PRIOR_DEFECT_FEATURE_COLS: tuple[str, ...] = (
    "bugfix_commits_pre",
    "bugfix_commits_90d",
    "bug_density_pre",
    "n_jira_bugs_pre",
    "jira_blocker_flag",
)


def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts


def _empty_features_for_universe(universe: pd.DataFrame) -> pd.DataFrame:
    out = universe.copy()
    out["bugfix_commits_pre"] = 0
    out["bugfix_commits_90d"] = 0
    out["bug_density_pre"] = 0.0
    out["n_jira_bugs_pre"] = 0
    out["jira_blocker_flag"] = 0
    for col in ("bugfix_commits_pre", "bugfix_commits_90d", "n_jira_bugs_pre", "jira_blocker_flag"):
        out[col] = out[col].astype("int64")
    return out


def _bugfix_aggregates(
    commits: pd.DataFrame,
    changes: pd.DataFrame,
    project_id: str,
    t: pd.Timestamp,
) -> pd.DataFrame:
    """Per-basename bug-fix counts + total commits for the bug_density_pre denominator."""
    c_all = commits[commits["PROJECT_ID"] == project_id]
    c_all = c_all[c_all["AUTHOR_DATE"] <= t][["COMMIT_HASH", "AUTHOR_DATE", "is_bugfix"]]

    ch = changes[changes["PROJECT_ID"] == project_id]
    ch = ch[ch["DATE"] <= t][["COMMIT_HASH", "basename"]]
    if ch.empty or c_all.empty:
        return pd.DataFrame(
            columns=["basename", "bugfix_commits_pre", "bugfix_commits_90d", "total_commits_pre"]
        )

    touches = ch.merge(c_all, on="COMMIT_HASH", how="inner")
    touches = touches.drop_duplicates(["basename", "COMMIT_HASH"])
    t_90 = t - pd.Timedelta(days=90)
    touches["is_bugfix"] = touches["is_bugfix"].fillna(False).astype(bool)
    touches["in_90d"] = touches["AUTHOR_DATE"] > t_90

    agg = touches.groupby("basename").agg(
        total_commits_pre=("COMMIT_HASH", "size"),
        bugfix_commits_pre=("is_bugfix", "sum"),
        bugfix_commits_90d=("is_bugfix", lambda s: int((s & touches.loc[s.index, "in_90d"]).sum())),
    )
    for col in ("total_commits_pre", "bugfix_commits_pre", "bugfix_commits_90d"):
        agg[col] = agg[col].fillna(0).astype("int64")
    return agg.reset_index()


def _jira_bug_aggregates(
    commits: pd.DataFrame,
    jira: pd.DataFrame,
    changes: pd.DataFrame,
    project_id: str,
    t: pd.Timestamp,
) -> pd.DataFrame:
    """Per-basename JIRA Bug counts and Blocker/Critical priority flag, pre-t."""
    j = jira[jira["PROJECT_ID"] == project_id].copy()
    j = j[j["is_bug"] & j["HASH"].notna()]
    if j.empty:
        return pd.DataFrame(columns=["basename", "n_jira_bugs_pre", "jira_blocker_flag"])

    # Resolved before t (or commit-date before t) -- use the earlier of the two if available.
    j["link_date"] = j[["RESOLUTION_DATE", "COMMIT_DATE", "UPDATE_DATE"]].min(axis=1)
    j = j[j["link_date"].isna() | (j["link_date"] <= t)]
    if j.empty:
        return pd.DataFrame(columns=["basename", "n_jira_bugs_pre", "jira_blocker_flag"])

    j["priority_high"] = (
        j["PRIORITY"].fillna("").astype(str).str.upper().isin({"BLOCKER", "CRITICAL"})
    )

    bug_hashes = j[["HASH", "priority_high"]].drop_duplicates(subset=["HASH"])

    c = commits[commits["PROJECT_ID"] == project_id]
    c = c[(c["AUTHOR_DATE"] <= t) & c["COMMIT_HASH"].isin(bug_hashes["HASH"])][
        ["COMMIT_HASH"]
    ]
    if c.empty:
        return pd.DataFrame(columns=["basename", "n_jira_bugs_pre", "jira_blocker_flag"])

    ch = changes[changes["PROJECT_ID"] == project_id]
    ch = ch[(ch["DATE"] <= t) & ch["COMMIT_HASH"].isin(c["COMMIT_HASH"])][
        ["COMMIT_HASH", "basename"]
    ]
    if ch.empty:
        return pd.DataFrame(columns=["basename", "n_jira_bugs_pre", "jira_blocker_flag"])

    ch = ch.merge(bug_hashes, left_on="COMMIT_HASH", right_on="HASH", how="left")

    agg = ch.groupby("basename").agg(
        n_jira_bugs_pre=("HASH", "nunique"),
        jira_blocker_flag=("priority_high", "any"),
    )
    agg["n_jira_bugs_pre"] = agg["n_jira_bugs_pre"].fillna(0).astype("int64")
    agg["jira_blocker_flag"] = agg["jira_blocker_flag"].fillna(False).astype("int64")
    return agg.reset_index()


def build_priordefect_features_for_project(
    project_id: str,
    snapshot: pd.Timestamp,
    commits: pd.DataFrame,
    changes: pd.DataFrame,
    szz: pd.DataFrame,  # kept in signature for stage-5 caller compatibility; not used
    jira: pd.DataFrame,
) -> pd.DataFrame:
    t = _ensure_utc(snapshot)
    universe = basename_universe_at_snapshot(changes, project_id, t)
    if universe.empty:
        return _empty_features_for_universe(universe)

    bf = _bugfix_aggregates(commits, changes, project_id, t)
    jb = _jira_bug_aggregates(commits, jira, changes, project_id, t)

    out = universe.merge(bf, on="basename", how="left")
    out = out.merge(jb, on="basename", how="left")

    for col in ("bugfix_commits_pre", "bugfix_commits_90d", "total_commits_pre",
                "n_jira_bugs_pre", "jira_blocker_flag"):
        if col not in out.columns:
            out[col] = 0
        out[col] = out[col].fillna(0).astype("int64")

    out["bug_density_pre"] = np.where(
        out["total_commits_pre"] > 0,
        out["bugfix_commits_pre"] / out["total_commits_pre"].replace(0, np.nan),
        0.0,
    ).astype(float)
    out["bug_density_pre"] = out["bug_density_pre"].fillna(0.0)

    keep = [
        "project_id",
        "basename",
        "bugfix_commits_pre",
        "bugfix_commits_90d",
        "bug_density_pre",
        "n_jira_bugs_pre",
        "jira_blocker_flag",
    ]
    return out[keep]
