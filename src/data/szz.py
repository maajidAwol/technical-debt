"""
SZZ, bug-fix and Jira resolution helpers operating on cleaned parquets.

These helpers bridge commit-level tables (GIT_COMMITS, GIT_COMMITS_CHANGES,
SZZ_FAULT_INDUCING_COMMITS, JIRA_ISSUES) and the basename-level units of
analysis used throughout the pipeline.

Key operations
--------------
- ``bugfix_touches_in_window`` - for a project, a snapshot ``t`` and a
  window length ``W``, return a DataFrame of ``(basename,
  n_bugfix_commits, churn_add, churn_removed)`` covering bug-fix commits
  whose ``AUTHOR_DATE`` falls in ``(t, t+W]``.
- ``churn_in_window`` - per-basename churn (sum of LINES_ADDED +
  LINES_REMOVED) in the post-snapshot window, irrespective of bug-fix
  flag. Used as the "future_churn" component of the consequence score.
- ``szz_events_in_window`` - per-basename count of SZZ fault-fixing
  commits in ``(t, t+W]`` that touched the basename (i.e. the fix
  modified the file). Used as the "szz_defects_future" component.
- ``jira_bug_commits_in_window`` - commits linked to Jira ``Bug`` issues
  that close inside the observation window. Used as a stricter bug-fix
  signal (sensitivity analysis; not in primary risk score to keep the
  score fully reproducible from the DB).

Everything is indexed by ``(project_id, basename)`` to match the chosen
unit of analysis (see RESEARCH_LOG.md 2026-04-23 entry).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def _window_mask(dates: pd.Series, t: pd.Timestamp, t_end: pd.Timestamp) -> pd.Series:
    """Boolean mask for ``t < dates <= t_end`` with NaT-safe handling."""
    if dates.dtype.kind != "M":
        dates = pd.to_datetime(dates, errors="coerce", utc=True)
    return (dates > t) & (dates <= t_end)


def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts


def bugfix_touches_in_window(
    commits: pd.DataFrame,
    changes: pd.DataFrame,
    project_id: str,
    t: pd.Timestamp,
    window_months: int = 6,
) -> pd.DataFrame:
    """Bug-fix commit count and churn per basename in ``(t, t+W]``.

    Parameters
    ----------
    commits :
        ``clean_git_commits`` DataFrame with an ``is_bugfix`` column.
    changes :
        ``clean_git_commits_changes`` DataFrame with ``basename``.
    project_id :
        Which project to compute for.
    t :
        Snapshot timestamp (UTC-aware).
    window_months :
        Observation window length.

    Returns
    -------
    DataFrame with columns ``basename``, ``n_bugfix_commits_future``,
    ``bugfix_churn_future`` (added + removed lines in bug-fix commits).
    """
    t = _ensure_utc(t)
    t_end = t + pd.DateOffset(months=window_months)

    c = commits[commits["PROJECT_ID"] == project_id].copy()
    c = c[_window_mask(c["AUTHOR_DATE"], t, t_end) & c["is_bugfix"].fillna(False)]
    if c.empty:
        return pd.DataFrame(columns=["basename", "n_bugfix_commits_future", "bugfix_churn_future"])

    ch = changes[changes["PROJECT_ID"] == project_id].copy()
    ch = ch[ch["COMMIT_HASH"].isin(c["COMMIT_HASH"])]

    if ch.empty:
        return pd.DataFrame(columns=["basename", "n_bugfix_commits_future", "bugfix_churn_future"])

    ch = ch.assign(_row_churn=ch["LINES_ADDED"].astype("int64") + ch["LINES_REMOVED"].astype("int64"))
    agg = (
        ch.groupby("basename")
        .agg(
            n_bugfix_commits_future=("COMMIT_HASH", "nunique"),
            bugfix_churn_future=("_row_churn", "sum"),
        )
        .reset_index()
    )
    agg["bugfix_churn_future"] = agg["bugfix_churn_future"].astype("int64")
    return agg


def churn_in_window(
    changes: pd.DataFrame,
    project_id: str,
    t: pd.Timestamp,
    window_months: int = 6,
) -> pd.DataFrame:
    """Total churn per basename in ``(t, t+W]`` (all commits, not only bug-fixes)."""
    t = _ensure_utc(t)
    t_end = t + pd.DateOffset(months=window_months)

    ch = changes[changes["PROJECT_ID"] == project_id].copy()
    ch = ch[_window_mask(ch["DATE"], t, t_end)]
    if ch.empty:
        return pd.DataFrame(
            columns=["basename", "future_churn", "future_add", "future_removed", "future_commits"]
        )

    agg = (
        ch.groupby("basename")
        .agg(
            future_add=("LINES_ADDED", "sum"),
            future_removed=("LINES_REMOVED", "sum"),
            future_commits=("COMMIT_HASH", "nunique"),
        )
        .reset_index()
    )
    agg["future_churn"] = (agg["future_add"] + agg["future_removed"]).astype("int64")
    return agg[["basename", "future_churn", "future_add", "future_removed", "future_commits"]]


def szz_events_in_window(
    szz: pd.DataFrame,
    changes: pd.DataFrame,
    project_id: str,
    t: pd.Timestamp,
    window_months: int = 6,
) -> pd.DataFrame:
    """Per-basename count of SZZ fault-fixing commits in ``(t, t+W]`` touching it.

    Uses the ``fix_date`` attached in ``clean_szz_with_dates``. The file
    resolution is done by joining SZZ's ``FAULT_FIXING_COMMIT_HASH`` to
    ``GIT_COMMITS_CHANGES.COMMIT_HASH`` to find touched basenames.
    """
    t = _ensure_utc(t)
    t_end = t + pd.DateOffset(months=window_months)

    s = szz[szz["PROJECT_ID"] == project_id].copy()
    s = s[_window_mask(s["fix_date"], t, t_end)]
    if s.empty:
        return pd.DataFrame(columns=["basename", "n_szz_fixes_future", "n_szz_inducing_past"])

    ch = changes[changes["PROJECT_ID"] == project_id]

    # (1) Basenames touched by fault-fixing commits in window
    fix_hashes = set(s["FAULT_FIXING_COMMIT_HASH"].dropna().unique())
    fix_touches = ch[ch["COMMIT_HASH"].isin(fix_hashes)][["basename", "COMMIT_HASH"]]
    fix_agg = (
        fix_touches.groupby("basename")["COMMIT_HASH"]
        .nunique()
        .rename("n_szz_fixes_future")
        .reset_index()
    )

    # (2) Basenames that were originally modified by fault-inducing commits
    # before the snapshot (these are the "ticking-time-bomb" files)
    induce = s[s["induce_date"].notna() & (s["induce_date"] <= t)]
    induce_hashes = set(induce["FAULT_INDUCING_COMMIT_HASH"].dropna().unique())
    induce_touches = ch[ch["COMMIT_HASH"].isin(induce_hashes)][["basename", "COMMIT_HASH"]]
    induce_agg = (
        induce_touches.groupby("basename")["COMMIT_HASH"]
        .nunique()
        .rename("n_szz_inducing_past")
        .reset_index()
    )

    out = pd.merge(fix_agg, induce_agg, on="basename", how="outer").fillna(0)
    for c in ("n_szz_fixes_future", "n_szz_inducing_past"):
        out[c] = out[c].astype("int64")
    return out


def jira_bug_commits_in_window(
    commits: pd.DataFrame,
    jira: pd.DataFrame,
    changes: pd.DataFrame,
    project_id: str,
    t: pd.Timestamp,
    window_months: int = 6,
) -> pd.DataFrame:
    """Per-basename count of commits that close a Jira *Bug* in ``(t, t+W]``.

    Uses the pre-populated ``JIRA_ISSUES.HASH`` column to link tickets to
    commits directly; supplements by scanning for Jira keys in commit
    messages of bug-fix commits. Returned as a separate diagnostic
    feature - not part of the primary risk score to keep the score
    computable for projects that lack rich Jira linkage.
    """
    t = _ensure_utc(t)
    t_end = t + pd.DateOffset(months=window_months)

    j = jira[jira["PROJECT_ID"] == project_id].copy()
    j = j[j["is_bug"]]
    bug_keys = set(j["KEY"].dropna().unique())
    hash_links = set(j["HASH"].dropna().unique())

    c = commits[commits["PROJECT_ID"] == project_id].copy()
    c = c[_window_mask(c["AUTHOR_DATE"], t, t_end)]
    if c.empty:
        return pd.DataFrame(columns=["basename", "n_jira_bug_commits_future"])

    # Commits that either link to a bug ticket via hash OR mention a known bug key
    c["mentions_bug_key"] = c["jira_keys"].apply(
        lambda keys: any(k in bug_keys for k in keys) if isinstance(keys, list) else False
    )
    c["hash_in_bug_links"] = c["COMMIT_HASH"].isin(hash_links)
    c = c[c["mentions_bug_key"] | c["hash_in_bug_links"]]
    if c.empty:
        return pd.DataFrame(columns=["basename", "n_jira_bug_commits_future"])

    ch = changes[changes["PROJECT_ID"] == project_id]
    ch = ch[ch["COMMIT_HASH"].isin(c["COMMIT_HASH"])]
    if ch.empty:
        return pd.DataFrame(columns=["basename", "n_jira_bug_commits_future"])
    agg = (
        ch.groupby("basename")["COMMIT_HASH"]
        .nunique()
        .rename("n_jira_bug_commits_future")
        .reset_index()
    )
    return agg


def basename_universe_at_snapshot(
    changes: pd.DataFrame,
    project_id: str,
    t: pd.Timestamp,
) -> pd.DataFrame:
    """Return the set of basenames that exist in ``project_id`` at time ``t``.

    A basename "exists at t" if any change with ``DATE <= t`` touched it.
    Output columns: ``project_id``, ``basename``.
    """
    t = _ensure_utc(t)
    ch = changes[changes["PROJECT_ID"] == project_id]
    ch = ch[ch["DATE"] <= t]
    universe = ch[["basename"]].drop_duplicates().copy()
    universe["project_id"] = project_id
    return universe[["project_id", "basename"]].reset_index(drop=True)


def open_issues_at_snapshot(
    issues: pd.DataFrame,
    project_id: str,
    t: pd.Timestamp,
) -> pd.DataFrame:
    """Filter ``clean_sonar_issues`` to issues open at time ``t``.

    An issue is "open at ``t``" iff ``CREATION_DATE <= t`` and
    (``CLOSE_DATE`` is NaT or ``CLOSE_DATE > t``).
    """
    t = _ensure_utc(t)
    df = issues[issues["PROJECT_ID"] == project_id]
    creation_ok = df["CREATION_DATE"] <= t
    close_ok = df["CLOSE_DATE"].isna() | (df["CLOSE_DATE"] > t)
    return df[creation_ok & close_ok].copy()
