"""
Snapshot-aware historical (process) feature extraction at (project,
basename) granularity.

Consumes the cleaned ``GIT_COMMITS`` and ``GIT_COMMITS_CHANGES`` parquets
produced in Stage 3 and restricts every statistic to events with
``AUTHOR_DATE <= snapshot``. This is the temporal honesty guarantee:
**no feature value depends on a commit that happened after the snapshot
used to derive its labels**.

Features produced (``_pre`` suffix denotes pre-snapshot window):

- ``total_commits_pre`` - distinct commits touching the basename.
- ``total_contributors_pre`` - distinct authors.
- ``code_added_pre``, ``code_removed_pre`` - sum of lines.
- ``code_churn_pre`` - added + removed.
- ``recent_churn_30d_pre`` / ``recent_churn_90d_pre`` - churn in the
  last 30 / 90 days before ``t``.
- ``recent_commits_30d_pre`` / ``recent_commits_90d_pre`` - commit
  counts in the last 30 / 90 days before ``t``.
- ``file_age_days_at_snapshot`` - days between first commit and ``t``.
- ``days_since_last_change_at_snapshot`` - days between last pre-``t``
  commit and ``t``.
- ``ownership_ratio_pre`` - max author's commits / total commits.
- ``avg_change_size_pre``, ``max_single_commit_churn_pre``,
  ``std_change_size_pre`` - distribution of per-commit churn.

No feature uses ``CHANGE_TYPE`` (not available in TD Dataset v2.0);
change-type counts are replaced by distributional churn statistics
(max, std) which capture volatility without needing those annotations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.szz import basename_universe_at_snapshot  # noqa: E402


def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts


def build_historical_features_for_project(
    project_id: str,
    snapshot: pd.Timestamp,
    commits: pd.DataFrame,
    changes: pd.DataFrame,
) -> pd.DataFrame:
    """Per-basename historical features for one project at snapshot ``t``.

    Parameters
    ----------
    project_id :
        Project key to compute for.
    snapshot :
        Snapshot date ``t``; only commits with ``AUTHOR_DATE <= t`` are used.
    commits :
        ``clean_git_commits`` parquet DataFrame.
    changes :
        ``clean_git_commits_changes`` parquet DataFrame, one row per
        (commit, file) with ``basename``, ``LINES_ADDED``, ``LINES_REMOVED``.
    """
    t = _ensure_utc(snapshot)
    universe = basename_universe_at_snapshot(changes, project_id, t)
    if universe.empty:
        return universe

    # Restrict commits and changes to this project, pre-snapshot only
    c = commits[commits["PROJECT_ID"] == project_id]
    c = c[c["AUTHOR_DATE"] <= t][["COMMIT_HASH", "AUTHOR_DATE", "AUTHOR"]]

    ch = changes[changes["PROJECT_ID"] == project_id]
    ch = ch[ch["DATE"] <= t].copy()

    # Attach author info onto each change row so we can compute author-level aggregates
    ch = ch.merge(c[["COMMIT_HASH", "AUTHOR_DATE", "AUTHOR"]], on="COMMIT_HASH", how="left")
    ch["row_churn"] = ch["LINES_ADDED"].astype("int64") + ch["LINES_REMOVED"].astype("int64")

    t_30 = t - pd.Timedelta(days=30)
    t_90 = t - pd.Timedelta(days=90)
    ch["in_30d"] = (ch["AUTHOR_DATE"] > t_30) & (ch["AUTHOR_DATE"] <= t)
    ch["in_90d"] = (ch["AUTHOR_DATE"] > t_90) & (ch["AUTHOR_DATE"] <= t)

    # ----- Basename-level aggregates over all pre-snapshot commits -----
    base = ch.groupby("basename").agg(
        total_commits_pre=("COMMIT_HASH", "nunique"),
        total_contributors_pre=("AUTHOR", "nunique"),
        code_added_pre=("LINES_ADDED", "sum"),
        code_removed_pre=("LINES_REMOVED", "sum"),
        first_commit_date=("AUTHOR_DATE", "min"),
        last_commit_date=("AUTHOR_DATE", "max"),
        avg_change_size_pre=("row_churn", "mean"),
        max_single_commit_churn_pre=("row_churn", "max"),
        std_change_size_pre=("row_churn", "std"),
    )
    base["code_churn_pre"] = (base["code_added_pre"] + base["code_removed_pre"]).astype("int64")

    # ----- Recency windows -----
    def _window_agg(col_mask: str, out_churn: str, out_commits: str) -> pd.DataFrame:
        sub = ch[ch[col_mask]]
        if sub.empty:
            return pd.DataFrame(columns=["basename", out_churn, out_commits]).set_index("basename")
        return (
            sub.groupby("basename")
            .agg(**{out_churn: ("row_churn", "sum"), out_commits: ("COMMIT_HASH", "nunique")})
        )

    w30 = _window_agg("in_30d", "recent_churn_30d_pre", "recent_commits_30d_pre")
    w90 = _window_agg("in_90d", "recent_churn_90d_pre", "recent_commits_90d_pre")

    # ----- Ownership: max author commits / total commits -----
    author_commits = (
        ch.groupby(["basename", "AUTHOR"])["COMMIT_HASH"]
        .nunique()
        .reset_index(name="author_commits")
    )
    top_author = author_commits.groupby("basename")["author_commits"].max().rename("max_commits_by_author")

    # ----- Join everything -----
    df = universe.merge(base.reset_index(), on="basename", how="left")
    df = df.merge(w30.reset_index(), on="basename", how="left")
    df = df.merge(w90.reset_index(), on="basename", how="left")
    df = df.merge(top_author.reset_index(), on="basename", how="left")

    # ----- Derived fields -----
    df["file_age_days_at_snapshot"] = (t - df["first_commit_date"]).dt.days
    df["days_since_last_change_at_snapshot"] = (t - df["last_commit_date"]).dt.days
    df["ownership_ratio_pre"] = np.where(
        df["total_commits_pre"] > 0,
        df["max_commits_by_author"] / df["total_commits_pre"],
        np.nan,
    )

    # Fill NaNs: numeric aggregates default to 0, std defaults to 0, dates stay NaT
    zero_fill_cols = [
        "total_commits_pre",
        "total_contributors_pre",
        "code_added_pre",
        "code_removed_pre",
        "code_churn_pre",
        "recent_churn_30d_pre",
        "recent_commits_30d_pre",
        "recent_churn_90d_pre",
        "recent_commits_90d_pre",
        "max_single_commit_churn_pre",
    ]
    for c_ in zero_fill_cols:
        if c_ in df.columns:
            df[c_] = df[c_].fillna(0).astype("int64")

    for c_ in ("avg_change_size_pre", "std_change_size_pre", "ownership_ratio_pre"):
        if c_ in df.columns:
            df[c_] = df[c_].fillna(0.0).astype(float)

    for c_ in ("file_age_days_at_snapshot", "days_since_last_change_at_snapshot"):
        if c_ in df.columns:
            df[c_] = df[c_].fillna(0).astype("int64")

    df["snapshot_date"] = t
    return df.drop(columns=["first_commit_date", "last_commit_date", "max_commits_by_author"])
