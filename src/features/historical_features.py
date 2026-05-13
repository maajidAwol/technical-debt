"""
Family 3 - Historical change metrics at (project, basename) granularity.

8 features derived from ``GIT_COMMITS`` and ``GIT_COMMITS_CHANGES``
restricted to ``AUTHOR_DATE <= snapshot``:

    total_commits_pre       - distinct commits touching the file
    code_churn_pre          - lifetime added + removed lines
    recent_churn_90d        - churn in last 90 days before t
    commit_frequency_30d    - commit count in last 30 days before t
    file_age_days           - days between first commit and t
    days_since_last_change  - days between most recent commit and t
    contributor_count       - distinct authors
    ownership_ratio         - max single author commits / total
                              (1.0 if total_commits_pre == 0)

References: Kamei TSE 2013, Hassan ICSE 2009.
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
    t = _ensure_utc(snapshot)
    universe = basename_universe_at_snapshot(changes, project_id, t)
    if universe.empty:
        return universe

    c = commits[commits["PROJECT_ID"] == project_id]
    c = c[c["AUTHOR_DATE"] <= t][["COMMIT_HASH", "AUTHOR_DATE", "AUTHOR"]]

    ch = changes[changes["PROJECT_ID"] == project_id]
    ch = ch[ch["DATE"] <= t].copy()
    ch = ch.merge(c, on="COMMIT_HASH", how="left")
    ch["row_churn"] = ch["LINES_ADDED"].fillna(0).astype("int64") + ch["LINES_REMOVED"].fillna(0).astype("int64")

    t_30 = t - pd.Timedelta(days=30)
    t_90 = t - pd.Timedelta(days=90)
    ch["in_30d"] = (ch["AUTHOR_DATE"] > t_30) & (ch["AUTHOR_DATE"] <= t)
    ch["in_90d"] = (ch["AUTHOR_DATE"] > t_90) & (ch["AUTHOR_DATE"] <= t)

    base = ch.groupby("basename").agg(
        total_commits_pre=("COMMIT_HASH", "nunique"),
        contributor_count=("AUTHOR", "nunique"),
        code_added_pre=("LINES_ADDED", "sum"),
        code_removed_pre=("LINES_REMOVED", "sum"),
        first_commit_date=("AUTHOR_DATE", "min"),
        last_commit_date=("AUTHOR_DATE", "max"),
    )
    base["code_churn_pre"] = (
        base["code_added_pre"].fillna(0).astype("int64")
        + base["code_removed_pre"].fillna(0).astype("int64")
    )

    w90 = (
        ch[ch["in_90d"]]
        .groupby("basename")
        .agg(recent_churn_90d=("row_churn", "sum"))
    )
    w30 = (
        ch[ch["in_30d"]]
        .groupby("basename")
        .agg(commit_frequency_30d=("COMMIT_HASH", "nunique"))
    )

    author_commits = (
        ch.dropna(subset=["AUTHOR"])
        .groupby(["basename", "AUTHOR"])["COMMIT_HASH"]
        .nunique()
        .reset_index(name="author_commits")
    )
    top_author = (
        author_commits.groupby("basename")["author_commits"].max().rename("max_commits_by_author")
    )

    df = universe.merge(base.reset_index(), on="basename", how="left")
    df = df.merge(w90.reset_index(), on="basename", how="left")
    df = df.merge(w30.reset_index(), on="basename", how="left")
    df = df.merge(top_author.reset_index(), on="basename", how="left")

    df["file_age_days"] = (t - df["first_commit_date"]).dt.days
    df["days_since_last_change"] = (t - df["last_commit_date"]).dt.days
    df["ownership_ratio"] = np.where(
        df["total_commits_pre"].fillna(0) > 0,
        df["max_commits_by_author"] / df["total_commits_pre"].replace(0, np.nan),
        1.0,
    )

    int_cols = [
        "total_commits_pre",
        "contributor_count",
        "code_churn_pre",
        "recent_churn_90d",
        "commit_frequency_30d",
        "file_age_days",
        "days_since_last_change",
    ]
    for col in int_cols:
        df[col] = df[col].fillna(0).astype("int64")
    df["ownership_ratio"] = df["ownership_ratio"].fillna(1.0).astype(float)

    keep = [
        "project_id",
        "basename",
        "total_commits_pre",
        "code_churn_pre",
        "recent_churn_90d",
        "commit_frequency_30d",
        "file_age_days",
        "days_since_last_change",
        "contributor_count",
        "ownership_ratio",
    ]
    return df[keep]
