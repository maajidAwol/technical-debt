"""
Snapshot date selection and temporal-split utilities.

Implements the temporal-split protocol described in the approved proposal
Section 3.4: for each project, a snapshot time ``t`` is selected such that
features are computed from commits with ``AUTHOR_DATE <= t`` and labels are
derived from commits in the observation window ``(t, t + W]`` where ``W``
is the observation window (6 months primary).

The default strategy is the **median master-branch commit date** per project,
which guarantees every project has both a meaningful pre-snapshot history
(for features) and a meaningful post-snapshot window (for labels).

References
----------
- Zimmermann, T., Nagappan, N. (2008). Predicting defects using network analysis
  on dependency graphs. ICSE 2008 - basis for 6-month observation window.
- Jiang, Z., Chen, T., Zhou, Y. (2024). Improving technical debt prediction with
  graph-based and social-network metrics. Empir. Softw. Eng. 29 - uses median
  snapshot strategy.
"""
from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    MIN_POST_SNAPSHOT_COMMITS,
    MIN_PRE_SNAPSHOT_COMMITS,
    OBSERVATION_WINDOW_MONTHS,
    SNAPSHOT_STRATEGY,
)


@dataclass
class ProjectSnapshot:
    """Per-project snapshot metadata."""

    project_id: str
    snapshot_date: pd.Timestamp
    window_months: int
    first_commit: pd.Timestamp
    last_commit: pd.Timestamp
    total_commits: int
    pre_snapshot_commits: int
    post_snapshot_commits: int
    distinct_files_pre: int
    distinct_authors_pre: int
    eligible: bool
    exclusion_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "snapshot_date": self.snapshot_date,
            "window_months": self.window_months,
            "window_end": self.window_end,
            "first_commit": self.first_commit,
            "last_commit": self.last_commit,
            "total_commits": self.total_commits,
            "pre_snapshot_commits": self.pre_snapshot_commits,
            "post_snapshot_commits": self.post_snapshot_commits,
            "distinct_files_pre": self.distinct_files_pre,
            "distinct_authors_pre": self.distinct_authors_pre,
            "eligible": self.eligible,
            "exclusion_reason": self.exclusion_reason,
        }

    @property
    def window_end(self) -> pd.Timestamp:
        return self.snapshot_date + pd.DateOffset(months=self.window_months)


def _add_months(t: pd.Timestamp, months: int) -> pd.Timestamp:
    return t + pd.DateOffset(months=months)


def compute_project_snapshot(
    conn: sqlite3.Connection,
    project_id: str,
    strategy: str = SNAPSHOT_STRATEGY,
    window_months: int = OBSERVATION_WINDOW_MONTHS,
    min_pre: int = MIN_PRE_SNAPSHOT_COMMITS,
    min_post: int = MIN_POST_SNAPSHOT_COMMITS,
) -> ProjectSnapshot:
    """Compute the snapshot date for a single project.

    Parameters
    ----------
    conn :
        SQLite connection to the Technical Debt Dataset.
    project_id :
        PROJECT_ID value (e.g. ``'org.apache:batik'``).
    strategy :
        One of ``'median'`` (default), ``'fixed'`` (uses ``window_months`` back
        from the last commit), or ``'release'`` (not implemented yet).
    window_months :
        Length of the post-snapshot observation window.
    min_pre, min_post :
        Minimum number of pre- and post-snapshot commits required for a
        project to be considered eligible.
    """
    query = (
        "SELECT AUTHOR_DATE, COMMIT_HASH, AUTHOR "
        "FROM GIT_COMMITS "
        "WHERE PROJECT_ID = ? AND IN_MAIN_BRANCH = 'True' "
        "ORDER BY AUTHOR_DATE ASC"
    )
    commits = pd.read_sql_query(query, conn, params=[project_id])
    commits["AUTHOR_DATE"] = pd.to_datetime(commits["AUTHOR_DATE"], errors="coerce", utc=True)
    commits = commits.dropna(subset=["AUTHOR_DATE"])

    if len(commits) == 0:
        return ProjectSnapshot(
            project_id=project_id,
            snapshot_date=pd.NaT,
            window_months=window_months,
            first_commit=pd.NaT,
            last_commit=pd.NaT,
            total_commits=0,
            pre_snapshot_commits=0,
            post_snapshot_commits=0,
            distinct_files_pre=0,
            distinct_authors_pre=0,
            eligible=False,
            exclusion_reason="no_commits",
        )

    first = commits["AUTHOR_DATE"].min()
    last = commits["AUTHOR_DATE"].max()

    if strategy == "median":
        snapshot = commits["AUTHOR_DATE"].quantile(0.5, interpolation="nearest")
    elif strategy == "fixed":
        snapshot = _add_months(last, -window_months)
    elif strategy == "release":
        raise NotImplementedError("release-tag strategy not implemented yet")
    else:
        raise ValueError(f"Unknown snapshot strategy: {strategy}")

    snapshot = pd.Timestamp(snapshot)
    window_end = _add_months(snapshot, window_months)

    pre_mask = commits["AUTHOR_DATE"] <= snapshot
    post_mask = (commits["AUTHOR_DATE"] > snapshot) & (commits["AUTHOR_DATE"] <= window_end)
    pre_count = int(pre_mask.sum())
    post_count = int(post_mask.sum())

    # Distinct pre-snapshot files / authors for profiling
    if pre_count > 0:
        pre_hashes = commits.loc[pre_mask, "COMMIT_HASH"].tolist()
        if len(pre_hashes) > 0:
            placeholders = ",".join("?" * len(pre_hashes))
            file_q = (
                f"SELECT COUNT(DISTINCT FILE) AS n "
                f"FROM GIT_COMMITS_CHANGES "
                f"WHERE PROJECT_ID = ? AND COMMIT_HASH IN ({placeholders})"
            )
            distinct_files = pd.read_sql_query(file_q, conn, params=[project_id, *pre_hashes]).iloc[0]["n"]
        else:
            distinct_files = 0
        distinct_authors = int(commits.loc[pre_mask, "AUTHOR"].nunique())
    else:
        distinct_files = 0
        distinct_authors = 0

    eligible = pre_count >= min_pre and post_count >= min_post
    reason = ""
    if not eligible:
        reasons = []
        if pre_count < min_pre:
            reasons.append(f"pre<{min_pre}")
        if post_count < min_post:
            reasons.append(f"post<{min_post}")
        reason = ",".join(reasons)

    return ProjectSnapshot(
        project_id=project_id,
        snapshot_date=snapshot,
        window_months=window_months,
        first_commit=pd.Timestamp(first),
        last_commit=pd.Timestamp(last),
        total_commits=int(len(commits)),
        pre_snapshot_commits=pre_count,
        post_snapshot_commits=post_count,
        distinct_files_pre=int(distinct_files),
        distinct_authors_pre=int(distinct_authors),
        eligible=bool(eligible),
        exclusion_reason=reason,
    )


def compute_all_snapshots(
    conn: sqlite3.Connection,
    projects: Optional[list[str]] = None,
    strategy: str = SNAPSHOT_STRATEGY,
    window_months: int = OBSERVATION_WINDOW_MONTHS,
) -> pd.DataFrame:
    """Compute snapshots for every project in the dataset (or a given list)."""
    if projects is None:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT PROJECT_ID FROM GIT_COMMITS ORDER BY PROJECT_ID")
        projects = [r[0] for r in cur.fetchall()]

    rows: list[dict] = []
    for pid in projects:
        snap = compute_project_snapshot(conn, pid, strategy, window_months)
        rows.append(snap.to_dict())
    return pd.DataFrame(rows)


def load_snapshots(path: Path) -> pd.DataFrame:
    """Load previously computed snapshots from disk (Parquet)."""
    df = pd.read_parquet(path)
    for col in ("snapshot_date", "window_end", "first_commit", "last_commit"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df
