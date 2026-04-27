"""
Thin SQL readers for the Technical Debt Dataset v2.0.

This module contains ONLY raw SELECT helpers - no cleaning, no aggregation,
no domain logic. Cleaning lives in ``src.data.clean`` and labeling in
``src.data.labeling``.

Every function accepts an optional ``sqlite3.Connection``; callers that open
many queries should share one connection to avoid file-open overhead on a
1.5 GB database.

References
----------
Lenarduzzi, V., Saarimaki, N., Taibi, D. (2019). The Technical Debt Dataset.
Proc. PROMISE 2019.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import TD_DATASET_PATH  # noqa: E402


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------
def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open a SQLite connection to the Technical Debt Dataset."""
    db_path = Path(db_path) if db_path else TD_DATASET_PATH
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}\n"
            f"Download: https://github.com/clowee/The-Technical-Debt-Dataset/releases"
        )
    return sqlite3.connect(str(db_path))


def get_table_names(conn: sqlite3.Connection) -> list[str]:
    """List all tables in the database."""
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    return [r[0] for r in cur.fetchall()]


def get_table_schema(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    """Return PRAGMA table_info rows for a table."""
    return pd.read_sql_query(f"PRAGMA table_info({table});", conn)


def get_row_count(conn: sqlite3.Connection, table: str) -> int:
    """Return COUNT(*) of a table."""
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    return cur.fetchone()[0]


def get_sample_rows(conn: sqlite3.Connection, table: str, n: int = 3) -> pd.DataFrame:
    """Return the first ``n`` rows of a table."""
    return pd.read_sql_query(f"SELECT * FROM {table} LIMIT {int(n)};", conn)


# ---------------------------------------------------------------------------
# Table-specific loaders
# ---------------------------------------------------------------------------
def list_projects(conn: sqlite3.Connection) -> list[str]:
    """Return the sorted list of distinct project IDs using GIT_COMMITS."""
    try:
        rows = pd.read_sql_query(
            "SELECT DISTINCT PROJECT_ID FROM GIT_COMMITS ORDER BY PROJECT_ID",
            conn,
        )
    except Exception:
        rows = pd.read_sql_query(
            "SELECT DISTINCT PROJECT_ID FROM SONAR_ISSUES ORDER BY PROJECT_ID",
            conn,
        )
    return rows["PROJECT_ID"].dropna().astype(str).tolist()


def load_git_commits(
    conn: sqlite3.Connection,
    projects: Optional[Iterable[str]] = None,
    main_branch_only: bool = True,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load GIT_COMMITS, optionally restricted to projects / main branch.

    Column names verified in Stage 1: ``COMMIT_HASH``, ``COMMIT_MESSAGE``,
    ``AUTHOR_DATE``, ``COMMITTER_DATE``, ``IN_MAIN_BRANCH`` (stored as text
    strings ``'True'``/``'False'``).
    """
    cols_expr = ", ".join(columns) if columns else "*"
    sql = f"SELECT {cols_expr} FROM GIT_COMMITS"
    where = []
    if main_branch_only:
        where.append("IN_MAIN_BRANCH = 'True'")
    if projects:
        ids = ",".join(f"'{p}'" for p in projects)
        where.append(f"PROJECT_ID IN ({ids})")
    if where:
        sql += " WHERE " + " AND ".join(where)
    df = pd.read_sql_query(sql, conn)
    if "AUTHOR_DATE" in df.columns:
        df["AUTHOR_DATE"] = pd.to_datetime(df["AUTHOR_DATE"], errors="coerce", utc=True)
    if "COMMITTER_DATE" in df.columns:
        df["COMMITTER_DATE"] = pd.to_datetime(df["COMMITTER_DATE"], errors="coerce", utc=True)
    return df


def load_git_commits_changes(
    conn: sqlite3.Connection,
    projects: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load GIT_COMMITS_CHANGES, optionally restricted to projects."""
    sql = "SELECT * FROM GIT_COMMITS_CHANGES"
    if projects:
        ids = ",".join(f"'{p}'" for p in projects)
        sql += f" WHERE PROJECT_ID IN ({ids})"
    return pd.read_sql_query(sql, conn)


def load_sonar_measures(
    conn: sqlite3.Connection,
    projects: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load SONAR_MEASURES; schema is wide (one column per metric)."""
    sql = "SELECT * FROM SONAR_MEASURES"
    if projects:
        ids = ",".join(f"'{p}'" for p in projects)
        sql += f" WHERE PROJECT_ID IN ({ids})"
    return pd.read_sql_query(sql, conn)


def load_sonar_issues(
    conn: sqlite3.Connection,
    projects: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load SONAR_ISSUES."""
    sql = "SELECT * FROM SONAR_ISSUES"
    if projects:
        ids = ",".join(f"'{p}'" for p in projects)
        sql += f" WHERE PROJECT_ID IN ({ids})"
    return pd.read_sql_query(sql, conn)


def load_jira_issues(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load the JIRA_ISSUES table. Schema is confirmed by Stage 1."""
    return pd.read_sql_query("SELECT * FROM JIRA_ISSUES", conn)


def load_szz_fault_inducing(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load SZZ fault-inducing / fault-fixing commit links.

    Columns verified in Stage 1: ``PROJECT_ID``, ``FAULT_FIXING_COMMIT_HASH``,
    ``FAULT_INDUCING_COMMIT_HASH``.
    """
    return pd.read_sql_query("SELECT * FROM SZZ_FAULT_INDUCING_COMMITS", conn)


def load_sonar_analysis(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load SONAR_ANALYSIS (maps ANALYSIS_KEY to snapshot DATE + Git REVISION).

    Columns verified in Stage 1: ``PROJECT_ID``, ``ANALYSIS_KEY``, ``DATE``,
    ``REVISION``. This table is the bridge between ``SONAR_MEASURES`` snapshots
    and Git history (via REVISION hash and DATE).
    """
    df = pd.read_sql_query("SELECT * FROM SONAR_ANALYSIS", conn)
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce", utc=True)
    return df


def load_projects_meta(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load the PROJECTS metadata table (31 rows, PROJECT_ID + GIT_LINK + JIRA_LINK)."""
    return pd.read_sql_query("SELECT * FROM PROJECTS", conn)


def load_refactorings(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load refactoring records (optional qualitative validation)."""
    try:
        return pd.read_sql_query("SELECT * FROM REFACTORING_MINER", conn)
    except Exception:
        return pd.read_sql_query("SELECT * FROM REFACTORINGS", conn)


# ---------------------------------------------------------------------------
# Windowed helpers
# ---------------------------------------------------------------------------
def commits_in_window(
    conn: sqlite3.Connection,
    project_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    main_branch_only: bool = True,
) -> pd.DataFrame:
    """Return commits in the half-open window (start, end] for a project."""
    parts = ["PROJECT_ID = ?", "AUTHOR_DATE > ?", "AUTHOR_DATE <= ?"]
    params: list = [project_id, start.isoformat(), end.isoformat()]
    if main_branch_only:
        parts.append("IN_MAIN_BRANCH = 'True'")
    sql = "SELECT * FROM GIT_COMMITS WHERE " + " AND ".join(parts)
    df = pd.read_sql_query(sql, conn, params=params)
    if "AUTHOR_DATE" in df.columns:
        df["AUTHOR_DATE"] = pd.to_datetime(df["AUTHOR_DATE"], errors="coerce", utc=True)
    return df
