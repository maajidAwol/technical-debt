"""
Data cleaning and normalization for the Technical Debt Dataset v2.0.

This module converts raw SQLite tables into tidy pandas DataFrames suitable
for feature extraction (Stage 5), labeling (Stage 4) and cross-table joins
(Stage 6). The main transformations are:

1. **Path normalization** - ``SONAR_ISSUES.COMPONENT`` stores file identifiers
   as ``SonarProjectKey:path/to/File.java`` whereas ``GIT_COMMITS_CHANGES.FILE``
   stores them as plain repo-relative paths. We strip the prefix so the two
   tables join cleanly on ``(PROJECT_ID, file_path)``.
2. **Temporal enrichment** - ``SONAR_MEASURES`` rows carry an analysis key
   but not a date. We join ``SONAR_ANALYSIS`` to attach ``DATE`` and
   ``REVISION`` (Git hash of the analyzed commit) so measures can be
   temporally aligned to the snapshot ``t``.
3. **Bug-fix tagging** - regex-based flagging of commit messages plus a
   Jira-link flag derived from the pre-populated ``JIRA_ISSUES.HASH``
   column and Jira key mentions in commit messages.
4. **Scope filtering** - restricts to Java source files (excluding
   tests, generated sources, build artefacts) per ``config.py``.
5. **Project filtering** - keeps only projects flagged *eligible* by
   Stage 2.
6. **Type coercion** - parses dates to tz-aware ``datetime64[ns, UTC]``,
   casts numeric columns, removes obvious duplicates.

Each cleaned DataFrame is persisted as Parquet in ``data/processed/``.

References
----------
- Lenarduzzi, V., et al. (2019). The Technical Debt Dataset. PROMISE 2019.
- Mockus, A., Votta, L. (2000). Identifying reasons for software changes
  using historic databases. ICSM 2000 - bug-fix keyword regex basis.
"""
from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    BUG_FIX_KEYWORDS,
    JIRA_ISSUE_KEY_PATTERN,
    PATH_EXCLUSION_PATTERNS,
    SOURCE_FILE_EXTENSIONS,
)


_BUGFIX_REGEX = re.compile("|".join(BUG_FIX_KEYWORDS), re.IGNORECASE)
_JIRA_KEY_REGEX = re.compile(JIRA_ISSUE_KEY_PATTERN)


# ---------------------------------------------------------------------------
# Path normalization
# ---------------------------------------------------------------------------
def normalize_component_path(component: Optional[str]) -> Optional[str]:
    """Strip the ``SonarProjectKey:`` prefix from ``SONAR_ISSUES.COMPONENT``.

    Returns the repo-relative file path, e.g.
    ``"Apache_Cayenne:framework/.../Fault.java"`` -> ``"framework/.../Fault.java"``.
    Components without a colon are returned unchanged. ``None`` values are
    preserved.
    """
    if component is None or not isinstance(component, str):
        return component
    idx = component.find(":")
    if idx < 0:
        return component
    return component[idx + 1 :]


def is_java_source(path: Optional[str]) -> bool:
    """True if ``path`` is a non-test, non-generated Java source file."""
    if path is None or not isinstance(path, str) or not path:
        return False
    p = path.replace("\\", "/").lower()
    if not any(p.endswith(ext) for ext in SOURCE_FILE_EXTENSIONS):
        return False
    return not any(bad in p for bad in PATH_EXCLUSION_PATTERNS)


def extract_basename(path: Optional[str]) -> Optional[str]:
    """Return the final segment of a slash- or backslash-separated path.

    This is the canonical file-identifier key used across the pipeline because
    ``GIT_COMMITS_CHANGES.FILE`` in TD Dataset v2.0 stores only the basename
    for most projects (see RESEARCH_LOG.md 2026-04-23 path-format entry).
    """
    if path is None or not isinstance(path, str) or not path:
        return path
    p = path.replace("\\", "/")
    return p.rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# Commit tagging
# ---------------------------------------------------------------------------
def flag_bugfix_commits(messages: pd.Series) -> pd.Series:
    """Return a boolean Series marking messages that look like bug-fixes.

    Uses the union regex of ``BUG_FIX_KEYWORDS`` (Mockus & Votta 2000).
    Empty / NaN messages return ``False``.
    """
    return messages.fillna("").astype(str).str.contains(_BUGFIX_REGEX, regex=True)


def extract_jira_keys(messages: pd.Series) -> pd.Series:
    """Return a Series of lists of Jira keys mentioned in each commit message."""

    def _find(msg: str) -> list[str]:
        if not isinstance(msg, str) or not msg:
            return []
        return [f"{m.group(1)}-{m.group(2)}" for m in _JIRA_KEY_REGEX.finditer(msg)]

    return messages.apply(_find)


# ---------------------------------------------------------------------------
# Per-table cleaners
# ---------------------------------------------------------------------------
def clean_git_commits(
    conn: sqlite3.Connection,
    projects: Iterable[str],
) -> pd.DataFrame:
    """Load master-branch commits for ``projects`` and add cleaning columns.

    Columns added:
    - ``AUTHOR_DATE`` parsed to UTC-aware datetime
    - ``is_bugfix`` (bool)
    - ``jira_keys`` (list[str])
    """
    ids = ",".join(f"'{p}'" for p in projects)
    sql = (
        "SELECT PROJECT_ID, COMMIT_HASH, COMMIT_MESSAGE, AUTHOR, AUTHOR_DATE, "
        "       COMMITTER, COMMITTER_DATE, MERGE "
        "FROM GIT_COMMITS "
        f"WHERE IN_MAIN_BRANCH = 'True' AND PROJECT_ID IN ({ids})"
    )
    df = pd.read_sql_query(sql, conn)
    df["AUTHOR_DATE"] = pd.to_datetime(df["AUTHOR_DATE"], errors="coerce", utc=True)
    df["COMMITTER_DATE"] = pd.to_datetime(df["COMMITTER_DATE"], errors="coerce", utc=True)
    df = df.dropna(subset=["AUTHOR_DATE", "COMMIT_HASH"]).copy()
    df["is_bugfix"] = flag_bugfix_commits(df["COMMIT_MESSAGE"])
    df["jira_keys"] = extract_jira_keys(df["COMMIT_MESSAGE"])
    df["MERGE"] = df["MERGE"].astype(str).str.lower().isin(("true", "1"))
    df = df.drop_duplicates(subset=["PROJECT_ID", "COMMIT_HASH"])
    return df.reset_index(drop=True)


def clean_git_commits_changes(
    conn: sqlite3.Connection,
    projects: Iterable[str],
    java_only: bool = True,
) -> pd.DataFrame:
    """Load per-file changes for ``projects``; optionally keep only Java sources.

    Output columns: ``PROJECT_ID``, ``COMMIT_HASH``, ``file_path`` (normalized),
    ``DATE`` (UTC), ``COMMITTER_ID``, ``LINES_ADDED``, ``LINES_REMOVED``.
    """
    ids = ",".join(f"'{p}'" for p in projects)
    sql = (
        "SELECT PROJECT_ID, COMMIT_HASH, FILE, DATE, COMMITTER_ID, "
        "       LINES_ADDED, LINES_REMOVED "
        "FROM GIT_COMMITS_CHANGES "
        f"WHERE PROJECT_ID IN ({ids})"
    )
    df = pd.read_sql_query(sql, conn)
    df = df.rename(columns={"FILE": "file_path"})
    df["file_path"] = df["file_path"].astype(str).str.replace("\\", "/", regex=False)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce", utc=True)
    for c in ("LINES_ADDED", "LINES_REMOVED"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
    df = df.dropna(subset=["file_path", "COMMIT_HASH", "DATE"])
    df = df[df["file_path"].str.len() > 0]
    if java_only:
        df = df[df["file_path"].apply(is_java_source)]
    df["basename"] = df["file_path"].apply(extract_basename)
    df = df.drop_duplicates(subset=["PROJECT_ID", "COMMIT_HASH", "file_path"])
    return df.reset_index(drop=True)


def clean_sonar_issues(
    conn: sqlite3.Connection,
    projects: Iterable[str],
    java_only: bool = True,
) -> pd.DataFrame:
    """Load SONAR_ISSUES, normalize COMPONENT to ``file_path``, coerce types."""
    ids = ",".join(f"'{p}'" for p in projects)
    sql = (
        "SELECT PROJECT_ID, ISSUE_KEY, TYPE, RULE, SEVERITY, STATUS, RESOLUTION, "
        "       EFFORT, DEBT, CREATION_DATE, CLOSE_DATE, COMPONENT, "
        "       START_LINE, END_LINE "
        "FROM SONAR_ISSUES "
        f"WHERE PROJECT_ID IN ({ids})"
    )
    df = pd.read_sql_query(sql, conn)
    df["file_path"] = df["COMPONENT"].apply(normalize_component_path)
    df["file_path"] = df["file_path"].astype(str).str.replace("\\", "/", regex=False)
    df["CREATION_DATE"] = pd.to_datetime(df["CREATION_DATE"], errors="coerce", utc=True)
    df["CLOSE_DATE"] = pd.to_datetime(df["CLOSE_DATE"], errors="coerce", utc=True)
    for c in ("EFFORT", "DEBT", "START_LINE", "END_LINE"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["file_path", "ISSUE_KEY"])
    df = df[df["file_path"].str.len() > 0]
    if java_only:
        df = df[df["file_path"].apply(is_java_source)]
    df["basename"] = df["file_path"].apply(extract_basename)
    return df.drop(columns=["COMPONENT"]).reset_index(drop=True)


def clean_sonar_measures_with_dates(
    conn: sqlite3.Connection,
    projects: Iterable[str],
) -> pd.DataFrame:
    """Load SONAR_MEASURES and join SONAR_ANALYSIS to attach DATE + REVISION.

    Returns one row per ``(PROJECT_ID, ANALYSIS_KEY)`` with ``analysis_date``
    (UTC-aware) and the canonical subset of project-level metrics used as
    features (NCLOC, COMPLEXITY, SQALE_INDEX, ...). Leaky raw severity counts
    are kept here because they will be selectively dropped at dataset-build
    time depending on the label variant.
    """
    ids = ",".join(f"'{p}'" for p in projects)
    metric_cols = [
        "NCLOC",
        "LINES",
        "CLASSES",
        "FILES",
        "FUNCTIONS",
        "STATEMENTS",
        "COMPLEXITY",
        "COGNITIVE_COMPLEXITY",
        "FILE_COMPLEXITY",
        "FUNCTION_COMPLEXITY",
        "CLASS_COMPLEXITY",
        "COMMENT_LINES",
        "COMMENT_LINES_DENSITY",
        "DUPLICATED_LINES",
        "DUPLICATED_LINES_DENSITY",
        "DUPLICATED_BLOCKS",
        "DUPLICATED_FILES",
        "COVERAGE",
        "LINE_COVERAGE",
        "LINES_TO_COVER",
        "UNCOVERED_LINES",
        "VIOLATIONS",
        "BLOCKER_VIOLATIONS",
        "CRITICAL_VIOLATIONS",
        "MAJOR_VIOLATIONS",
        "MINOR_VIOLATIONS",
        "INFO_VIOLATIONS",
        "CODE_SMELLS",
        "BUGS",
        "VULNERABILITIES",
        "SQALE_INDEX",
        "SQALE_DEBT_RATIO",
        "SQALE_RATING",
        "RELIABILITY_RATING",
        "SECURITY_RATING",
        "RELIABILITY_REMEDIATION_EFFORT",
        "SECURITY_REMEDIATION_EFFORT",
        "OPEN_ISSUES",
    ]
    cols_sql = ", ".join(f"m.{c}" for c in metric_cols)
    sql = (
        f"SELECT m.PROJECT_ID, m.ANALYSIS_KEY, a.DATE AS analysis_date, "
        f"       a.REVISION AS analysis_revision, {cols_sql} "
        "FROM SONAR_MEASURES m "
        "LEFT JOIN SONAR_ANALYSIS a "
        "       ON a.PROJECT_ID = m.PROJECT_ID "
        "      AND a.ANALYSIS_KEY = m.ANALYSIS_KEY "
        f"WHERE m.PROJECT_ID IN ({ids})"
    )
    df = pd.read_sql_query(sql, conn)
    df["analysis_date"] = pd.to_datetime(df["analysis_date"], errors="coerce", utc=True)
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["analysis_date"]).copy()
    df.columns = [c.lower() if c.isupper() else c for c in df.columns]
    return df.sort_values(["project_id", "analysis_date"]).reset_index(drop=True)


def clean_szz_with_dates(
    conn: sqlite3.Connection,
    projects: Iterable[str],
    git_commits_clean: pd.DataFrame,
) -> pd.DataFrame:
    """Load SZZ links and enrich with fault-fixing and fault-inducing dates.

    Joins to ``git_commits_clean`` on ``(PROJECT_ID, COMMIT_HASH)`` to attach
    ``AUTHOR_DATE`` for both sides of the pair.
    """
    ids = ",".join(f"'{p}'" for p in projects)
    sql = (
        "SELECT PROJECT_ID, FAULT_FIXING_COMMIT_HASH, FAULT_INDUCING_COMMIT_HASH "
        "FROM SZZ_FAULT_INDUCING_COMMITS "
        f"WHERE PROJECT_ID IN ({ids})"
    )
    df = pd.read_sql_query(sql, conn)

    commits_small = git_commits_clean[["PROJECT_ID", "COMMIT_HASH", "AUTHOR_DATE"]]
    df = df.merge(
        commits_small.rename(
            columns={"COMMIT_HASH": "FAULT_FIXING_COMMIT_HASH", "AUTHOR_DATE": "fix_date"}
        ),
        on=["PROJECT_ID", "FAULT_FIXING_COMMIT_HASH"],
        how="left",
    )
    df = df.merge(
        commits_small.rename(
            columns={"COMMIT_HASH": "FAULT_INDUCING_COMMIT_HASH", "AUTHOR_DATE": "induce_date"}
        ),
        on=["PROJECT_ID", "FAULT_INDUCING_COMMIT_HASH"],
        how="left",
    )
    return df.drop_duplicates().reset_index(drop=True)


def clean_jira_issues(
    conn: sqlite3.Connection,
    projects: Iterable[str],
) -> pd.DataFrame:
    """Load JIRA_ISSUES for projects with dates parsed and a ``is_bug`` flag."""
    ids = ",".join(f"'{p}'" for p in projects)
    sql = (
        "SELECT PROJECT_ID, KEY, PRIORITY, TYPE, STATUS, RESOLUTION, "
        "       CREATION_DATE, RESOLUTION_DATE, UPDATE_DATE, HASH, COMMIT_DATE "
        "FROM JIRA_ISSUES "
        f"WHERE PROJECT_ID IN ({ids})"
    )
    df = pd.read_sql_query(sql, conn)
    for c in ("CREATION_DATE", "RESOLUTION_DATE", "UPDATE_DATE", "COMMIT_DATE"):
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    df["is_bug"] = df["TYPE"].fillna("").str.lower().eq("bug")
    return df.reset_index(drop=True)
