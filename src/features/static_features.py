"""
Snapshot-aware static feature extraction at (project, basename) granularity.

This module consumes the cleaned parquets produced by Stage 3 and emits one
row per (project, basename) with:

- **Per-basename aggregated SonarQube issue features at snapshot ``t``**
  (counts by severity / by type, technical-debt minutes, distinct rules).
  These are the "static code features" of Proposal Table 1 at the
  granularity imposed by the dataset (see RESEARCH_LOG.md 2026-04-23).
- **Project-level context features from the most recent
  ``SONAR_ANALYSIS`` with ``analysis_date <= t``**, replicated for every
  basename in the project. These capture project-wide maintainability
  indicators (NCLOC, COMPLEXITY, SQALE_INDEX, COVERAGE, etc.) as the
  macro-context each file lives in.
- **Size proxy from Git history**: cumulative ``LINES_ADDED`` minus
  ``LINES_REMOVED`` up to ``t`` (floored at zero) - a file-level
  substitute for per-file NCLOC that the dataset does not store.

Columns that could leak the severity-baseline label (``n_blocker``,
``n_critical``, etc.) are computed here but the dataset-build stage
selectively drops them when training on the severity variant.

References
----------
- Lenarduzzi, V., et al. (2019). The Technical Debt Dataset. PROMISE.
- Tsoukalas, D., et al. (2020). Machine learning for technical debt
  identification. IEEE TSE.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.szz import basename_universe_at_snapshot, open_issues_at_snapshot  # noqa: E402


# ---------------------------------------------------------------------------
# Issue-based static features per basename at snapshot
# ---------------------------------------------------------------------------
def sonar_issue_features_at_snapshot(
    sonar_issues: pd.DataFrame,
    project_id: str,
    snapshot: pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate SonarQube issues open at ``snapshot`` into per-basename features.

    Columns produced (all non-negative integers or floats):
    - ``n_issues_open``
    - ``n_blocker``, ``n_critical``, ``n_major``, ``n_minor``, ``n_info``
    - ``n_code_smell``, ``n_bug``, ``n_vulnerability``
    - ``total_debt_minutes`` - sum of ``DEBT`` (minutes of remediation effort)
    - ``total_effort_minutes`` - sum of ``EFFORT`` (if present)
    - ``n_distinct_rules``
    - ``max_severity_rank`` in ``{0..4}`` where INFO=0 and BLOCKER=4
    """
    open_iss = open_issues_at_snapshot(sonar_issues, project_id, snapshot)
    if open_iss.empty:
        return pd.DataFrame(
            columns=[
                "basename",
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
            ]
        )

    rank_map = {"INFO": 0, "MINOR": 1, "MAJOR": 2, "CRITICAL": 3, "BLOCKER": 4}
    open_iss = open_iss.assign(_rank=open_iss["SEVERITY"].map(rank_map).fillna(0).astype("int64"))

    # Severity counts via pivot
    sev = (
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
    )
    for col in ("n_blocker", "n_critical", "n_major", "n_minor", "n_info"):
        if col not in sev.columns:
            sev[col] = 0

    # Type counts via pivot
    typ = (
        open_iss.pivot_table(
            index="basename", columns="TYPE", values="ISSUE_KEY", aggfunc="count", fill_value=0
        )
        .rename(
            columns={
                "CODE_SMELL": "n_code_smell",
                "BUG": "n_bug",
                "VULNERABILITY": "n_vulnerability",
            }
        )
    )
    for col in ("n_code_smell", "n_bug", "n_vulnerability"):
        if col not in typ.columns:
            typ[col] = 0

    agg = open_iss.groupby("basename").agg(
        n_issues_open=("ISSUE_KEY", "count"),
        total_debt_minutes=("DEBT", "sum"),
        total_effort_minutes=("EFFORT", "sum"),
        n_distinct_rules=("RULE", "nunique"),
        max_severity_rank=("_rank", "max"),
    )

    out = agg.join(sev).join(typ).reset_index()
    for c in (
        "n_blocker",
        "n_critical",
        "n_major",
        "n_minor",
        "n_info",
        "n_code_smell",
        "n_bug",
        "n_vulnerability",
        "n_issues_open",
        "n_distinct_rules",
        "max_severity_rank",
    ):
        out[c] = out[c].fillna(0).astype("int64")
    for c in ("total_debt_minutes", "total_effort_minutes"):
        out[c] = out[c].fillna(0.0).astype(float)
    return out


# ---------------------------------------------------------------------------
# Git-derived pseudo size at snapshot
# ---------------------------------------------------------------------------
def git_pseudo_size_at_snapshot(
    changes: pd.DataFrame,
    project_id: str,
    snapshot: pd.Timestamp,
) -> pd.DataFrame:
    """Cumulative added - removed lines up to snapshot, floored at zero.

    A file-level approximation of current NCLOC when true per-file LOC is not
    available in the dataset. See Proposal 3.3 and thesis Threats to
    Construct Validity.
    """
    t = snapshot
    if t.tz is None:
        t = t.tz_localize("UTC")
    ch = changes[changes["PROJECT_ID"] == project_id]
    ch = ch[ch["DATE"] <= t]
    if ch.empty:
        return pd.DataFrame(columns=["basename", "pseudo_ncloc_at_t"])
    g = ch.groupby("basename").agg(
        _add_sum=("LINES_ADDED", "sum"),
        _rem_sum=("LINES_REMOVED", "sum"),
    )
    g["pseudo_ncloc_at_t"] = (g["_add_sum"] - g["_rem_sum"]).clip(lower=0).astype("int64")
    return g[["pseudo_ncloc_at_t"]].reset_index()


# ---------------------------------------------------------------------------
# Project-level context features at snapshot
# ---------------------------------------------------------------------------
_PROJECT_CONTEXT_COLS: tuple[str, ...] = (
    # Size
    "ncloc",
    "lines",
    "classes",
    "files",
    "functions",
    "statements",
    "comment_lines",
    # Complexity
    "complexity",
    "cognitive_complexity",
    "file_complexity",
    "function_complexity",
    "class_complexity",
    # Density / quality
    "comment_lines_density",
    "duplicated_lines",
    "duplicated_lines_density",
    "duplicated_blocks",
    "duplicated_files",
    "coverage",
    "line_coverage",
    "lines_to_cover",
    "uncovered_lines",
    # Technical debt at project scope (structural context - not label leakage)
    "sqale_index",
    "sqale_debt_ratio",
    "sqale_rating",
    "reliability_rating",
    "security_rating",
    "reliability_remediation_effort",
    "security_remediation_effort",
    "open_issues",
)


def project_context_at_snapshot(
    sonar_measures: pd.DataFrame,
    project_id: str,
    snapshot: pd.Timestamp,
) -> dict:
    """Return project-level SonarQube measures from the latest analysis <= ``t``.

    Returns an empty dict if no analysis exists before the snapshot (very rare,
    would indicate a misconfigured project).
    """
    t = snapshot
    if t.tz is None:
        t = t.tz_localize("UTC")
    m = sonar_measures[sonar_measures["project_id"] == project_id]
    m = m[m["analysis_date"] <= t]
    if m.empty:
        return {}
    # Take the most recent analysis at or before t
    row = m.sort_values("analysis_date").iloc[-1]
    out = {f"project_{c}": row.get(c) for c in _PROJECT_CONTEXT_COLS if c in m.columns}
    out["project_context_analysis_date"] = row["analysis_date"]
    return out


# ---------------------------------------------------------------------------
# Combined static features for a single project
# ---------------------------------------------------------------------------
def build_static_features_for_project(
    project_id: str,
    snapshot: pd.Timestamp,
    sonar_issues: pd.DataFrame,
    sonar_measures: pd.DataFrame,
    changes: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the full per-basename static feature row for one project."""
    universe = basename_universe_at_snapshot(changes, project_id, snapshot)
    if universe.empty:
        return universe

    issue_feats = sonar_issue_features_at_snapshot(sonar_issues, project_id, snapshot)
    size_feats = git_pseudo_size_at_snapshot(changes, project_id, snapshot)
    ctx = project_context_at_snapshot(sonar_measures, project_id, snapshot)

    df = universe.merge(issue_feats, on="basename", how="left")
    df = df.merge(size_feats, on="basename", how="left")

    # Fill issue-count NaNs (files with zero issues) with zero
    issue_fill_zero = [
        "n_issues_open",
        "n_blocker",
        "n_critical",
        "n_major",
        "n_minor",
        "n_info",
        "n_code_smell",
        "n_bug",
        "n_vulnerability",
        "n_distinct_rules",
        "max_severity_rank",
    ]
    for c in issue_fill_zero:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype("int64")
    for c in ("total_debt_minutes", "total_effort_minutes"):
        if c in df.columns:
            df[c] = df[c].fillna(0.0).astype(float)
    if "pseudo_ncloc_at_t" in df.columns:
        df["pseudo_ncloc_at_t"] = df["pseudo_ncloc_at_t"].fillna(0).astype("int64")

    # Derived ratios
    df["issue_density"] = np.where(
        df.get("pseudo_ncloc_at_t", 0) > 0,
        df["n_issues_open"] / df["pseudo_ncloc_at_t"].replace(0, np.nan),
        0.0,
    ).astype(float)
    df["debt_per_loc"] = np.where(
        df.get("pseudo_ncloc_at_t", 0) > 0,
        df["total_debt_minutes"] / df["pseudo_ncloc_at_t"].replace(0, np.nan),
        0.0,
    ).astype(float)

    # Attach project-level context (same for every row of this project)
    for k, v in ctx.items():
        df[k] = v

    df["snapshot_date"] = snapshot
    return df
