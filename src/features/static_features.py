"""
Static feature extraction at (project, basename) granularity.

Emits two of the five feature families (10 features total):

Family 1 - Size / Complexity (project-level context replicated per file):
    ncloc, complexity, cognitive_complexity, functions, classes

Family 2 - Static Debt (per-file aggregates over SONAR_ISSUES at t):
    n_code_smells, n_bugs, total_debt_minutes, issue_density,
    duplicated_lines_density

``ncloc`` and ``duplicated_lines_density`` come from the most recent
SONAR_ANALYSIS at or before ``t`` (project-level snapshot). All issue
aggregates use ``CREATION_DATE <= t`` and respect the issue's
``CLOSE_DATE`` window (open at snapshot).

Leakage note: raw severity counts (n_blocker, n_critical, etc.) are
deliberately NOT emitted - the dual-signal label S1 keys on
BLOCKER/CRITICAL counts, so including them as features would leak the
label. See assertion in scripts/06_build_dataset.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.szz import basename_universe_at_snapshot, open_issues_at_snapshot  # noqa: E402


# ---------------------------------------------------------------------------
# Family 2 - Static Debt (per-file from SONAR_ISSUES)
# ---------------------------------------------------------------------------
def _issue_aggregates_at_snapshot(
    sonar_issues: pd.DataFrame,
    project_id: str,
    snapshot: pd.Timestamp,
) -> pd.DataFrame:
    """Per-basename counts of CODE_SMELL, BUG, and total debt minutes."""
    open_iss = open_issues_at_snapshot(sonar_issues, project_id, snapshot)
    cols = ["basename", "n_code_smells", "n_bugs", "total_debt_minutes", "n_issues_open"]
    if open_iss.empty:
        return pd.DataFrame(columns=cols)

    issue_type = open_iss["TYPE"].astype(str).str.upper()
    smell_mask = issue_type == "CODE_SMELL"
    bug_mask = issue_type == "BUG"

    n_smells = open_iss.loc[smell_mask].groupby("basename").size().rename("n_code_smells")
    n_bugs = open_iss.loc[bug_mask].groupby("basename").size().rename("n_bugs")
    debt = open_iss.groupby("basename")["DEBT"].sum(min_count=1).rename("total_debt_minutes")
    n_open = open_iss.groupby("basename").size().rename("n_issues_open")

    out = pd.concat([n_smells, n_bugs, debt, n_open], axis=1).reset_index()
    out["n_code_smells"] = out["n_code_smells"].fillna(0).astype("int64")
    out["n_bugs"] = out["n_bugs"].fillna(0).astype("int64")
    out["n_issues_open"] = out["n_issues_open"].fillna(0).astype("int64")
    out["total_debt_minutes"] = out["total_debt_minutes"].fillna(0.0).astype(float)
    return out


# ---------------------------------------------------------------------------
# Family 1 - Size / Complexity (project-level context at t)
# ---------------------------------------------------------------------------
_PROJECT_LEVEL_METRICS = (
    "ncloc",
    "complexity",
    "cognitive_complexity",
    "functions",
    "classes",
    "duplicated_lines_density",
)


def _project_metrics_at_snapshot(
    sonar_measures: pd.DataFrame,
    project_id: str,
    snapshot: pd.Timestamp,
) -> dict:
    """Most recent SONAR_ANALYSIS metrics at or before ``t``.

    TD Dataset v2 stores SonarQube measures at project granularity (no
    per-file COMPONENT column), so the same row is replicated for every
    basename in the project. Empty dict if no analysis exists before t.
    """
    t = snapshot
    if t.tz is None:
        t = t.tz_localize("UTC")
    m = sonar_measures[sonar_measures["project_id"] == project_id]
    m = m[m["analysis_date"] <= t]
    if m.empty:
        return {}
    row = m.sort_values("analysis_date").iloc[-1]
    return {c: row.get(c) for c in _PROJECT_LEVEL_METRICS if c in m.columns}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_static_features_for_project(
    project_id: str,
    snapshot: pd.Timestamp,
    sonar_issues: pd.DataFrame,
    sonar_measures: pd.DataFrame,
    changes: pd.DataFrame,
) -> pd.DataFrame:
    """Return ``(project_id, basename, <10 static features>)`` for one project."""
    universe = basename_universe_at_snapshot(changes, project_id, snapshot)
    if universe.empty:
        return universe.assign(
            **{
                "ncloc": 0.0,
                "complexity": 0.0,
                "cognitive_complexity": 0.0,
                "functions": 0.0,
                "classes": 0.0,
                "n_code_smells": 0,
                "n_bugs": 0,
                "total_debt_minutes": 0.0,
                "issue_density": 0.0,
                "duplicated_lines_density": 0.0,
            }
        )

    issue_feats = _issue_aggregates_at_snapshot(sonar_issues, project_id, snapshot)
    ctx = _project_metrics_at_snapshot(sonar_measures, project_id, snapshot)

    df = universe.merge(issue_feats, on="basename", how="left")
    for c in ("n_code_smells", "n_bugs", "n_issues_open"):
        df[c] = df[c].fillna(0).astype("int64")
    df["total_debt_minutes"] = df["total_debt_minutes"].fillna(0.0).astype(float)

    # Project-level metrics replicated per file (NaN -> 0.0 if no analysis <= t).
    for col in ("ncloc", "complexity", "cognitive_complexity", "functions", "classes",
                "duplicated_lines_density"):
        val = ctx.get(col)
        df[col] = float(val) if val is not None and not pd.isna(val) else 0.0

    # Issue density (per spec). Guard ncloc == 0.
    ncloc_safe = df["ncloc"].replace(0, np.nan)
    df["issue_density"] = np.where(
        df["ncloc"] > 0,
        (df["n_issues_open"] / ncloc_safe).fillna(0.0),
        0.0,
    ).astype(float)

    # Drop helper column not in the 27 features.
    df = df.drop(columns=["n_issues_open"])
    df["snapshot_date"] = snapshot
    return df
