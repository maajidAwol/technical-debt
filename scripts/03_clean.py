"""
Stage 3 - Clean, normalize and persist the raw dataset tables.

Reads the eligible-projects list produced by Stage 2 and writes tidy Parquet
files to ``data/processed/`` that downstream stages will consume:

- ``clean_git_commits.parquet``           - master-branch commits with
  bug-fix flags + extracted Jira keys.
- ``clean_git_commits_changes.parquet``   - per-file Java-source changes
  with normalized paths, numeric churn columns.
- ``clean_sonar_issues.parquet``          - per-file SonarQube issues with
  normalized ``file_path``.
- ``clean_sonar_measures.parquet``        - project-level Sonar measures
  joined to ``SONAR_ANALYSIS.DATE``.
- ``clean_szz.parquet``                   - SZZ pairs with enriched fix /
  induce dates.
- ``clean_jira_issues.parquet``           - Jira issues with parsed dates
  and an ``is_bug`` flag.

Also writes a join-sanity-check table
``results/tables/path_overlap_report.csv`` reporting the fraction of
``SONAR_ISSUES.file_path`` values that match a file appearing in
``GIT_COMMITS_CHANGES`` per project. This is the critical integrity
measurement for later feature joins.

Run
---
.. code-block:: bash

    .\\venv\\Scripts\\python.exe scripts/03_clean.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DATA_DIR, TABLES_DIR  # noqa: E402
from src.data.clean import (  # noqa: E402
    clean_git_commits,
    clean_git_commits_changes,
    clean_jira_issues,
    clean_sonar_issues,
    clean_sonar_measures_with_dates,
    clean_szz_with_dates,
)
from src.data.load_data import get_connection  # noqa: E402
from src.data.snapshot import load_snapshots  # noqa: E402


def _fmt_rows(n: int) -> str:
    return f"{n:>12,}"


def main() -> None:
    t0 = time.time()
    snap_path = PROCESSED_DATA_DIR / "project_snapshots.parquet"
    if not snap_path.exists():
        raise FileNotFoundError(
            f"Missing {snap_path}. Run scripts/02_profile_projects.py first."
        )

    snaps = load_snapshots(snap_path)
    eligible = snaps[snaps["eligible"]].copy()
    projects = eligible["project_id"].tolist()
    print(f"[Stage 3] Eligible projects    : {len(projects)}")
    print(f"[Stage 3] Starting cleaning ...")

    with get_connection() as conn:
        print("[Stage 3] (1/6) GIT_COMMITS  ...", end=" ", flush=True)
        t = time.time()
        gc = clean_git_commits(conn, projects)
        print(f"{_fmt_rows(len(gc))} rows  ({time.time()-t:.1f}s)")

        print("[Stage 3] (2/6) GIT_COMMITS_CHANGES (Java only) ...", end=" ", flush=True)
        t = time.time()
        gcc = clean_git_commits_changes(conn, projects, java_only=True)
        print(f"{_fmt_rows(len(gcc))} rows  ({time.time()-t:.1f}s)")

        print("[Stage 3] (3/6) SONAR_ISSUES (Java only) ...", end=" ", flush=True)
        t = time.time()
        si = clean_sonar_issues(conn, projects, java_only=True)
        print(f"{_fmt_rows(len(si))} rows  ({time.time()-t:.1f}s)")

        print("[Stage 3] (4/6) SONAR_MEASURES (with dates) ...", end=" ", flush=True)
        t = time.time()
        sm = clean_sonar_measures_with_dates(conn, projects)
        print(f"{_fmt_rows(len(sm))} rows  ({time.time()-t:.1f}s)")

        print("[Stage 3] (5/6) SZZ (with dates) ...", end=" ", flush=True)
        t = time.time()
        szz = clean_szz_with_dates(conn, projects, gc)
        print(f"{_fmt_rows(len(szz))} rows  ({time.time()-t:.1f}s)")

        print("[Stage 3] (6/6) JIRA_ISSUES ...", end=" ", flush=True)
        t = time.time()
        ji = clean_jira_issues(conn, projects)
        print(f"{_fmt_rows(len(ji))} rows  ({time.time()-t:.1f}s)")

    print("[Stage 3] Writing Parquet ...")
    gc.to_parquet(PROCESSED_DATA_DIR / "clean_git_commits.parquet", index=False)
    gcc.to_parquet(PROCESSED_DATA_DIR / "clean_git_commits_changes.parquet", index=False)
    si.to_parquet(PROCESSED_DATA_DIR / "clean_sonar_issues.parquet", index=False)
    sm.to_parquet(PROCESSED_DATA_DIR / "clean_sonar_measures.parquet", index=False)
    szz.to_parquet(PROCESSED_DATA_DIR / "clean_szz.parquet", index=False)
    ji.to_parquet(PROCESSED_DATA_DIR / "clean_jira_issues.parquet", index=False)

    print("[Stage 3] Computing basename-overlap sanity check ...")
    # TD Dataset v2.0 stores Git file paths at basename granularity for most
    # projects, so ``basename`` is the canonical join key. The thesis reports
    # per-project basename-collision statistics in Threats to Validity.
    gcc_paths = gcc.groupby("PROJECT_ID")["basename"].agg(set).rename("git_basenames")
    si_paths = si.groupby("PROJECT_ID")["basename"].agg(set).rename("sonar_basenames")
    overlap = pd.concat([gcc_paths, si_paths], axis=1)
    for col in ("git_basenames", "sonar_basenames"):
        overlap[col] = overlap[col].apply(lambda x: x if isinstance(x, set) else set())
    overlap["n_sonar_basenames"] = overlap["sonar_basenames"].map(len)
    overlap["n_git_basenames"] = overlap["git_basenames"].map(len)
    overlap["n_intersection"] = overlap.apply(
        lambda r: len(r["sonar_basenames"] & r["git_basenames"]), axis=1
    )
    overlap["sonar_coverage_pct"] = (
        overlap["n_intersection"] / overlap["n_sonar_basenames"].where(overlap["n_sonar_basenames"] > 0)
    ).round(4) * 100
    overlap = overlap[
        ["n_sonar_basenames", "n_git_basenames", "n_intersection", "sonar_coverage_pct"]
    ]

    collision_rows = []
    for pid, sub in si.groupby("PROJECT_ID"):
        files_per_base = sub.groupby("basename")["file_path"].nunique()
        collision_rows.append(
            {
                "PROJECT_ID": pid,
                "sonar_full_paths": int(sub["file_path"].nunique()),
                "sonar_basenames": int(files_per_base.shape[0]),
                "max_files_per_basename": int(files_per_base.max()) if len(files_per_base) else 0,
                "basename_collision_pct": (
                    round((1 - files_per_base.shape[0] / sub["file_path"].nunique()) * 100, 1)
                    if sub["file_path"].nunique()
                    else 0.0
                ),
            }
        )
    collisions = pd.DataFrame(collision_rows).set_index("PROJECT_ID")
    overlap = overlap.join(collisions)
    overlap_path = TABLES_DIR / "path_overlap_report.csv"
    overlap.reset_index().to_csv(overlap_path, index=False)

    print("\n[Stage 3] Row-count summary (written to data/processed/):")
    print(f"   clean_git_commits           {_fmt_rows(len(gc))}")
    print(f"   clean_git_commits_changes   {_fmt_rows(len(gcc))}")
    print(f"   clean_sonar_issues          {_fmt_rows(len(si))}")
    print(f"   clean_sonar_measures        {_fmt_rows(len(sm))}")
    print(f"   clean_szz                   {_fmt_rows(len(szz))}")
    print(f"   clean_jira_issues           {_fmt_rows(len(ji))}")

    print("\n[Stage 3] Bug-fix tag coverage:")
    bf_pct = 100 * gc["is_bugfix"].mean()
    print(f"   is_bugfix = True            {gc['is_bugfix'].sum():,} / {len(gc):,}  ({bf_pct:.1f}%)")
    jk_any = gc["jira_keys"].apply(lambda x: len(x) > 0).mean() * 100
    print(f"   has jira_keys               {jk_any:.1f}% of commits mention a Jira key")

    print("\n[Stage 3] Basename-overlap report (Sonar basenames found in Git history):")
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(overlap.sort_values("sonar_coverage_pct", ascending=False).to_string())

    min_cov = overlap["sonar_coverage_pct"].min()
    avg_cov = overlap["sonar_coverage_pct"].mean()
    med_coll = overlap["basename_collision_pct"].median()
    max_coll = overlap["basename_collision_pct"].max()
    print(
        f"\n[Stage 3] Sonar->Git basename coverage : min={min_cov:.1f}%  avg={avg_cov:.1f}%"
    )
    print(
        f"[Stage 3] Basename collision (SonarQube): median={med_coll:.1f}%  max={max_coll:.1f}%"
    )
    print(f"[Stage 3] Saved overlap report: {overlap_path}")

    print(f"\n[Stage 3] Total elapsed : {time.time() - t0:.1f}s")
    print("[Stage 3] Complete.")


if __name__ == "__main__":
    main()
