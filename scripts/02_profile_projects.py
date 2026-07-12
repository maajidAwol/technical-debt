"""
Stage 2 - Per-project profiling and snapshot-date selection.

For each project, compute:
- First / last master-branch commit date
- Total commit count (master branch)
- Snapshot date ``t`` (median of master-branch commit dates)
- Pre-snapshot commits (features are drawn from here)
- Post-snapshot commits within a 6-month window (labels are drawn from here)
- Distinct pre-snapshot files and authors
- Eligibility flag: requires >=500 pre-snapshot and >=100 post-snapshot commits

Outputs
-------
- ``data/processed/project_snapshots.parquet``: machine-readable snapshot metadata
- ``results/tables/project_stats.csv``: paper-ready descriptive-statistics table

Run
---
.. code-block:: bash

    .\\venv\\Scripts\\python.exe scripts/02_profile_projects.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    OBSERVATION_WINDOW_MONTHS,
    PROCESSED_DATA_DIR,
    SNAPSHOT_STRATEGY,
    TABLES_DIR,
)
from src.data.load_data import get_connection  # noqa: E402
from src.data.snapshot import compute_all_snapshots  # noqa: E402


def main() -> None:
    t0 = time.time()
    print(f"[Stage 2] Strategy        : {SNAPSHOT_STRATEGY}")
    print(f"[Stage 2] Observation win : {OBSERVATION_WINDOW_MONTHS} months")

    with get_connection() as conn:
        print("[Stage 2] Computing snapshots for every project (master branch only) ...")
        df = compute_all_snapshots(conn)

    print(f"[Stage 2] Computed {len(df)} project profiles in {time.time() - t0:.1f}s")

    out_parquet = PROCESSED_DATA_DIR / "project_snapshots.parquet"
    out_csv = TABLES_DIR / "project_stats.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    # Corpus summary: one row per eligible project for the thesis tables.
    corpus = df[df["eligible"]].copy()
    corpus_summary = pd.DataFrame(
        {
            "project_id": corpus["project_id"],
            "snapshot_date": corpus["snapshot_date"].dt.strftime("%Y-%m-%d"),
            "n_pre_commits": corpus["pre_snapshot_commits"].astype("int64"),
            "n_post_commits": corpus["post_snapshot_commits"].astype("int64"),
            "n_java_files": corpus["distinct_files_pre"].astype("int64"),
        }
    ).sort_values("project_id").reset_index(drop=True)
    corpus_summary.to_csv(TABLES_DIR / "corpus_summary.csv", index=False)

    eligible = df[df["eligible"]].copy()
    excluded = df[~df["eligible"]].copy()

    print(f"\n[Stage 2] Eligibility summary:")
    print(f"  Eligible projects : {len(eligible)} / {len(df)}")
    print(f"  Excluded projects : {len(excluded)}")
    if len(excluded):
        print("\n  Excluded details:")
        for _, r in excluded.iterrows():
            print(f"    - {r['project_id']:<35}  reason: {r['exclusion_reason']}")

    print(f"\n[Stage 2] Per-project profile (eligible, sorted by total_commits desc):")
    display_cols = [
        "project_id",
        "first_commit",
        "last_commit",
        "snapshot_date",
        "total_commits",
        "pre_snapshot_commits",
        "post_snapshot_commits",
        "distinct_files_pre",
        "distinct_authors_pre",
    ]
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 50, "display.width", 200):
        print(eligible[display_cols].sort_values("total_commits", ascending=False).to_string(index=False))

    print(f"\n[Stage 2] Saved: {out_parquet}")
    print(f"[Stage 2] Saved: {out_csv}")
    print(f"[Stage 2] Total elapsed : {time.time() - t0:.1f}s")
    print("[Stage 2] Complete.")


if __name__ == "__main__":
    main()
