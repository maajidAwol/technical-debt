"""
Stage 2 - Per-project profiling and snapshot-date selection.

Multi-snapshot mode (Option C): produces one row per ``(project_id,
snapshot_id)`` where ``snapshot_id`` is one of ``t25``, ``t50``, ``t75``
(25th / 50th / 75th percentile of master-branch commit dates).

Each snapshot is independently subject to the eligibility thresholds
``MIN_PRE_SNAPSHOT_COMMITS`` and ``MIN_POST_SNAPSHOT_COMMITS`` (strict
policy, Option C Q1=a).

Outputs
-------
- ``data/processed/project_snapshots.parquet``
- ``results/tables/project_stats.csv``
- ``results/tables/corpus_summary.csv``
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
from src.data.snapshot import (  # noqa: E402
    DEFAULT_SNAPSHOT_PERCENTILES,
    compute_all_multi_snapshots,
)


def main() -> None:
    t0 = time.time()
    print(f"[Stage 2] Strategy        : {SNAPSHOT_STRATEGY} (multi-snapshot {list(DEFAULT_SNAPSHOT_PERCENTILES)})")
    print(f"[Stage 2] Observation win : {OBSERVATION_WINDOW_MONTHS} months")

    with get_connection() as conn:
        print("[Stage 2] Computing multi-snapshots (t25, t50, t75) for every project ...")
        df = compute_all_multi_snapshots(conn)

    print(f"[Stage 2] Computed {len(df)} (project, snapshot) profiles in {time.time() - t0:.1f}s")

    out_parquet = PROCESSED_DATA_DIR / "project_snapshots.parquet"
    out_csv = TABLES_DIR / "project_stats.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    eligible = df[df["eligible"]].copy()
    excluded = df[~df["eligible"]].copy()

    # Corpus summary: one row per eligible (project, snapshot).
    corpus_summary = pd.DataFrame(
        {
            "project_id": eligible["project_id"],
            "snapshot_id": eligible["snapshot_id"],
            "snapshot_date": eligible["snapshot_date"].dt.strftime("%Y-%m-%d"),
            "n_pre_commits": eligible["pre_snapshot_commits"].astype("int64"),
            "n_post_commits": eligible["post_snapshot_commits"].astype("int64"),
            "n_java_files": eligible["distinct_files_pre"].astype("int64"),
        }
    ).sort_values(["project_id", "snapshot_id"]).reset_index(drop=True)
    corpus_summary.to_csv(TABLES_DIR / "corpus_summary.csv", index=False)

    n_projects_with_any_eligible = eligible["project_id"].nunique()
    print(f"\n[Stage 2] Eligibility summary:")
    print(f"  Eligible (project, snapshot) rows : {len(eligible)} / {len(df)}")
    print(f"  Excluded (project, snapshot) rows : {len(excluded)}")
    print(f"  Distinct projects with >=1 eligible snapshot : {n_projects_with_any_eligible}")

    per_snap = eligible.groupby("snapshot_id").size()
    print("\n  Eligible per snapshot_id:")
    for sid, n in per_snap.items():
        print(f"    {sid}: {n} projects")

    if len(excluded):
        print("\n  Excluded (project, snapshot) details:")
        for _, r in excluded.iterrows():
            print(f"    - {r['project_id']:<35} [{r['snapshot_id']}] reason: {r['exclusion_reason']}")

    print(f"\n[Stage 2] Per-(project, snapshot) profile (eligible only):")
    display_cols = [
        "project_id",
        "snapshot_id",
        "snapshot_date",
        "pre_snapshot_commits",
        "post_snapshot_commits",
        "distinct_files_pre",
        "distinct_authors_pre",
    ]
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 50, "display.width", 200):
        print(eligible[display_cols].sort_values(["project_id", "snapshot_id"]).to_string(index=False))

    print(f"\n[Stage 2] Saved: {out_parquet}")
    print(f"[Stage 2] Saved: {out_csv}")
    print(f"[Stage 2] Total elapsed : {time.time() - t0:.1f}s")
    print("[Stage 2] Complete.")


if __name__ == "__main__":
    main()
