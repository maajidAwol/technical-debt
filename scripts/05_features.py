"""
Stage 5 - Snapshot-aware feature extraction.

For each eligible project (``snapshot`` from Stage 2), builds:

1. Per-(project, basename) *static* features (SonarQube issue aggregates
   at snapshot + git-derived size + project-level context from the
   latest analysis <= ``t``).
2. Per-(project, basename) *historical* (process) features restricted
   to commits with ``AUTHOR_DATE <= t``.

Outputs
-------
- ``data/processed/features_static.parquet``
- ``data/processed/features_historical.parquet``
- ``results/tables/feature_summary.csv`` - per-project row counts and
  quick statistics.

Run
---
.. code-block:: bash

    .\\venv\\Scripts\\python.exe scripts/05_features.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DATA_DIR, TABLES_DIR  # noqa: E402
from src.data.snapshot import load_snapshots  # noqa: E402
from src.features.historical_features import build_historical_features_for_project  # noqa: E402
from src.features.static_features import build_static_features_for_project  # noqa: E402


def _load_clean() -> dict[str, pd.DataFrame]:
    commits = pd.read_parquet(PROCESSED_DATA_DIR / "clean_git_commits.parquet")
    changes = pd.read_parquet(PROCESSED_DATA_DIR / "clean_git_commits_changes.parquet")
    sonar_issues = pd.read_parquet(PROCESSED_DATA_DIR / "clean_sonar_issues.parquet")
    sonar_measures = pd.read_parquet(PROCESSED_DATA_DIR / "clean_sonar_measures.parquet")

    for df, cols in (
        (commits, ["AUTHOR_DATE", "COMMITTER_DATE"]),
        (changes, ["DATE"]),
        (sonar_issues, ["CREATION_DATE", "CLOSE_DATE"]),
        (sonar_measures, ["analysis_date"]),
    ):
        for col in cols:
            if col in df.columns and df[col].dtype.kind == "M" and df[col].dt.tz is None:
                df[col] = df[col].dt.tz_localize("UTC")
    return {
        "commits": commits,
        "changes": changes,
        "sonar_issues": sonar_issues,
        "sonar_measures": sonar_measures,
    }


def main() -> None:
    t0 = time.time()
    snaps = load_snapshots(PROCESSED_DATA_DIR / "project_snapshots.parquet")
    eligible = snaps[snaps["eligible"]].copy()
    print(f"[Stage 5] Eligible projects: {len(eligible)}")

    print("[Stage 5] Loading cleaned parquets ...")
    data = _load_clean()
    print(f"   commits        : {len(data['commits']):>10,}")
    print(f"   changes        : {len(data['changes']):>10,}")
    print(f"   sonar_issues   : {len(data['sonar_issues']):>10,}")
    print(f"   sonar_measures : {len(data['sonar_measures']):>10,}")

    static_parts, hist_parts = [], []
    summary = []

    for _, row in eligible.sort_values("project_id").iterrows():
        pid = row["project_id"]
        t = row["snapshot_date"]
        ts = time.time()

        static_df = build_static_features_for_project(
            pid, t, data["sonar_issues"], data["sonar_measures"], data["changes"]
        )
        hist_df = build_historical_features_for_project(
            pid, t, data["commits"], data["changes"]
        )

        static_parts.append(static_df)
        hist_parts.append(hist_df)
        summary.append(
            {
                "project_id": pid,
                "n_basenames": len(static_df),
                "static_cols": len(static_df.columns),
                "hist_cols": len(hist_df.columns),
                "mean_n_issues_open": round(float(static_df["n_issues_open"].mean()), 2)
                if "n_issues_open" in static_df.columns
                else 0.0,
                "mean_total_commits_pre": round(float(hist_df["total_commits_pre"].mean()), 2)
                if "total_commits_pre" in hist_df.columns
                else 0.0,
                "elapsed_s": round(time.time() - ts, 2),
            }
        )
        print(
            f"   {pid:<35}  N={len(static_df):>6}  "
            f"static_cols={len(static_df.columns):>3}  "
            f"hist_cols={len(hist_df.columns):>3}  "
            f"({time.time() - ts:.1f}s)"
        )

    static_all = pd.concat(static_parts, ignore_index=True)
    hist_all = pd.concat(hist_parts, ignore_index=True)

    static_all.to_parquet(PROCESSED_DATA_DIR / "features_static.parquet", index=False)
    hist_all.to_parquet(PROCESSED_DATA_DIR / "features_historical.parquet", index=False)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(TABLES_DIR / "feature_summary.csv", index=False)

    print("\n[Stage 5] Feature summary:")
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(summary_df.to_string(index=False))

    print(
        f"\n[Stage 5] Total rows - static: {len(static_all):,} | "
        f"historical: {len(hist_all):,}"
    )
    print(f"[Stage 5] Static columns     : {list(static_all.columns)[:12]} ... total {len(static_all.columns)}")
    print(f"[Stage 5] Historical columns : {list(hist_all.columns)[:12]} ... total {len(hist_all.columns)}")
    print(f"\n[Stage 5] Elapsed: {time.time() - t0:.1f}s")
    print("[Stage 5] Complete.")


if __name__ == "__main__":
    main()
