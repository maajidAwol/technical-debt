"""
Stage 5 - Snapshot-aware feature extraction.

For each eligible project (``snapshot`` from Stage 2), builds:

1. Per-(project, basename) *static* features (SonarQube issue aggregates
   at snapshot + git-derived size + project-level context from the
   latest analysis <= ``t``).
2. Per-(project, basename) *historical* (process) features restricted
   to commits with ``AUTHOR_DATE <= t``.
3. Per-(project, basename) *co-change graph* features (centrality and
   coupling metrics on the snapshot-time co-change graph).
4. Per-(project, basename) *prior-defect* features (pre-snapshot
   bug-fix counts, SZZ-inducing flags, linked Jira tickets).

Outputs
-------
- ``data/processed/features_static.parquet``
- ``data/processed/features_historical.parquet``
- ``data/processed/features_graph.parquet``
- ``data/processed/features_priordefect.parquet``
- ``results/tables/feature_summary.csv`` - per-project row counts and
  quick statistics.

Parallelism
-----------
Projects are independent: each project's features are computed from a
self-contained slice of the cleaned dataframes. We process them in
parallel with ``joblib.Parallel`` using ``STAGE5_N_JOBS`` workers
(default ``cpu_count // 2``; override with the ``TD_N_JOBS`` env
var). Workers see only their project's pre-filtered slice, which
keeps per-worker RAM small and pickling fast.

Run
---
.. code-block:: bash

    .\\venv\\Scripts\\python.exe scripts/05_features.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import Parallel, delayed

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DATA_DIR, STAGE5_N_JOBS, TABLES_DIR  # noqa: E402
from src.data.snapshot import load_snapshots  # noqa: E402
from src.features.graph_features import build_graph_features_for_project  # noqa: E402
from src.features.historical_features import build_historical_features_for_project  # noqa: E402
from src.features.priordefect_features import build_priordefect_features_for_project  # noqa: E402
from src.features.static_features import build_static_features_for_project  # noqa: E402


def _load_clean() -> dict[str, pd.DataFrame]:
    commits = pd.read_parquet(PROCESSED_DATA_DIR / "clean_git_commits.parquet")
    changes = pd.read_parquet(PROCESSED_DATA_DIR / "clean_git_commits_changes.parquet")
    sonar_issues = pd.read_parquet(PROCESSED_DATA_DIR / "clean_sonar_issues.parquet")
    sonar_measures = pd.read_parquet(PROCESSED_DATA_DIR / "clean_sonar_measures.parquet")
    szz = pd.read_parquet(PROCESSED_DATA_DIR / "clean_szz.parquet")
    jira = pd.read_parquet(PROCESSED_DATA_DIR / "clean_jira_issues.parquet")

    for df, cols in (
        (commits, ["AUTHOR_DATE", "COMMITTER_DATE"]),
        (changes, ["DATE"]),
        (sonar_issues, ["CREATION_DATE", "CLOSE_DATE"]),
        (sonar_measures, ["analysis_date"]),
        (szz, ["fix_date", "induce_date"]),
        (jira, ["CREATION_DATE", "RESOLUTION_DATE", "UPDATE_DATE", "COMMIT_DATE"]),
    ):
        for col in cols:
            if col in df.columns and df[col].dtype.kind == "M" and df[col].dt.tz is None:
                df[col] = df[col].dt.tz_localize("UTC")
    return {
        "commits": commits,
        "changes": changes,
        "sonar_issues": sonar_issues,
        "sonar_measures": sonar_measures,
        "szz": szz,
        "jira": jira,
    }


def _slice_for_project(data: dict[str, pd.DataFrame], pid: str) -> dict[str, pd.DataFrame]:
    """Return per-project slices of every cleaned frame.

    The feature builders all filter on ``PROJECT_ID`` internally, so
    pre-slicing here is purely a memory and pickling optimisation:
    each worker receives only its project's data instead of all 22.
    """
    out: dict[str, pd.DataFrame] = {}
    for name, df in data.items():
        if "PROJECT_ID" in df.columns:
            out[name] = df[df["PROJECT_ID"] == pid].copy()
        else:
            out[name] = df
    return out


def _one_project(pid: str, t: pd.Timestamp, sliced: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Compute all four feature families for a single project.

    Pure function: no shared mutable state, no I/O. Safe to invoke in
    parallel worker processes via joblib.
    """
    ts = time.time()
    static_df = build_static_features_for_project(
        pid, t, sliced["sonar_issues"], sliced["sonar_measures"], sliced["changes"]
    )
    hist_df = build_historical_features_for_project(
        pid, t, sliced["commits"], sliced["changes"]
    )
    graph_df = build_graph_features_for_project(
        pid, t, sliced["changes"], verbose=False
    )
    prior_df = build_priordefect_features_for_project(
        pid, t, sliced["commits"], sliced["changes"], sliced["szz"], sliced["jira"]
    )
    elapsed = time.time() - ts

    summary_row = {
        "project_id": pid,
        "n_basenames": len(static_df),
        "static_cols": len(static_df.columns),
        "hist_cols": len(hist_df.columns),
        "graph_cols": len(graph_df.columns),
        "prior_cols": len(prior_df.columns),
        "mean_n_code_smells": round(float(static_df["n_code_smells"].mean()), 2)
        if "n_code_smells" in static_df.columns and len(static_df)
        else 0.0,
        "mean_total_commits_pre": round(float(hist_df["total_commits_pre"].mean()), 2)
        if "total_commits_pre" in hist_df.columns and len(hist_df)
        else 0.0,
        "mean_cocg_degree": round(float(graph_df["cocg_degree"].mean()), 2)
        if "cocg_degree" in graph_df.columns and len(graph_df)
        else 0.0,
        "mean_bugfix_pre": round(float(prior_df["bugfix_commits_pre"].mean()), 2)
        if "bugfix_commits_pre" in prior_df.columns and len(prior_df)
        else 0.0,
        "elapsed_s": round(elapsed, 2),
    }

    print(
        f"   done {pid:<35}  N={len(static_df):>6}  "
        f"static={len(static_df.columns):>3}  "
        f"hist={len(hist_df.columns):>3}  "
        f"graph={len(graph_df.columns):>3}  "
        f"prior={len(prior_df.columns):>3}  "
        f"({elapsed:.1f}s)",
        flush=True,
    )

    return {
        "pid": pid,
        "static": static_df,
        "hist": hist_df,
        "graph": graph_df,
        "prior": prior_df,
        "summary": summary_row,
    }


def main() -> None:
    t0 = time.time()
    snaps = load_snapshots(PROCESSED_DATA_DIR / "project_snapshots.parquet")
    eligible = snaps[snaps["eligible"]].copy()
    print(f"[Stage 5] Eligible projects: {len(eligible)}", flush=True)
    print(f"[Stage 5] Parallel workers (STAGE5_N_JOBS): {STAGE5_N_JOBS}", flush=True)

    print("[Stage 5] Loading cleaned parquets ...", flush=True)
    data = _load_clean()
    print(f"   commits        : {len(data['commits']):>10,}", flush=True)
    print(f"   changes        : {len(data['changes']):>10,}", flush=True)
    print(f"   sonar_issues   : {len(data['sonar_issues']):>10,}", flush=True)
    print(f"   sonar_measures : {len(data['sonar_measures']):>10,}", flush=True)
    print(f"   szz            : {len(data['szz']):>10,}", flush=True)
    print(f"   jira           : {len(data['jira']):>10,}", flush=True)

    eligible_sorted = eligible.sort_values("project_id").reset_index(drop=True)
    n_total = len(eligible_sorted)

    print("[Stage 5] Pre-slicing data per project ...", flush=True)
    tasks: list[tuple[str, pd.Timestamp, dict[str, pd.DataFrame]]] = []
    for _, row in eligible_sorted.iterrows():
        pid = row["project_id"]
        t = row["snapshot_date"]
        sliced = _slice_for_project(data, pid)
        n_pid_changes = len(sliced["changes"])
        tasks.append((pid, t, sliced))
        print(
            f"   queued {pid:<35} (snapshot={t:%Y-%m-%d}, "
            f"pre-snap changes={n_pid_changes:,})",
            flush=True,
        )
    # Drop the global frames once all per-project slices are built;
    # the workers no longer need them and this halves peak RAM.
    del data

    print(
        f"[Stage 5] Computing features in parallel "
        f"(n_jobs={STAGE5_N_JOBS}, n_projects={n_total}) ...",
        flush=True,
    )
    results = Parallel(n_jobs=STAGE5_N_JOBS, verbose=5)(
        delayed(_one_project)(pid, t, sliced) for (pid, t, sliced) in tasks
    )

    # Sort by project_id to make the concatenated outputs deterministic
    # regardless of worker completion order.
    results.sort(key=lambda r: r["pid"])

    static_all = pd.concat([r["static"] for r in results], ignore_index=True)
    hist_all = pd.concat([r["hist"] for r in results], ignore_index=True)
    graph_all = pd.concat([r["graph"] for r in results], ignore_index=True)
    prior_all = pd.concat([r["prior"] for r in results], ignore_index=True)

    static_all.to_parquet(PROCESSED_DATA_DIR / "features_static.parquet", index=False)
    hist_all.to_parquet(PROCESSED_DATA_DIR / "features_historical.parquet", index=False)
    graph_all.to_parquet(PROCESSED_DATA_DIR / "features_graph.parquet", index=False)
    prior_all.to_parquet(PROCESSED_DATA_DIR / "features_priordefect.parquet", index=False)

    summary_df = pd.DataFrame([r["summary"] for r in results])
    summary_df.to_csv(TABLES_DIR / "feature_summary.csv", index=False)

    print("\n[Stage 5] Feature summary:")
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(summary_df.to_string(index=False))

    print(
        f"\n[Stage 5] Total rows - static: {len(static_all):,} | "
        f"historical: {len(hist_all):,} | graph: {len(graph_all):,} | "
        f"prior_defect: {len(prior_all):,}"
    )
    print(f"[Stage 5] Static columns      : {list(static_all.columns)[:12]} ... total {len(static_all.columns)}")
    print(f"[Stage 5] Historical columns  : {list(hist_all.columns)[:12]} ... total {len(hist_all.columns)}")
    print(f"[Stage 5] Graph columns       : {list(graph_all.columns)[:12]} ... total {len(graph_all.columns)}")
    print(f"[Stage 5] Prior-defect columns: {list(prior_all.columns)[:12]} ... total {len(prior_all.columns)}")
    print(f"\n[Stage 5] Elapsed: {time.time() - t0:.1f}s")
    print("[Stage 5] Complete.")


if __name__ == "__main__":
    main()
