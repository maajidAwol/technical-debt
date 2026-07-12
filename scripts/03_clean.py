"""
Stage 3 - Clean, normalize and persist the raw dataset tables.

Pipeline:
  1. Resolve basename collisions in SONAR_ISSUES.COMPONENT (full paths)
     vs GIT_COMMITS_CHANGES.FILE (basenames in most projects) and write
     a per-project ``clean_basenames`` table that downstream stages use
     to filter rows to the unambiguous / resolved subset.
  2. Clean each raw table (paths normalised, dates parsed, Java-only).
  3. Filter every per-file frame to rows whose
     ``(project_id, basename)`` has ``is_clean == 1``.
  4. Persist cleaned parquets and the collision report.

Outputs
-------
- ``data/processed/clean_basenames.parquet``
    (project_id, basename, is_clean, resolution_status, resolved_full_path)
- ``data/processed/collision_report.csv``
    (project_id, n_total, n_kept, n_dropped, drop_rate_pct)
- ``data/processed/clean_git_commits.parquet``
- ``data/processed/clean_git_commits_changes.parquet``
- ``data/processed/clean_sonar_issues.parquet``
- ``data/processed/clean_sonar_measures.parquet``
- ``data/processed/clean_szz.parquet``
- ``data/processed/clean_jira_issues.parquet``
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DATA_DIR  # noqa: E402
from src.data.clean import (  # noqa: E402
    clean_git_commits,
    clean_git_commits_changes,
    clean_jira_issues,
    clean_sonar_issues,
    clean_sonar_measures_with_dates,
    clean_szz_with_dates,
    extract_basename,
    is_java_source,
    normalize_component_path,
)
from src.data.load_data import get_connection  # noqa: E402
from src.data.snapshot import load_snapshots  # noqa: E402


def _fmt_rows(n: int) -> str:
    return f"{n:>12,}"


# ---------------------------------------------------------------------------
# Basename collision resolution
# ---------------------------------------------------------------------------
def _resolve_basename(paths: List[str]) -> Tuple[Optional[str], str]:
    """Pick a canonical full_path for an ambiguous basename, or drop it.

    Rules (in order):
      1. Unique ``src/main`` non-test path -> resolved_main
      2. Multiple ``src/main`` paths       -> shortest one (resolved_shortest)
      3. Unique non-test path              -> resolved_nontest
      4. Otherwise                         -> dropped
    """
    main = [p for p in paths if "src/main" in p and "test" not in p.lower()]
    if len(main) == 1:
        return main[0], "resolved_main"
    if len(main) > 1:
        return min(main, key=lambda p: p.count("/")), "resolved_shortest"
    non_test = [p for p in paths if "test" not in p.lower()]
    if len(non_test) == 1:
        return non_test[0], "resolved_nontest"
    return None, "dropped"


def build_clean_basenames(conn) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the (basename -> canonical path) map and a per-project report.

    Reads SONAR_ISSUES directly (only table that carries full paths),
    derives basename, counts paths per basename, and applies the
    resolution rules above. Java-source filtering is applied so we
    never adjudicate test files.

    Returns
    -------
    clean : DataFrame with columns
        (project_id, basename, is_clean, resolution_status, resolved_full_path)
    report : DataFrame with columns
        (project_id, n_total, n_kept, n_dropped, drop_rate_pct)
    """
    sonar_paths = pd.read_sql(
        """
        SELECT PROJECT_ID  AS project_id,
               COMPONENT   AS raw_component
          FROM SONAR_ISSUES
         GROUP BY PROJECT_ID, COMPONENT
        """,
        conn,
    )
    sonar_paths["full_path"] = sonar_paths["raw_component"].apply(normalize_component_path)
    sonar_paths["full_path"] = (
        sonar_paths["full_path"].astype(str).str.replace("\\", "/", regex=False)
    )
    sonar_paths = sonar_paths[sonar_paths["full_path"].apply(is_java_source)].copy()
    sonar_paths["basename"] = sonar_paths["full_path"].apply(extract_basename)
    sonar_paths = sonar_paths.dropna(subset=["basename"]).drop(columns=["raw_component"])

    grouped = (
        sonar_paths.groupby(["project_id", "basename"])["full_path"]
        .apply(lambda s: list(dict.fromkeys(s)))  # deduped, order-preserving
        .reset_index(name="paths")
    )
    grouped["n_paths"] = grouped["paths"].map(len)

    statuses: List[str] = []
    resolved_paths: List[Optional[str]] = []
    for paths in grouped["paths"]:
        if len(paths) == 1:
            statuses.append("unambiguous")
            resolved_paths.append(paths[0])
        else:
            chosen, status = _resolve_basename(paths)
            statuses.append(status)
            resolved_paths.append(chosen)
    grouped["resolution_status"] = statuses
    grouped["resolved_full_path"] = resolved_paths
    grouped["is_clean"] = (grouped["resolution_status"] != "dropped").astype("int64")

    clean = grouped[
        ["project_id", "basename", "is_clean", "resolution_status", "resolved_full_path"]
    ].copy()

    report_rows = []
    for pid, sub in clean.groupby("project_id"):
        n_total = int(len(sub))
        n_kept = int(sub["is_clean"].sum())
        n_dropped = n_total - n_kept
        report_rows.append(
            {
                "project_id": pid,
                "n_total": n_total,
                "n_kept": n_kept,
                "n_dropped": n_dropped,
                "drop_rate_pct": round(100 * n_dropped / n_total, 2) if n_total else 0.0,
            }
        )
    report = pd.DataFrame(report_rows).sort_values("project_id").reset_index(drop=True)
    return clean, report


def _filter_to_clean(
    df: pd.DataFrame,
    clean_basenames: pd.DataFrame,
    pid_col: str = "PROJECT_ID",
) -> pd.DataFrame:
    """Keep rows whose ``(project_id, basename)`` has ``is_clean == 1``.

    ``df`` is expected to carry both the uppercase project key (``PROJECT_ID``
    in cleaned tables) and a ``basename`` column.
    """
    if "basename" not in df.columns:
        raise KeyError(f"_filter_to_clean: 'basename' column missing in frame with cols {list(df.columns)[:8]}")
    keep = clean_basenames[clean_basenames["is_clean"] == 1][["project_id", "basename"]]
    keep = keep.rename(columns={"project_id": pid_col})
    return df.merge(keep, on=[pid_col, "basename"], how="inner").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stage driver
# ---------------------------------------------------------------------------
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

    with get_connection() as conn:
        # --- Step 1: basename collision resolution -------------------------
        print("[Stage 3] (1/7) Resolving basename collisions ...", end=" ", flush=True)
        t = time.time()
        clean_basenames, collision_report = build_clean_basenames(conn)
        # Restrict to eligible projects only (the SQL pulled all projects).
        clean_basenames = clean_basenames[clean_basenames["project_id"].isin(projects)].copy()
        collision_report = collision_report[
            collision_report["project_id"].isin(projects)
        ].reset_index(drop=True)
        print(
            f"{_fmt_rows(len(clean_basenames))} basenames "
            f"(kept={int(clean_basenames['is_clean'].sum()):,}, "
            f"dropped={int((1 - clean_basenames['is_clean']).sum()):,})  "
            f"({time.time() - t:.1f}s)"
        )

        print("[Stage 3] (2/7) GIT_COMMITS ...", end=" ", flush=True)
        t = time.time()
        gc = clean_git_commits(conn, projects)
        print(f"{_fmt_rows(len(gc))} rows  ({time.time()-t:.1f}s)")

        print("[Stage 3] (3/7) GIT_COMMITS_CHANGES (Java only) ...", end=" ", flush=True)
        t = time.time()
        gcc = clean_git_commits_changes(conn, projects, java_only=True)
        print(f"{_fmt_rows(len(gcc))} rows  ({time.time()-t:.1f}s)")

        print("[Stage 3] (4/7) SONAR_ISSUES (Java only) ...", end=" ", flush=True)
        t = time.time()
        si = clean_sonar_issues(conn, projects, java_only=True)
        print(f"{_fmt_rows(len(si))} rows  ({time.time()-t:.1f}s)")

        print("[Stage 3] (5/7) SONAR_MEASURES (with dates) ...", end=" ", flush=True)
        t = time.time()
        sm = clean_sonar_measures_with_dates(conn, projects)
        print(f"{_fmt_rows(len(sm))} rows  ({time.time()-t:.1f}s)")

        print("[Stage 3] (6/7) SZZ (with dates) ...", end=" ", flush=True)
        t = time.time()
        szz = clean_szz_with_dates(conn, projects, gc)
        print(f"{_fmt_rows(len(szz))} rows  ({time.time()-t:.1f}s)")

        print("[Stage 3] (7/7) JIRA_ISSUES ...", end=" ", flush=True)
        t = time.time()
        ji = clean_jira_issues(conn, projects)
        print(f"{_fmt_rows(len(ji))} rows  ({time.time()-t:.1f}s)")

    # --- Filter to is_clean == 1 for per-file frames -----------------------
    print("[Stage 3] Filtering per-file frames to resolved basenames ...")
    gcc_pre = len(gcc)
    si_pre = len(si)
    gcc = _filter_to_clean(gcc, clean_basenames)
    si = _filter_to_clean(si, clean_basenames)
    print(
        f"   git_commits_changes : {gcc_pre:>10,} -> {len(gcc):>10,} "
        f"({100*(gcc_pre-len(gcc))/max(gcc_pre,1):.1f}% removed)"
    )
    print(
        f"   sonar_issues        : {si_pre:>10,} -> {len(si):>10,} "
        f"({100*(si_pre-len(si))/max(si_pre,1):.1f}% removed)"
    )

    # --- Write outputs -----------------------------------------------------
    print("[Stage 3] Writing parquet outputs ...")
    clean_basenames.to_parquet(PROCESSED_DATA_DIR / "clean_basenames.parquet", index=False)
    collision_report.to_csv(PROCESSED_DATA_DIR / "collision_report.csv", index=False)
    gc.to_parquet(PROCESSED_DATA_DIR / "clean_git_commits.parquet", index=False)
    gcc.to_parquet(PROCESSED_DATA_DIR / "clean_git_commits_changes.parquet", index=False)
    si.to_parquet(PROCESSED_DATA_DIR / "clean_sonar_issues.parquet", index=False)
    sm.to_parquet(PROCESSED_DATA_DIR / "clean_sonar_measures.parquet", index=False)
    szz.to_parquet(PROCESSED_DATA_DIR / "clean_szz.parquet", index=False)
    ji.to_parquet(PROCESSED_DATA_DIR / "clean_jira_issues.parquet", index=False)

    print("\n[Stage 3] Collision report (data/processed/collision_report.csv):")
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(collision_report.to_string(index=False))

    median_drop = collision_report["drop_rate_pct"].median()
    max_drop = collision_report["drop_rate_pct"].max()
    print(f"\n[Stage 3] Basename drop rate: median={median_drop:.1f}%  max={max_drop:.1f}%")
    if not (0 <= median_drop <= 25):
        print(f"[Stage 3] WARNING: median drop rate {median_drop:.1f}% outside expected 5-15% band.")

    print(f"\n[Stage 3] Total elapsed : {time.time() - t0:.1f}s")
    print("[Stage 3] Complete.")


if __name__ == "__main__":
    main()
