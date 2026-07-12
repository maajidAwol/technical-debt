"""
Stage 3 debug - Quantify basename-collision risk in SONAR_ISSUES.

Since GIT_COMMITS_CHANGES uses basenames for many projects, the only
universal join key between Sonar and Git is the basename. This script
measures how many distinct full SonarQube paths share the same basename,
by project - the higher the collision rate, the more file-level features
must be aggregated over all files with the same basename.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.load_data import get_connection  # noqa: E402


def main() -> None:
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT DISTINCT PROJECT_ID, COMPONENT FROM SONAR_ISSUES "
            "WHERE COMPONENT LIKE '%.java'",
            conn,
        )
    df["full_path"] = df["COMPONENT"].str.split(":", n=1).str[-1]
    df["basename"] = df["full_path"].str.rsplit("/", n=1).str[-1]

    rows = []
    for pid, sub in df.groupby("PROJECT_ID"):
        n_full = sub["full_path"].nunique()
        n_base = sub["basename"].nunique()
        colliding = sub.groupby("basename").size()
        n_collisions = int((colliding > 1).sum())
        max_group = int(colliding.max())
        rows.append(
            {
                "project_id": pid,
                "distinct_full_paths": n_full,
                "distinct_basenames": n_base,
                "colliding_basenames": n_collisions,
                "max_files_per_basename": max_group,
                "collision_rate_pct": round((1 - n_base / n_full) * 100, 1) if n_full else 0.0,
            }
        )
    report = pd.DataFrame(rows).sort_values("collision_rate_pct", ascending=False)
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(report.to_string(index=False))

    avg = report["collision_rate_pct"].mean()
    print(f"\nAverage collision rate: {avg:.2f}%")


if __name__ == "__main__":
    main()
