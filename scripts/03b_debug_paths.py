"""
Stage 3 debug - Compare file-path formats between SONAR_ISSUES and GIT_COMMITS_CHANGES.

The automated overlap check reported 0 percent matches, which means the two
tables use different repo-relative path conventions. This script prints the
first 30 distinct paths from each side for two projects (one small, one
medium) so we can design a correct normalization rule.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.load_data import get_connection  # noqa: E402


def main() -> None:
    for project in ("org.apache:codec", "org.apache:cayenne"):
        print(f"\n{'=' * 80}")
        print(f"PROJECT: {project}")
        print("=" * 80)

        with get_connection() as conn:
            q_sonar = (
                "SELECT DISTINCT COMPONENT FROM SONAR_ISSUES "
                "WHERE PROJECT_ID = ? LIMIT 15"
            )
            sonar = pd.read_sql_query(q_sonar, conn, params=[project])

            q_git = (
                "SELECT DISTINCT FILE FROM GIT_COMMITS_CHANGES "
                "WHERE PROJECT_ID = ? AND FILE LIKE '%.java' LIMIT 15"
            )
            git = pd.read_sql_query(q_git, conn, params=[project])

        print("\n-- SONAR_ISSUES.COMPONENT (raw) --")
        for c in sorted(sonar["COMPONENT"].dropna().unique())[:15]:
            print(f"  {c}")

        print("\n-- SONAR_ISSUES after stripping 'Key:' prefix --")
        for c in sorted(sonar["COMPONENT"].dropna().unique())[:15]:
            idx = c.find(":")
            stripped = c[idx + 1 :] if idx >= 0 else c
            print(f"  {stripped}")

        print("\n-- GIT_COMMITS_CHANGES.FILE (first 15 .java files) --")
        for f in sorted(git["FILE"].dropna().unique())[:15]:
            print(f"  {f}")


if __name__ == "__main__":
    main()
