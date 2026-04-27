"""
Stage 4 - Compute three label variants for every eligible project.

For each project, applies the ``compute_consequence_labels``,
``compute_severity_labels`` and ``compute_szz_labels`` functions using the
snapshot ``t`` computed in Stage 2 and the observation window ``W`` from
``config.OBSERVATION_WINDOW_MONTHS``.

Outputs
-------
- ``data/processed/labels_consequence.parquet`` - consequence labels and
  risk-score components.
- ``data/processed/labels_severity.parquet`` - severity labels and per-
  severity counts at snapshot.
- ``data/processed/labels_szz.parquet`` - SZZ labels and SZZ-event counts.
- ``results/tables/label_summary.csv`` - per-project positive-rate table
  (thesis Table X).
- ``results/tables/label_agreement.csv`` - pairwise kappa / Jaccard
  between the three variants (thesis Table Y).

Run
---
.. code-block:: bash

    .\\venv\\Scripts\\python.exe scripts/04_label.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    HIGH_RISK_PERCENTILE,
    OBSERVATION_WINDOW_MONTHS,
    PROCESSED_DATA_DIR,
    SEVERITY_BASELINE_LEVELS,
    TABLES_DIR,
)
from src.data.labeling import (  # noqa: E402
    compute_consequence_labels,
    compute_severity_labels,
    compute_szz_labels,
    label_agreement,
)
from src.data.snapshot import load_snapshots  # noqa: E402


def _load_clean() -> dict[str, pd.DataFrame]:
    commits = pd.read_parquet(PROCESSED_DATA_DIR / "clean_git_commits.parquet")
    changes = pd.read_parquet(PROCESSED_DATA_DIR / "clean_git_commits_changes.parquet")
    sonar_issues = pd.read_parquet(PROCESSED_DATA_DIR / "clean_sonar_issues.parquet")
    szz = pd.read_parquet(PROCESSED_DATA_DIR / "clean_szz.parquet")
    jira = pd.read_parquet(PROCESSED_DATA_DIR / "clean_jira_issues.parquet")
    # pandas can emit tz-naive when round-tripping to parquet on Windows; normalize
    for df, cols in (
        (commits, ["AUTHOR_DATE", "COMMITTER_DATE"]),
        (changes, ["DATE"]),
        (sonar_issues, ["CREATION_DATE", "CLOSE_DATE"]),
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
        "szz": szz,
        "jira": jira,
    }


def main() -> None:
    t0 = time.time()
    print(f"[Stage 4] Observation window  : {OBSERVATION_WINDOW_MONTHS} months")
    print(f"[Stage 4] High-risk percentile: top {HIGH_RISK_PERCENTILE}% (consequence)")
    print(f"[Stage 4] Severity baseline   : {SEVERITY_BASELINE_LEVELS}")

    snaps = load_snapshots(PROCESSED_DATA_DIR / "project_snapshots.parquet")
    eligible = snaps[snaps["eligible"]].copy()
    print(f"[Stage 4] Eligible projects   : {len(eligible)}")

    print("[Stage 4] Loading cleaned parquets ...")
    data = _load_clean()
    print(f"[Stage 4]   commits       : {len(data['commits']):>10,}")
    print(f"[Stage 4]   changes       : {len(data['changes']):>10,}")
    print(f"[Stage 4]   sonar_issues  : {len(data['sonar_issues']):>10,}")
    print(f"[Stage 4]   szz           : {len(data['szz']):>10,}")
    print(f"[Stage 4]   jira          : {len(data['jira']):>10,}")

    cons_parts, sev_parts, szz_parts = [], [], []
    summary_rows = []

    print("\n[Stage 4] Computing labels per project ...")
    for _, row in eligible.sort_values("project_id").iterrows():
        pid = row["project_id"]
        t = row["snapshot_date"]
        t_start = time.time()

        cons = compute_consequence_labels(
            pid,
            t,
            data["commits"],
            data["changes"],
            data["szz"],
            data["jira"],
        )
        sev = compute_severity_labels(pid, t, data["sonar_issues"], data["changes"])
        szzl = compute_szz_labels(pid, t, data["changes"], data["szz"])

        cons_parts.append(cons)
        sev_parts.append(sev)
        szz_parts.append(szzl)

        summary_rows.append(
            {
                "project_id": pid,
                "n_basenames": len(cons),
                "consequence_positives": int(cons["is_high_risk"].sum()),
                "consequence_rate_pct": round(100 * cons["is_high_risk"].mean(), 2),
                "severity_positives": int(sev["is_high_risk"].sum()),
                "severity_rate_pct": round(100 * sev["is_high_risk"].mean(), 2),
                "szz_positives": int(szzl["is_high_risk"].sum()),
                "szz_rate_pct": round(100 * szzl["is_high_risk"].mean(), 2),
            }
        )

        print(
            f"   {pid:<35}  N={len(cons):>6}  "
            f"cons={cons['is_high_risk'].sum():>5}  "
            f"sev={sev['is_high_risk'].sum():>5}  "
            f"szz={szzl['is_high_risk'].sum():>5}  "
            f"({time.time() - t_start:.1f}s)"
        )

    cons_all = pd.concat(cons_parts, ignore_index=True)
    sev_all = pd.concat(sev_parts, ignore_index=True)
    szz_all = pd.concat(szz_parts, ignore_index=True)

    cons_all.to_parquet(PROCESSED_DATA_DIR / "labels_consequence.parquet", index=False)
    sev_all.to_parquet(PROCESSED_DATA_DIR / "labels_severity.parquet", index=False)
    szz_all.to_parquet(PROCESSED_DATA_DIR / "labels_szz.parquet", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES_DIR / "label_summary.csv", index=False)

    agreement = label_agreement(
        {"consequence": cons_all, "severity": sev_all, "szz": szz_all}
    )
    agreement.to_csv(TABLES_DIR / "label_agreement.csv", index=False)

    print("\n[Stage 4] Per-project positive rates:")
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(summary.to_string(index=False))

    print("\n[Stage 4] Label agreement (pairwise):")
    print(agreement.to_string(index=False))

    print("\n[Stage 4] Overall positive rates:")
    print(
        f"   consequence : {cons_all['is_high_risk'].mean() * 100:5.2f}%  "
        f"({cons_all['is_high_risk'].sum():,} of {len(cons_all):,} basenames)"
    )
    print(
        f"   severity    : {sev_all['is_high_risk'].mean() * 100:5.2f}%  "
        f"({sev_all['is_high_risk'].sum():,} of {len(sev_all):,})"
    )
    print(
        f"   szz         : {szz_all['is_high_risk'].mean() * 100:5.2f}%  "
        f"({szz_all['is_high_risk'].sum():,} of {len(szz_all):,})"
    )

    print(f"\n[Stage 4] Total elapsed : {time.time() - t0:.1f}s")
    print("[Stage 4] Complete.")


if __name__ == "__main__":
    main()
