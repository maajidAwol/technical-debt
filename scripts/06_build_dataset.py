"""
Stage 6 - Assemble the final modeling dataset.

Merges the four feature-family parquets with ``labels.parquet`` into a
single ``dataset_final.parquet`` with exactly 27 feature columns +
``is_high_risk`` (28 modelling columns) plus ``project_id`` and
``basename`` for downstream grouping.

Pre-processing:
    1. NaN -> 0 for every feature column
    2. log1p transform for heavy-tailed counts (LOG1P_FEATURES)

Scaling is NOT applied here: ``StandardScaler`` is fitted per CV fold
inside the training pipelines so it never leaks across folds.

Leakage assertions:
    - No raw severity counts (n_blocker / n_critical) in the matrix.
    - The weight-derivation surrogate (future_bugfix_count) is absent.

Outputs
-------
- ``data/processed/dataset_final.parquet``
- ``data/processed/feature_catalog.csv``
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    ALL_FEATURES,
    FEATURE_FAMILIES,
    LOG1P_FEATURES,
    PROCESSED_DATA_DIR,
    TABLES_DIR,
)


KEY_COLS = ["project_id", "basename"]
LABEL_COL = "is_high_risk"


FEATURE_CATALOG_ROWS = [
    # Family 1: Size / Complexity (project-level context replicated per file)
    ("ncloc", "size_complexity", "SONAR_MEASURES",
     "Nagappan ICSE 2006",
     "Larger files accumulate more debt opportunities"),
    ("complexity", "size_complexity", "SONAR_MEASURES",
     "McCabe 1976",
     "Cyclomatic complexity increases change effort and error-proneness"),
    ("cognitive_complexity", "size_complexity", "SONAR_MEASURES",
     "Campbell 2018 (SonarSource)",
     "Human-perceived complexity; hard to read = hard to maintain"),
    ("functions", "size_complexity", "SONAR_MEASURES",
     "Nagappan ICSE 2006",
     "Method count - more methods = more potential debt entry points"),
    ("classes", "size_complexity", "SONAR_MEASURES",
     "Nagappan ICSE 2006",
     "Class count - captures OO design scale"),

    # Family 2: Static debt
    ("n_code_smells", "static_debt", "SONAR_ISSUES",
     "Tsoukalas JSS 2020",
     "Maintainability violations (CODE_SMELL type)"),
    ("n_bugs", "static_debt", "SONAR_ISSUES",
     "Tsoukalas JSS 2020",
     "BUG-type issues regardless of severity"),
    ("total_debt_minutes", "static_debt", "SONAR_ISSUES",
     "Tsoukalas JSS 2020",
     "Remediation effort estimate; debt principal in minutes"),
    ("issue_density", "static_debt", "SONAR_ISSUES / SONAR_MEASURES",
     "Tsoukalas JSS 2020",
     "total_issues / ncloc; size-normalized debt intensity"),
    ("duplicated_lines_density", "static_debt", "SONAR_MEASURES",
     "Fowler 1999",
     "Duplication raises cost of propagating fixes"),

    # Family 3: Historical change
    ("total_commits_pre", "historical", "GIT_COMMITS",
     "Kamei TSE 2013",
     "Total commits touching the file before t"),
    ("code_churn_pre", "historical", "GIT_COMMITS_CHANGES",
     "Kamei TSE 2013",
     "Lifetime lines added + removed; total volatility"),
    ("recent_churn_90d", "historical", "GIT_COMMITS_CHANGES",
     "Hassan ICSE 2009",
     "Churn in last 90 days; current hotspot signal"),
    ("commit_frequency_30d", "historical", "GIT_COMMITS",
     "Hassan ICSE 2009",
     "Commits in last 30 days; recent activity level"),
    ("file_age_days", "historical", "GIT_COMMITS",
     "Kamei TSE 2013",
     "Days from first commit to t; older files carry more debt"),
    ("days_since_last_change", "historical", "GIT_COMMITS",
     "Kamei TSE 2013",
     "Days from last commit to t; stale files may need attention"),
    ("contributor_count", "historical", "GIT_COMMITS",
     "Bird FSE 2011",
     "Distinct authors before t; coordination overhead risk"),
    ("ownership_ratio", "historical", "GIT_COMMITS",
     "Bird FSE 2011",
     "max_single_author_commits / total_commits_pre; diffuse responsibility risk"),

    # Family 4: Co-change graph
    ("cocg_degree", "graph", "GIT_COMMITS_CHANGES (co-change graph)",
     "Jiang EMSE 2024",
     "Number of co-change neighbours; architectural coupling"),
    ("cocg_pagerank", "graph", "GIT_COMMITS_CHANGES (co-change graph)",
     "Jiang EMSE 2024",
     "Recursive importance via weighted PageRank"),
    ("cocg_betweenness", "graph", "GIT_COMMITS_CHANGES (co-change graph)",
     "Jiang EMSE 2024",
     "Bridge status; changes here ripple to many modules"),
    ("cocg_entropy", "graph", "GIT_COMMITS_CHANGES (co-change graph)",
     "Ethari & Bhardwaj 2025",
     "Shannon entropy over neighbour edge weights; high = scattered coupling"),

    # Family 5: Prior defect
    ("bugfix_commits_pre", "prior_defect", "GIT_COMMITS (regex)",
     "Hassan ICSE 2009",
     "Lifetime bug-fix commit count using the canonical regex"),
    ("bugfix_commits_90d", "prior_defect", "GIT_COMMITS (regex)",
     "Hassan ICSE 2009",
     "Bug-fix commits last 90 days; recent defect activity"),
    ("bug_density_pre", "prior_defect", "GIT_COMMITS (regex)",
     "Hassan ICSE 2009",
     "bugfix_commits_pre / total_commits_pre; normalized"),
    ("n_jira_bugs_pre", "prior_defect", "JIRA_ISSUES",
     "Falessi ESEM 2020",
     "Officially confirmed bug tickets linked to the file before t"),
    ("jira_blocker_flag", "prior_defect", "JIRA_ISSUES",
     "Falessi ESEM 2020",
     "Flag for any linked JIRA Bug with PRIORITY Blocker or Critical"),
]


def _load_features() -> pd.DataFrame:
    static = pd.read_parquet(PROCESSED_DATA_DIR / "features_static.parquet")
    hist = pd.read_parquet(PROCESSED_DATA_DIR / "features_historical.parquet")
    graph = pd.read_parquet(PROCESSED_DATA_DIR / "features_graph.parquet")
    prior = pd.read_parquet(PROCESSED_DATA_DIR / "features_priordefect.parquet")

    # Drop helper columns that may still be present.
    for df in (static, hist, graph, prior):
        for col in ("snapshot_date",):
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    out = static.merge(hist, on=KEY_COLS, how="outer")
    out = out.merge(graph, on=KEY_COLS, how="outer")
    out = out.merge(prior, on=KEY_COLS, how="outer")
    return out


def main() -> None:
    t0 = time.time()
    print("[Stage 6] Loading feature parquets ...")
    features = _load_features()
    labels = pd.read_parquet(PROCESSED_DATA_DIR / "labels.parquet")[
        KEY_COLS + [LABEL_COL]
    ]
    print(f"   features rows={len(features):,}  cols={len(features.columns)}")
    print(f"   labels   rows={len(labels):,}  positive_rate={100*labels[LABEL_COL].mean():.2f}%")

    df = features.merge(labels, on=KEY_COLS, how="inner")
    print(f"[Stage 6] After merge: rows={len(df):,}")

    # ----- Restrict to the 27 declared features -----
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Stage 6: missing expected feature columns: {missing}")
    df = df[KEY_COLS + ALL_FEATURES + [LABEL_COL]].copy()

    # ----- NaN -> 0 across the 27 features -----
    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(0)

    # ----- log1p heavy-tailed columns -----
    print(f"[Stage 6] Applying log1p to {len(LOG1P_FEATURES)} columns ...")
    for col in LOG1P_FEATURES:
        df[col] = np.log1p(df[col].astype(float))

    # ----- Leakage assertions -----
    assert "n_blocker" not in df.columns and "n_critical" not in df.columns, (
        "Severity leakage: raw severity counts found in feature matrix"
    )
    assert "future_bugfix_count" not in df.columns, (
        "Surrogate leakage: future data found in feature matrix"
    )
    feat_cols = [c for c in df.columns if c not in KEY_COLS + [LABEL_COL]]
    assert len(feat_cols) == 27, f"expected 27 feature cols, got {len(feat_cols)}: {feat_cols}"
    assert df.isna().sum().sum() == 0, "NaN detected in dataset_final after preprocessing"

    # ----- Persist -----
    out_path = PROCESSED_DATA_DIR / "dataset_final.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[Stage 6] Wrote {out_path}  rows={len(df):,}  cols={len(df.columns)}")

    # ----- Feature catalog -----
    catalog = pd.DataFrame(
        FEATURE_CATALOG_ROWS,
        columns=["feature_name", "family", "source_table", "literature_citation", "rationale"],
    )
    assert len(catalog) == 27, f"feature_catalog has {len(catalog)} rows, expected 27"
    # Sanity: every catalog row corresponds to a column in the dataset
    miss = set(catalog["feature_name"]) - set(df.columns)
    assert not miss, f"catalog features not in dataset: {miss}"
    catalog.to_csv(PROCESSED_DATA_DIR / "feature_catalog.csv", index=False)
    print(f"[Stage 6] Wrote feature_catalog.csv ({len(catalog)} features)")

    # ----- Per-family summary log -----
    print("\n[Stage 6] Feature family breakdown:")
    for fam, cols in FEATURE_FAMILIES.items():
        nz = (df[cols] != 0).any(axis=1).mean() * 100
        print(f"   {fam:<16}  {len(cols)} features  non_zero_any={nz:.1f}%")

    # ----- Prior-defect coverage diagnostic -----
    # The prior-defect family depends on JIRA and the bug-fix regex
    # matching commits. Projects with sparse JIRA linkage or terse commit
    # messages may have most rows all-zero across this family - good to
    # know before reading family ablation results.
    prior_cols = FEATURE_FAMILIES["prior_defect"]
    pf_rows = []
    for pid, sub in df.groupby("project_id"):
        all_zero = (sub[prior_cols] == 0).all(axis=1)
        pf_rows.append(
            {
                "project_id": pid,
                "n_files": int(len(sub)),
                "n_all_zero_prior_defect": int(all_zero.sum()),
                "pct_all_zero": round(100 * float(all_zero.mean()), 2),
            }
        )
    pf_coverage = pd.DataFrame(pf_rows).sort_values("project_id").reset_index(drop=True)
    pf_coverage.to_csv(PROCESSED_DATA_DIR / "prior_defect_coverage.csv", index=False)

    high_zero = pf_coverage[pf_coverage["pct_all_zero"] > 80]
    if len(high_zero) > 0:
        print(f"\n[Stage 6] WARNING: {len(high_zero)} project(s) have > 80% rows with all-zero prior-defect features:")
        print(high_zero.to_string(index=False))

    pos_rate = 100 * df[LABEL_COL].mean()
    print(f"\n[Stage 6] Final positive rate : {pos_rate:.2f}%  ({int(df[LABEL_COL].sum()):,} of {len(df):,})")

    # Also emit a tiny tables/ ack so the verifier can find the summary.
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[Stage 6] Elapsed: {time.time() - t0:.1f}s")
    print("[Stage 6] Complete.")


if __name__ == "__main__":
    main()
