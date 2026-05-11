# Previous Version Report (Pre-Enhancement Baseline)

Snapshot of the pipeline as it stood **before** the Refined-Extended enhancement
package. All numbers below come from the artefacts already on disk
(`results/tables/*.csv`, `data/processed/*.parquet`).

---

## 1. Dataset

Source: **Technical Debt Dataset v2.0** (Lenarduzzi et al. 2019), one SQLite
file `data/raw/td_V2.db` covering 31 Apache Java projects.

### Tables and row counts (raw)

| Table                          | Rows         | What it is |
|---|---:|---|
| `GIT_COMMITS_CHANGES`          | 1,142,878    | One row per (commit × file) - the change log. Lines added/removed. |
| `SONAR_ISSUES`                 | 1,024,614    | One row per SonarQube rule violation found across all analyses. |
| `REFACTORING_MINER`            |   362,253    | RefactoringMiner output - one row per detected refactoring (unused in current pipeline). |
| `GIT_COMMITS`                  |   153,994    | One row per commit. Author, dates, branches, merge flag. |
| `SONAR_ANALYSIS`               |    67,550    | One row per SonarQube analysis run. Project + analysis date. |
| `SONAR_MEASURES`               |    66,711    | Project-level metric snapshot per analysis (NCLOC, complexity, ratings, etc.). |
| `JIRA_ISSUES`                  |    61,402    | One row per Jira ticket. Type, priority, dates, linking commit hash. |
| `SZZ_FAULT_INDUCING_COMMITS`   |    52,428    | Output of SZZ algorithm: pairs of (fault-fixing commit, fault-inducing commit). |
| `SONAR_RULES`                  |     1,819    | Rule catalogue: one row per Sonar rule definition. |
| `PROJECTS`                     |        31    | Project metadata: ID, git link, jira link. |

### Key columns per table (used by the pipeline)

- **`GIT_COMMITS`** : `PROJECT_ID`, `COMMIT_HASH`, `COMMIT_MESSAGE`, `AUTHOR`, `AUTHOR_DATE`, `IN_MAIN_BRANCH`, `MERGE`. Used for time ordering, authorship, bug-fix detection, JIRA-link extraction.
- **`GIT_COMMITS_CHANGES`** : `PROJECT_ID`, `COMMIT_HASH`, `FILE`, `DATE`, `COMMITTER_ID`, `LINES_ADDED`, `LINES_REMOVED`. Per-file churn over time. The `FILE` column is normalised to a **basename** (file name without directory) before everything else - this is the granularity of all features and labels.
- **`SONAR_MEASURES`** : project-level metrics at each analysis date (NCLOC, COMPLEXITY, COGNITIVE_COMPLEXITY, COVERAGE, DUPLICATED_LINES_DENSITY, COMMENT_LINES_DENSITY, SQALE_INDEX, SQALE_DEBT_RATIO, FILE_COMPLEXITY, FILES, CLASSES, FUNCTIONS, STATEMENTS, ...). 12 metrics promoted to per-(project, basename) features in Stage 5.
- **`SONAR_ISSUES`** : `PROJECT_ID`, `ISSUE_KEY`, `TYPE` (BUG/CODE_SMELL/VULNERABILITY), `RULE`, `SEVERITY` (BLOCKER/CRITICAL/MAJOR/MINOR/INFO), `STATUS` (OPEN/CLOSED), `CREATION_DATE`, `CLOSE_DATE`, `COMPONENT` (file path). Source of severity counts and label-leakage controls.
- **`SZZ_FAULT_INDUCING_COMMITS`** : `PROJECT_ID`, `FAULT_FIXING_COMMIT_HASH`, `FAULT_INDUCING_COMMIT_HASH`. Used to derive past defect signals and the SZZ baseline label.
- **`JIRA_ISSUES`** : `PROJECT_ID`, `KEY`, `PRIORITY`, `TYPE`, `STATUS`, `RESOLUTION`, dates, `COMMIT_DATE`/`HASH` linking it to a fixing commit. Bug-vs-non-bug and number of linked tickets.

A full `db_schema.csv` (350 rows, all 10 tables × every column with example values) is in `results/tables/db_schema.csv`.

---

## 2. Features - original vs engineered, dataset size

### What the *raw tables* already contain

The raw DB does **not** ship per-file features - it ships **events** (commits, changes, issues, analyses) and **project-level metrics**. So everything below is engineered; nothing is taken straight from the DB as a feature.

### Engineered feature catalogue (pre-enhancement)

Granularity throughout: **(project, basename)** = one row per source file (basename), per project. We do **not** preserve directory structure.

| Family | Module | Count | Examples |
|---|---|---:|---|
| **Static (sonar measures lifted to file granularity)** | `src/features/static_features.py` | 12 | `ncloc`, `complexity`, `cognitive_complexity`, `coverage`, `duplicated_lines_density`, `comment_lines_density`, `sqale_index`, `sqale_debt_ratio`, `file_complexity`, `classes`, `functions`, `statements` |
| **Static derived** | same | 2 | `cyclomatic_density` (= complexity / ncloc), `comment_to_code_ratio` |
| **Issue-count features (per file, at snapshot)** | same | ~30 | `n_issues_open`, `n_blocker`, `n_critical`, `n_major`, `n_minor`, `n_info`, `n_bugs`, `n_vulnerabilities`, `n_code_smells`, plus per-rule aggregates and severity-weighted counts |
| **Historical (commit-level process metrics, all `<= t`)** | `src/features/historical_features.py` | 18 | `total_commits_pre`, `total_contributors_pre`, `code_added_pre`, `code_removed_pre`, `code_churn_pre`, `recent_churn_30d_pre`, `recent_churn_90d_pre`, `recent_commits_30d_pre`, `recent_commits_90d_pre`, `file_age_days_at_snapshot`, `days_since_last_change_at_snapshot`, `ownership_ratio_pre`, `avg_change_size_pre`, `max_single_commit_churn_pre`, `std_change_size_pre` |

### Dataset size after cleaning + feature engineering (pre-enhancement)

| Cleaned parquet | Rows |
|---|---:|
| `clean_git_commits.parquet`            | 103,905 (from 153,994 raw - dropped non-master / merge / non-Java-only) |
| `clean_git_commits_changes.parquet`    | 526,035 (from 1,142,878 raw - kept only `.java`, dropped tests / generated / target / build) |
| `clean_sonar_issues.parquet`           | 680,794 (from 1,024,614 raw) |
| `clean_sonar_measures.parquet`         |  54,345 (from  66,711 raw) |
| `clean_szz.parquet`                    |  49,202 (from  52,428 raw) |
| `clean_jira_issues.parquet`            |  58,085 (from  61,402 raw) |
| `features_static.parquet`              |  23,911 (one row per (project, basename)) |
| `features_historical.parquet`          |  23,911 |
| `dataset_consequence.parquet`          |  23,911 rows × 66 cols (62 features + IDs/label) |
| `dataset_severity.parquet`             |  23,911 rows × 59 cols (56 features after leakage drop) |
| `dataset_szz.parquet`                  |  23,911 rows × 65 cols (62 features) |

22 of 31 projects are eligible (≥500 pre-snapshot commits and ≥50 post-snapshot commits). Snapshot policy: **median commit date** of each project's master branch. Observation window for the future signals that drive labels: **6 months**.

Per-project basename counts: smallest = `org.apache:daemon` (12 files), largest = `org.apache:hive` (5,307 files); median ~430.

---

## 3. First-version results: what was meaningful, what it yielded, how the update enhances

### Three label variants (already in the pre-enhancement run)

| Variant | Definition | Positive rate |
|---|---|---:|
| **Consequence** (primary) | top-20% by post-snapshot maintenance risk score = 0.5 × `bugfix_commits_future` + 0.3 × `future_churn` + 0.2 × `szz_defects_future`, normalised within project | 14.78% (3,533 / 23,911) |
| **Severity baseline** | `n_blocker_future + n_critical_future > 0` | 9.64% (2,305) |
| **SZZ baseline** | `n_szz_fixes_future > 0` | 1.24% (297) |

### Within-project 10-fold CV (`results/tables/within_project_summary.csv`)

| Variant | Best model | F1 | ROC-AUC | PR-AUC | MCC | CE@20 |
|---|---|---:|---:|---:|---:|---:|
| consequence | **lightgbm** | 0.581 | 0.897 | 0.652 | 0.506 | 0.702 |
| consequence | xgboost | 0.547 | 0.899 | 0.662 | 0.510 | 0.703 |
| consequence | random_forest | 0.525 | 0.900 | 0.663 | 0.497 | 0.716 |
| severity    | **xgboost** / lightgbm | 0.704 | 0.976 | 0.817 | 0.679 | 0.970 |
| severity    | random_forest | 0.680 | 0.974 | 0.811 | 0.662 | 0.964 |
| szz         | **lightgbm** | 0.273 | 0.926 | 0.230 | 0.266 | 0.889 |

**Read this honestly:**
- **Severity** is "easy" because the issue-count features carry strong residual signal even after dropping the leaky `n_blocker`/`n_critical`. It's a known limitation - this is why the proposal demands the consequence variant as primary.
- **Consequence** at F1≈0.58, ROC-AUC≈0.90, CE@20≈0.70 is the **headline number**. Tree ensembles dominate; logistic regression underperforms because the relationship is non-linear (size × churn × severity interactions).
- **SZZ** is hard because positives are extremely sparse (1.24%) and the leaky `szz_inducing_pre*` features were dropped. F1 of 0.27 with PR-AUC of 0.23 is realistic for this regime; the high ROC-AUC is misleading under heavy imbalance.

### Cross-project (LOPO, `results/tables/lopo_summary.csv`) - the harder test

| Variant | Best model | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|
| consequence | lightgbm | 0.418 | 0.776 | 0.466 |
| severity    | lightgbm | 0.613 | 0.958 | 0.708 |
| szz         | logistic_regression | 0.113 | 0.786 | 0.164 |

The **F1 drop from within-project to LOPO** (0.581 → 0.418 for consequence; 0.704 → 0.613 for severity) is the central scientific finding: features generalise across projects but at a measurable cost. SZZ effectively collapses cross-project, which matches Lenarduzzi et al.

### Which features were meaningful (top SHAP / permutation, pre-enhancement)

From `results/tables/shap_top15_consequence.csv` and `perm_top15_consequence.csv` (random_forest, full 10-fold mean):

Top contributors for **consequence**: `total_commits_pre`, `code_churn_pre`, `recent_commits_90d_pre`, `recent_churn_90d_pre`, `n_issues_open`, `complexity`, `ncloc`, `file_age_days_at_snapshot`, `n_major`, `cognitive_complexity`. Process / activity metrics dominate; static metrics are second; issue counts give the tie-breakers. This matches Hassan 2009 and confirms historical features carry most of the predictive power.

For **severity**, the top features are dominated by issue counts (`n_major`, `n_minor`, `n_code_smells`) - confirming the leakage concern that motivates the consequence variant.

### Limitations the new update is targeting

| Limitation in v1 | What v2 (current run) adds | Why it matters |
|---|---|---|
| No coupling features | **Co-change graph features** (`features_graph.parquet`): degree, strength, betweenness, closeness, clustering, pagerank, recency-windowed neighbour counts | Files defect together; structural centrality predicts maintenance risk. Jiang et al. 2024/2025. |
| No prior-defect signals | **Pre-snapshot defect features** (`features_priordefect.parquet`): `bugfix_commits_pre*`, `time_since_last_bugfix_days`, `linked_jira_bugs_pre`, `bug_density_pre` | Past bug-fix activity is the single strongest predictor of future defects (Hassan 2009, Kamei 2013). |
| Single point estimate per metric | **Bootstrap 95% CIs** (10k resamples) and **Wilcoxon paired tests** with Bonferroni correction | Without CIs and significance testing, claims like "lightgbm beats xgboost" are anecdotes. |
| No probability calibration | **Stage 7c**: Platt and isotonic calibration, ECE / Brier / NLL | Tree models give miscalibrated probabilities; CE@20 ranking is fine but actuarial use isn't. |
| No hyperparameter optimisation | **Stage 7b**: Optuna PR-AUC tuning, 30 trials × 5-fold inner CV | All v1 numbers used default hyperparameters. Tuning gives a defensible upper bound. |
| No alternative imbalance handling | **Stage 7d**: SMOTE vs `class_weight=balanced` head-to-head | Whether SMOTE actually helps on these features is an open empirical question. |
| Within-project CV is random K-fold | **Stage 7e**: temporal split T1→T2 (40th vs 70th percentile snapshots) | Shuffled CV optimistically leaks future information; temporal CV is the honest within-project test. |
| SVM never benchmarked | **SVM activated** in Stage 7 (within-project only; LOPO skips it because O(N²) is prohibitive) | Proposal explicitly lists SVM. |
| No fold-level prediction archive | **`within_project_predictions.parquet`** + **`lopo_predictions.parquet`** persisted | Required for paired significance tests and future post-hoc analyses. |
| Confusion matrices summarised verbally only | **3-variant × 6-model confusion-matrix grid** + per-project error CSV | Standard reporting expected by reviewers. |

---

## 4. Pipeline steps (current)

Each stage is one numbered Python script under `scripts/`. They write parquet/CSV/PNG into `data/processed/` and `results/`. Stages are reproducible from the cleaned parquets onward; Stages 1-3 only need to run when `td_V2.db` itself changes.

| # | Script | What it does | Output |
|---|---|---|---|
| 1 | `01_inspect_db.py`        | Schema dump, table row counts, sample rows | `results/tables/db_schema.csv`, `db_table_counts.csv`, `db_samples/*` |
| 2 | `02_profile_projects.py`  | Per-project commit/file/snapshot timeline; eligibility filter | `data/processed/project_snapshots.parquet` |
| 3 | `03_clean.py`             | Deduplicate, restrict to Java + non-test paths, basename-normalise, type-coerce | `data/processed/clean_*.parquet` |
| 3b/3c | `03b_debug_paths.py`, `03c_verify_path_format.py` | Path-format diagnostics (run manually if the cleaning rules change) | `results/tables/path_overlap_report.csv` |
| 4 | `04_label.py`             | Three label variants: consequence (primary), severity, SZZ | `data/processed/labels_*.parquet`, `results/tables/label_summary.csv` |
| 5 | `05_features.py`          | Static, historical, **graph (new)**, **prior-defect (new)** features | `data/processed/features_*.parquet`, `results/tables/feature_summary.csv` |
| 6 | `06_build_dataset.py`     | Join features + labels, drop leakage, produce 3 modelling tables | `data/processed/dataset_{consequence,severity,szz}.parquet`, `results/tables/dataset_summary.csv` |
| 7 | `07_train.py`             | Within-project 10-fold stratified CV, 6 models × 3 variants, persist fold preds, bootstrap CIs, Wilcoxon | `results/tables/within_project_{folds,summary,predictions}.{csv,parquet}`, `_with_ci.csv`, `pairwise_significance.csv` |
| 7b | `07b_tune.py`            | Optuna tuning for tree ensembles | `results/tables/tuned_params.json`, `within_project_summary_tuned.csv` |
| 7c | `07c_calibrate.py`       | Platt + isotonic calibration sweep | `results/tables/calibration_summary.csv`, `calibration_predictions.parquet` |
| 7d | `07d_resample.py`        | SMOTE vs class_weight comparison | `results/tables/within_project_summary_smote.csv`, `resampling_comparison.csv` |
| 7e | `07e_temporal.py`        | T1 → T2 temporal within-project split | `results/tables/temporal_summary.csv` |
| 8 | `08_lopo.py`              | Leave-One-Project-Out cross-project CV (5 models, no SVM), persist preds + CI + Wilcoxon | `results/tables/lopo_{folds,summary,predictions}.{csv,parquet}`, `lopo_vs_within.csv` |
| 9 | `09_sensitivity.py`       | Sensitivity grid (windows 3/6/12 mo × percentiles 10/20/30), feature-family ablation, SHAP, permutation importance | `results/tables/sensitivity_*.csv`, `feature_ablation.csv`, `shap_*.csv`, `perm_*.csv` |
| 10 | `10_report.py`           | Compose `docs/06_results.md` and `docs/07_discussion.md`, render figures | `results/figures/*.png`, `docs/06_results.md`, `docs/07_discussion.md` |

---

## 5. Per-file analysis - confirmation

**Yes.** Granularity is **(project, basename)** throughout:
- Features are aggregated at the basename level (file name without directory).
- Labels are at the basename level (top-20% within project for consequence, etc.).
- Train/test splits are stratified within project at the basename level.
- LOPO holds out an entire project's basenames at a time.

We do **not** model directories, packages, classes, or methods. One row per source file.

The basename normalisation is intentional and documented: it lets us reconcile Sonar's path conventions with Git's path conventions (which differed between the data source's commit log and Sonar's analysis URLs). The trade-off is that two files in different directories with the same basename collapse into one record. This is rare in the cleaned dataset and accepted as a known limitation.

---

## 6. Running the whole pipeline

The new (2026-04-28) `run_pipeline.py` runs every numbered script under `scripts/` in order and tee-logs each stage to `results/run_logs/`. It's the single entry point.

```powershell
.\venv\Scripts\python.exe run_pipeline.py
```

Selectively skip stages or start from a particular one:

```powershell
# Skip already-finished early stages
.\venv\Scripts\python.exe run_pipeline.py --skip 1,2,3,4

# Start from stage 7 (assumes 1-6 are already cached)
.\venv\Scripts\python.exe run_pipeline.py --from 7

# Run only a specific subset
.\venv\Scripts\python.exe run_pipeline.py --only 7b,7c,7d
```

End-to-end runtime estimate on this machine (Windows, no GPU):
- Stages 1-4 (one-time, already cached): ~10 min combined
- Stage 5 (features, with new graph + prior-defect): ~60-90 min
- Stage 6: ~2 min
- Stage 7 (within-project + SVM + persistence + significance): ~10-15 min
- Stage 7b (Optuna): ~60-90 min
- Stage 7c (calibration sweep): ~30-45 min
- Stage 7d (SMOTE sweep): ~20-30 min
- Stage 7e (temporal: rebuilds features at T1 and T2): ~2-3 hr
- Stage 8 (LOPO): ~30 min
- Stage 9 (ablation, sensitivity, SHAP, permutation): ~20-30 min
- Stage 10 (report): ~5 min

**Total fresh end-to-end: 6-9 hours.** From cached cleans (Stages 1-4 done): ~5-8 hours.
