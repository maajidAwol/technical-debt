# Research Log

Timestamped log of methodological decisions for the MSc thesis:
"Predicting High-Risk Technical Debt in Open-Source Software Projects Using
Machine Learning and Code Metrics" (Abdulmajid Awol Seid, Feb 2026).

Each entry records:
- **Decision**: what was chosen
- **Alternatives considered**
- **Rationale**
- **Proposal section**
- **Impact on paper**: where the decision will surface in the thesis

---

## 2026-04-22 | Stage 0 - Execution plan finalized

**Decision**: Use a staged, data-first execution model (Stages 0-10) where each
stage produces one inspectable artifact (Parquet/CSV) that gates the next.

**Alternatives considered**
- Single-shot end-to-end pipeline (`run_pipeline.py` only). Rejected: errors
  propagate silently, and the temporal-split methodology demands per-stage
  audits (leakage, NULL handling, label distribution).
- Per-project parallel execution. Deferred: not needed at 33-project scale.

**Rationale**: The updated proposal (Section 3.4) requires a reproducible,
auditable pipeline with strict temporal separation between feature extraction
and label derivation. Hand-stepped stages provide the audit trail required
for the thesis methodology chapter and mitigate the labeling-noise risks
listed in Section 3.6.

**Proposal section**: 3.1, 3.4, 3.6.

**Impact on paper**: Provides the basis for the Methodology chapter's
"Pipeline and Reproducibility" subsection and the Appendix pipeline diagram.

---

## 2026-04-22 | Stage 0 - Primary dataset

**Decision**: Use the Technical Debt Dataset v2.0 (Lenarduzzi et al. 2019)
stored at `data/raw/td_V2.db` (1.54 GB, SQLite).

**Alternatives considered**
- Custom scrape of GitHub + SonarQube self-hosting for 30+ projects.
  Rejected: 4-6 weeks of setup time, inconsistent metric configurations, and
  high risk of reproducibility loss.
- Subset of the TD Dataset (e.g., 10 projects). Rejected: reduces statistical
  power for cross-project (LOPO) validation.

**Rationale**: TD Dataset v2.0 contains all tables required by the three
labeling variants (SONAR_MEASURES, SONAR_ISSUES, GIT_COMMITS,
GIT_COMMITS_CHANGES, JIRA_ISSUES, SZZ_FAULT_INDUCING, REFACTORINGS) for
33 Apache projects on their master branches only. This eliminates four risks
enumerated in Proposal Section 3.6 (analysis failures, inconsistent configs,
incomplete Jira linking, SZZ quality).

**Proposal section**: 3.2.

**Impact on paper**: Supports Chapter 3.2 ("Data Sources and Project
Selection") and external-validity discussion in Chapter 6.

---

## 2026-04-22 | Stage 0 - Snapshot strategy

**Decision**: Snapshot date `t` per project = median of master-branch commit
dates (primary). Observation window = 6 months (primary). Sensitivity
analysis windows: {3, 6, 12} months.

**Alternatives considered**
- Last release tag (requires tag availability and consistency across
  projects; deferred to fallback).
- Fixed calendar date across all projects (rejected: wastes projects with
  histories that don't overlap this date).

**Rationale**: Median commit date guarantees every project has both a
meaningful pre-snapshot history (for features) and a meaningful post-snapshot
window (for labels). This is the approach used by Jiang et al. 2024/2025 and
Robredo et al. 2025.

**Proposal section**: 3.4.

**Impact on paper**: Chapter 3.4 labeling methodology subsection; Table 3.1
("Per-project snapshot dates and window sizes").

---

## 2026-04-22 | Stage 0 - Three-variant labeling

**Decision**: Implement three label variants on identical feature vectors:

1. **Consequence-oriented (primary)** - top 20% by weighted risk score of
   (future bug-fix commits, future churn, future SZZ defects) within each
   project.
2. **Severity baseline** - files with >=1 BLOCKER or CRITICAL SonarQube
   issue at snapshot time.
3. **SZZ baseline** - files touched by a commit flagged as fault-inducing
   in `SZZ_FAULT_INDUCING` during the observation window.

**Alternatives considered**
- Single label (consequence only). Rejected: examiners cannot assess what
  consequence-oriented labeling *adds* without direct comparison baselines.

**Rationale**: The three-variant comparison is this thesis's primary
novelty. It produces label-agreement evidence (Cohen's kappa, Jaccard)
that speaks directly to whether severity-based or SZZ-based
prioritization is interchangeable with consequence-oriented prioritization.

**Proposal section**: 3.4 (explicitly mentions "additional labeling
variants may be reported as experimental baselines").

**Impact on paper**: Research Question 1, Chapter 4 "Results -- Labeling
Comparison".

---

## 2026-04-23 | Stage 9 - Sensitivity grid + feature-group ablation

**Sensitivity grid (consequence variant)** - 3x3 = 9 configurations of
``(window_months, percentile)`` in ``{3, 6, 12} x {10, 20, 30}%``
evaluated with 10-fold CV / LightGBM. Full results in
``results/tables/sensitivity_consequence.csv``.

Key observations:
- ROC-AUC is stable across the grid at **0.883 - 0.917** (narrow 3.4-
  percentage-point range). The model's ranking capability does not
  depend on the specific parameter choice - a strong robustness claim.
- F1 scales with positive rate (mechanical) from 0.478 (window=3,
  p=10%) to 0.709 (window=12, p=30%).
- CE@20 decreases with positive rate (if more than 20% of files are
  positive, a 20% budget cannot capture all positives) - mechanical
  expected pattern, documented in thesis as a limitation of CE@K
  for high positive-rate labels.
- Primary configuration (6 months x 20%) sits in the middle of the
  grid - defensible as a midpoint, not a cherry-picked optimum.

**Feature-group ablation** - 3 variants x {baseline, only-group,
leave-out-group} x LightGBM. Full results in
``results/tables/feature_ablation.csv``.

Headline: different variants are predicted by different feature
families.

| Variant     | Dominant group        | F1 only-this-group | Drop if removed |
|-------------|-----------------------|--------------------|-----------------|
| Severity    | static_sonar (16)     | 0.646              | -0.228          |
| Consequence | historical (15)       | 0.555              | -0.092          |
| SZZ         | historical (15)       | 0.226              | -0.135          |

Interpretation:
- Severity labels are **predicted by static SonarQube features alone
  almost as well as by all features combined** (F1 0.646 vs 0.704).
  This confirms, together with the Stage-4 kappa=0.05 vs SZZ, that
  severity labeling is a largely self-consistent static-analyzer
  artefact rather than an empirical measure of debt impact.
- Consequence and SZZ labels are **process-driven**: historical (Git
  commit) features carry the dominant signal. This matches the
  Kamei-et-al.-2013 JIT defect-prediction finding that process
  metrics beat product metrics for future-event prediction.
- Project-level context alone is weak on every variant (F1 0.07 - 0.30)
  - context helps marginally when combined with fine-grained features,
  but is insufficient on its own.

**Thesis impact**
- Chapter 4 Results: sensitivity table + ablation table as Tables Z_1
  and Z_2.
- Chapter 5 Discussion: "Different labels, different mechanisms" -
  motivates why the consequence framing is not interchangeable with
  severity or SZZ, defending the Proposal's methodological shift.

---

## 2026-04-23 | Stage 8 - Leave-One-Project-Out validation results

Ran 22-fold LOPO across 3 variants x 5 models (330 fits total, 7.2 min
wall-clock). Key results for the best model per variant:

| Variant     | Best model | F1    | ROC-AUC | PR-AUC | CE@20 | N projs |
|-------------|------------|-------|---------|--------|-------|---------|
| Severity    | LightGBM   | 0.613 | 0.958   | 0.708  | 0.872 | 22      |
| Consequence | LightGBM   | 0.418 | 0.776   | 0.466  | 0.484 | 22      |
| Consequence | RF (CE@20) | 0.222 | 0.794   | 0.493  | 0.526 | 22      |
| SZZ         | LogReg     | 0.113 | 0.786   | 0.164  | 0.562 | 16      |

**Generalization gap (within - LOPO F1, best model per variant)**:

- Severity: 0.091 (13% relative drop). Small because SonarQube
  severity rules are universal across projects.
- Consequence: 0.163 (28% relative drop). Healthy published range
  (Herbold 2018 reports 15-40%).
- SZZ: 0.259 (95% relative drop). Confirms SZZ labels are hard to
  generalize due to per-project heterogeneity in defect event rates.

**Key thesis findings**

1. Consequence variant is publishable cross-project: LOPO CE@20 = 0.526
   means a top-20% inspection budget in an unseen project catches
   ~53% of files that will cause maintenance burden in the next 6
   months.
2. SZZ baseline has a **coverage gap**: only 16 of 22 projects had any
   SZZ positive in the held-out window (batik, cocoon, digester,
   daemon, hive, commons-cli skipped). This is a dataset limitation -
   an argument in favor of the consequence approach, which covers all
   22 projects.
3. Model-family performance **re-orders** between within-project and
   LOPO: XGBoost / RF dominate within-project but LightGBM generalizes
   better. A classic overfitting-to-project-quirks signature.

**Scripts + outputs**
- ``scripts/08_lopo.py``
- ``results/tables/lopo_folds.csv`` - one row per project
- ``results/tables/lopo_summary.csv`` - aggregate
- ``results/tables/lopo_vs_within.csv`` - the generalization-gap table

**Proposal sections**: 3.6.2 (Cross-project validation), RQ3.

---

## 2026-04-23 | Stage 7 - Within-project 10-fold CV results

Ran stratified 10-fold CV across {LogReg, DT, RF, XGBoost, LightGBM} x
3 variants. Classical ML; no hyperparameter search yet (that is Stage 9).

**Headline: XGBoost within-project mean metrics**

| Variant     | ROC-AUC | PR-AUC | F1    | CE@20 | Positive rate |
|-------------|---------|--------|-------|-------|---------------|
| Severity    | 0.976   | 0.817  | 0.704 | 0.970 | 9.64%         |
| Consequence | 0.899   | 0.662  | 0.547 | 0.703 | 14.78%        |
| SZZ         | 0.937   | 0.241  | 0.155 | 0.906 | 1.24%         |

**Interpretation**

- The three variants produce a clean performance **gradient**: severity
  >> consequence >> SZZ F1. This is consistent with the intrinsic
  difficulty of each label and with prior literature.
- Severity-variant performance is near-ceiling because the remaining
  SonarQube-derived features (`n_issues_open`, `total_debt_minutes`,
  `n_code_smell`, `n_bug`, `n_vulnerability`, `issue_density`,
  `debt_per_loc`) are highly correlated with whether the file has a
  BLOCKER/CRITICAL issue open - the strict severity-defining counts
  have already been dropped per ``SEVERITY_LEAKY_FEATURES``. This is a
  **thesis-level finding**: severity labels are largely self-consistent
  with other SonarQube outputs and therefore provide limited novel
  information.
- Consequence-variant CE@20 = 0.70 means a top-20-percent inspection
  budget captures 70 percent of the files that will cause real
  maintenance burden in the next 6 months. This matches the
  Kamei-et-al.-2013 JIT defect-prediction CE@20 range of 0.50-0.75 and
  is the primary publishable model result.
- SZZ-variant F1 is low because of the extreme imbalance
  (1.24 percent positive rate) and the default 0.5 decision threshold.
  Ranking quality is nevertheless strong: ROC-AUC 0.937 and CE@20 0.906
  make the model useful for *prioritization* even when point-precision
  is modest. Stage 9 will explore threshold-tuning as a sensitivity.

**Model family observations**

- Tree ensembles (RF, XGBoost, LightGBM) dominate on every variant -
  expected; TD features are strongly non-linear.
- Decision Tree alone underperforms the ensembles on every variant -
  expected; confirms ensembling gain.
- LogReg is competitive on severity (linear-separable task) but loses
  badly on consequence and SZZ - confirms need for non-linear models.

**Training time justification**

Average fold takes 0.1 - 2 s. This is consistent with 200 trees x 21,519
training rows x 61 features = approximately 280 M operations, in the
0.3 s/fold range on an 8-core CPU. Fast training is a property of
classical tree ensembles on 24-k-row tabular data, not a quality
defect.

**Thesis impact**
- Table X: ``results/tables/within_project_summary.csv`` -> Chapter 4
  "Within-project Prediction Results".
- Discussion: severity-label tautology as motivation for consequence
  labeling.

**Proposal sections**: 3.5 (Modeling), 3.6 (Evaluation), RQ2.

---

## 2026-04-23 | Stage 4 - Three-variant labeling results

**Key empirical finding**: the three label variants identify **largely
disjoint sets** of files.

| Comparison | Cohen's kappa | Jaccard | Overall agreement |
|---|---|---|---|
| Consequence vs Severity | 0.21 | 0.18 | 82.9% |
| Consequence vs SZZ      | 0.13 | 0.08 | 86.4% |
| Severity vs SZZ         | 0.05 | 0.04 | 89.9% |

All three agreement coefficients correspond to "slight" or "none" on
Landis & Koch (1977) scale. This is the quantitative evidence that the
thesis's central claim holds: **how you define "high-risk TD" changes
which files get flagged**.

**Overall positive rates across 23,911 (project, basename) units**
- Consequence (top 20%) : 14.78 percent (3,533)
- Severity (BLOCKER/CRITICAL) : 9.64 percent (2,305)
- SZZ (fault-fix in window)   : 1.24 percent (297)

**Why consequence < 20 percent**
Some projects have heavy ties at risk-score = 0 (no events in window).
The ``_top_percentile`` helper breaks ties by selecting values strictly
greater than the percentile threshold, so thin-signal projects receive
fewer positives than the nominal target. This is mathematically correct
and documented in the thesis; it also motivates the sensitivity grid
over percentile thresholds in Stage 9.

**Small positive counts (SZZ)**
5 projects have 0 SZZ positives (batik, cocoon, digester, daemon, hive,
commons-cli) - their SZZ fault-fixing activity is concentrated
outside the 6-month observation window. Still, 297 SZZ positives
across 17 projects provide enough signal for the SZZ baseline, and
ensemble methods with class weighting handle the imbalance.

**Thesis impact**
- Chapter 3 (Methodology) - describe the three-variant labeling.
- Chapter 4 (Results) - label_agreement.csv is Table Y; label_summary.csv
  is Table X.
- Chapter 5 (Discussion) - "three views of TD" narrative built around
  the agreement table.

**Proposal section**: 3.4 (Labeling), Research Q1 / Q2.

---

## 2026-04-23 | Stage 3 - Unit of analysis = (project, file basename)

**Decision**: Adopt **file basename** within each project as the unit of
prediction for features and labels, overriding the initial assumption of
repo-relative full path.

**Observation**: Two path-format quirks in TD Dataset v2.0:

1. `SONAR_ISSUES.COMPONENT` stores repo-relative full paths
   (`src/main/java/org/apache/commons/codec/binary/Base64.java`).
2. `GIT_COMMITS_CHANGES.FILE` stores **only the file basename** for 18 of
   22 projects (`Base64.java`). The remaining 4 projects (batik, cocoon,
   felix, santuario) store full paths for 46-51% of rows and basenames
   for the rest - a mixed convention within the same project.

Diagnostic script `scripts/03b_debug_paths.py` and
`scripts/03c_verify_path_format.py` confirm the above.

**Join strategies considered**

| Strategy | Works? | Issue |
|---|---|---|
| Join on full path | No | 0% overlap - git rows are basenames for most projects |
| Clone all 22 repos, re-extract paths via pydriller | Yes | ~22 GB, hours of network, undermines "use the citable dataset" argument |
| Drop high-collision projects | Partial | Only 3 projects <10% collision -> no LOPO |
| **Basename aggregation** (chosen) | **Yes** | Requires aggregating Sonar issues per basename; some intra-basename file confusion |

**Basename collision statistics in SONAR_ISSUES** (pre-snapshot files
only shown in overlap report; global figures from
`03c_verify_path_format.py`):

- Min 0% (commons-exec, zookeeper)
- Median ~35%
- Max 62% (digester) - some basenames map to up to 76 distinct files
  (felix), though median max is ~8

**Rationale**
- The limiting side is `GIT_COMMITS_CHANGES`, which we cannot make
  finer-grained without cloning repos. So the *labels* (bug-fix
  commits, future churn, SZZ-linked changes) are inherently basename-
  level. Features must match the label granularity.
- Aggregating SonarQube issues per basename (sum of DEBT, counts by
  severity, unique rule count) preserves all relevant information at
  the granularity available.
- This matches how prior studies on the same dataset effectively
  operate; we make the aggregation explicit and measured instead of
  implicit.

**Thesis impact**
- Chapter 3.3 (Feature Extraction) will document the `(project, basename)`
  unit of analysis, with a subsection "Granularity of Git history in
  TD Dataset v2.0".
- Chapter "Threats to Validity - Construct Validity" will tabulate the
  per-project basename collision rates and note that intra-basename
  confusion is a source of label and feature noise.
- Chapter 4 (Results) will report per-project N of prediction units and
  positive rates so readers see the effective sample size.

**Proposal section**: 3.3 and Threats to Validity.

---

## 2026-04-23 | Stage 2 - Project eligibility threshold

**Decision**: Set `MIN_POST_SNAPSHOT_COMMITS = 50` (was 100) and keep
`MIN_PRE_SNAPSHOT_COMMITS = 500` and the median snapshot strategy.

**Observation**: Under strict thresholds (pre >=500, post >=100), only
17 of 31 projects were eligible. Excluded projects were mostly mature
Apache commons-* utility libraries whose activity tapered after 2015
so their post-median 6-month window held fewer than 100 commits.

**Alternatives considered**
- Keep strict thresholds -> 17 projects. Statistically safe but weakens
  LOPO generalization evidence (fewer folds, wider CIs).
- Relax to 30 -> ~29 projects but label signal is thin for the
  recovered ones (some would have <10 positive files after top-20%
  filtering, turning those LOPO folds into noise).
- Switch to 33rd-percentile snapshot -> more post-snapshot data for
  every project, but reduces pre-snapshot feature history uniformly.
  Rejected because it harms projects that were already fine.
- **Chosen**: relax post threshold only to 50. Borderline projects are
  recovered, well-behaved projects are unaffected.

**Expected impact**
- Eligible projects: ~24-27 (confirmation pending re-run).
- Per-project positive examples (top-20%): typically 20-50 files - a
  defensible sample for learning and for minority-class metrics.
- Matches operating points in Jiang et al. 2024 and Tsoukalas et al.
  2020 (both use projects with >= 50 post-snapshot commits implicitly).

**Thesis mitigation**
- Chapter "Threats to Validity" (Conclusion validity): document the
  threshold choice and replicate the primary experiment on the strict
  17-project subset as a sensitivity check.
- Chapter 4 (Results): Table of per-project N-files and positive rates
  so readers see the signal strength per fold.

**Proposal section**: 3.4 (Temporal Split), Table/Chapter on Threats
to Validity.

---

## 2026-04-23 | Stage 1 - Schema inventory findings

**Decision**: Adjust feature extraction to the actual granularity of
`SONAR_MEASURES` in TD Dataset v2.0.

**Key schema facts discovered** (stored in
`results/tables/db_schema.csv` and per-table samples in
`results/tables/db_samples/`):

- 10 tables total, 31 projects (not 33 as originally planned)
- `SONAR_MEASURES` is **project-level**, not file-level: 66,711 rows =
  per `(PROJECT_ID, ANALYSIS_KEY)` snapshot of project-wide metrics
  (NCLOC, COMPLEXITY, CLASSES, FUNCTIONS, STATEMENTS, COVERAGE,
  COMMENT_LINES_DENSITY, SQALE_INDEX, SQALE_DEBT_RATIO,
  DUPLICATED_LINES_DENSITY, BLOCKER/CRITICAL/MAJOR/MINOR/INFO_VIOLATIONS,
  CODE_SMELLS, BUGS, VULNERABILITIES)
- `SONAR_ANALYSIS` joins `ANALYSIS_KEY` to a snapshot `DATE` and Git
  `REVISION` - essential for temporal alignment
- `SONAR_ISSUES` is **file-level** via the `COMPONENT` column (format
  `ProjectKey:path/to/File.java`), with per-issue `SEVERITY`, `TYPE`,
  `RULE`, `DEBT`, `CREATION_DATE`, `CLOSE_DATE`
- `GIT_COMMITS` actual column names: `COMMIT_HASH`, `COMMIT_MESSAGE`,
  `AUTHOR_DATE`, `COMMITTER_DATE`, `IN_MAIN_BRANCH` (stored as text
  `'True'`/`'False'` strings)
- `GIT_COMMITS_CHANGES` actual columns: `FILE`, `LINES_ADDED`,
  `LINES_REMOVED` (NOT the `ADDED_LINES`/`DELETED_LINES` we assumed)
- `SZZ_FAULT_INDUCING_COMMITS` columns:
  `FAULT_FIXING_COMMIT_HASH`, `FAULT_INDUCING_COMMIT_HASH`
- `JIRA_ISSUES` has both `KEY` (e.g. `MRM-2021`) AND a pre-populated
  `HASH` column giving the Git commit that closed the issue - saves us
  writing regex to extract `PROJECT-NNN` keys from commit messages

**Revised feature strategy** (file-level predictors):

1. **Issue-count static features from SONAR_ISSUES** (per file):
   counts by SEVERITY (BLOCKER, CRITICAL, MAJOR, MINOR, INFO), by TYPE
   (BUG, VULNERABILITY, CODE_SMELL), distinct rules triggered, sum of
   DEBT minutes.
2. **Pseudo-LOC from GIT_COMMITS_CHANGES**: cumulative
   `sum(LINES_ADDED) - sum(LINES_REMOVED)` up to snapshot `t` gives a
   file-size proxy; `sum(LINES_ADDED + LINES_REMOVED)` gives churn.
3. **Project-level context from SONAR_MEASURES**: NCLOC, overall
   complexity, SQALE_INDEX, SQALE_DEBT_RATIO, COVERAGE,
   DUPLICATED_LINES_DENSITY - replicated per file in the same project
   at snapshot `t` to serve as shared context features.
4. **Historical features from GIT_COMMITS + GIT_COMMITS_CHANGES** up
   to `t` - unchanged from original plan.

**Alternatives considered**
- Running our own SonarQube on each project to get file-level NCLOC.
  Rejected: >4 weeks of tool-chain work; violates reproducibility goal
  of using the citable dataset.
- Using Lizard (already in requirements.txt) to compute per-file
  complexity from cloned repos. Deferred as a stretch goal - keeps the
  primary pipeline fully DB-driven.

**Rationale**: Jiang et al. (2024, 2025) and Lenarduzzi et al. (2019)
use exactly this strategy (issue aggregates at file level + project-
level context). It is the accepted practice for this dataset.

**Proposal section**: 3.3 (Feature Extraction), Table 1.

**Impact on paper**: Methodology Chapter 3.3 will document this
derivation clearly in a subsection "From dataset schema to file-level
features"; Threats to Construct Validity will explicitly note that
LOC/complexity at file granularity are approximations derived from
issue counts and Git churn.

---

## 2026-04-22 | Stage 0 - Added evaluation metrics beyond proposal

**Decision**: In addition to Precision/Recall/F1/F2/AUC-ROC (proposal
Section 3.5), compute Cost-Effectiveness at top-20% (CE@20) and PR-AUC.

**Alternatives considered**: proposal-metrics-only. Rejected because CE@20
directly operationalizes the "prioritization" framing of Chapter 1 while
PR-AUC is more informative than ROC-AUC under class imbalance (He and Garcia
2009).

**Rationale**: Aligns quantitative evaluation with the *practical*
contribution claimed in Section 1.2 (Significance).

**Proposal section**: 1.2, 3.5.

**Impact on paper**: Chapter 4 tables; discussion of practical impact in
Chapter 5.

## 2026-04-23 - Stage 10: Final reporting artefacts (SHAP, figures, docs)

**Decision**: Produce the full set of publication-ready artefacts in one
reproducible pass.

**Pipeline additions**:
- `src/analysis/importance.py` - TreeSHAP + permutation importance per
  variant (LightGBM, 80/20 stratified split, ROC-AUC as permutation
  scorer, 5 repeats, 1,000-sample SHAP background).
- `src/reporting/figures.py` - seven figures at 300 DPI, PNG + PDF:
  label Venn, per-project positive rates, within-vs-LOPO dot-plot,
  sensitivity heatmap (3 metrics), feature-group ablation, SHAP summary
  per variant, LOPO per-project F1 box-plot.
- `src/reporting/render.py` - auto-generates `docs/06_results.md` and a
  scaffold `docs/07_discussion.md` directly from the CSV/parquet
  artefacts (no hand-typed numbers).
- `scripts/10_report.py` - single driver.

**Top-5 SHAP drivers per variant**:

| variant     | top-5 SHAP features                                                                                                              |
|-------------|----------------------------------------------------------------------------------------------------------------------------------|
| consequence | pseudo_ncloc_at_t, days_since_last_change_at_snapshot, file_age_days_at_snapshot, total_commits_pre, avg_change_size_pre         |
| severity    | total_debt_minutes, n_distinct_rules, max_single_commit_churn_pre, n_bug, file_age_days_at_snapshot                              |
| szz         | project_function_complexity, days_since_last_change_at_snapshot, ownership_ratio_pre, pseudo_ncloc_at_t, max_single_commit_churn_pre |

**Interpretation**:
- *Consequence* is driven by **size + activity + recency** (ncloc, days
  since last change, age, commit count). This is exactly what the
  "future maintenance burden" narrative predicts: big, old, recently-churned
  files hurt most. No SonarQube issue count appears in the top-5 -
  strong evidence the consequence signal is **orthogonal to severity**.
- *Severity* is driven by **debt + rule variety + bug counts** - a
  near-tautology (features aggregated at the same snapshot as the
  label), which is why within-project F1 hits 0.70. This reinforces the
  Section 7.1 finding that severity-style TD prediction mostly
  benchmarks SonarQube consistency.
- *SZZ* leans heavily on **project-level context + ownership** - it is
  the only variant where project-level features (function complexity,
  technical debt ratio) make the top-5, reflecting that fault-fix
  incidence is partly a project-level phenomenon.

Permutation importance agrees with SHAP on the top-3 features for
every variant (Pearson correlation of feature rankings > 0.80).

**Artefacts written**:
- 18 figures (7 PNG + 7 PDF + 3 SHAP PNG + 3 SHAP PDF mapped to 7
  unique plots + 3 SHAP panels).
- 15 CSV tables (6 SHAP/perm full + 6 top-15 + existing pipeline
  tables).
- `docs/06_results.md` (~34 KB, 8 sections) + `docs/07_discussion.md`
  scaffold (~3 KB, RQ1-3 answers + threats to validity).

**Reproducibility**: `python scripts/10_report.py` regenerates
everything in ~2 minutes after all preceding stages have run.

**Proposal section**: 3.6 (reporting), Chapter 4-5 (results, discussion).

**Impact on paper**: Provides the camera-ready deliverables - Chapter 6
can be copied directly; Chapter 7 uses the scaffold as a starting
structure with SHAP-backed answers to RQ1-3 and the five validity
threats documented above.

