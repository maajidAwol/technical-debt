# Advisor status update — MSc thesis (technical debt prediction)

**Student:** Abdulmajid Awol Seid  
**Advisor:** Dr. Tesfaye Gidey  
**Date:** May 2026  

---

Dear Dr. Gidey,

Below is an expanded plain-language summary of the empirical work for my thesis on predicting high-risk technical debt from repository and quality data. I keep jargon modest but add technical specificity where it clarifies decisions; exact artefacts appear only where helpful.

**1. Dataset.** I worked with the public Technical Debt Dataset (version 2.0), a single SQLite database spanning thirty-one Apache Java projects curated by Lenarduzzi et al. (*Proc. MSR*, Montreal, 2019). It integrates SonarQube analyses and findings, Git commits and per-commit file edits, JIRA issue histories where linked, SZZ-style fault-inducing links, plus ancillary catalogue rows—everything needed for reproducible mining without rerunning Sonar ourselves. Releases are maintained here: https://github.com/clowee/The-Technical-Debt-Dataset/releases  

The operational tables are approximately at the scales below (row counts from a full inventory of the shipped database—the modelling pipeline actively joins Git + Sonar + JIRA + SZZ surfaces):

| Table (purpose in brief) | Scale (~rows) |
|--------------------------|----------------|
| `GIT_COMMITS_CHANGES` — file-level edits per commit | ~1.1 million |
| `SONAR_ISSUES` — rule violations / smells per component | ~1.0 million |
| `GIT_COMMITS` — commit metadata on master-branch timeline | ~154 thousand |
| `REFACTORING_MINER` — detected refactor operations (corpus-wide auxiliary) | ~362 thousand |
| `SONAR_ANALYSIS` — Sonar analysis episodes per revision | ~68 thousand |
| `SONAR_MEASURES` — project-level metric snapshots | ~67 thousand |
| `JIRA_ISSUES` — issue tracker records | ~61 thousand |
| `SZZ_FAULT_INDUCING_COMMITS` — inducing/fix links | ~52 thousand |
| `SONAR_RULES` — rule catalogue | ~2 thousand |
| `PROJECTS` — bridge keys between Git / JIRA / Sonar | 31 |

**2. Project selection (logic).** Eligibility is computed **per project** after fixing snapshot date **t** as the **median master-branch commit timestamp** (`IN_MAIN_BRANCH = True`), which yields a stable divide between past (features) and future (labels). A project is retained only if it has **≥500 commits strictly before t** (enough history for process metrics and graphs) and **≥50 commits strictly after t within the primary six-month observation window** (enough future signal for consequence and SZZ-style labels). Nine catalogue projects fail these guards; **twenty-two** pass and form the thesis corpus. This policy intentionally trades off breadth for temporal honesty—thin histories would either starve features or make percentile-based consequence labels unstable.

**3. Data preparation.** All extraction respects **t**: features aggregate events with timestamps ≤ t; labels aggregate post-*t* windows only. Git stores paths inconsistently (often basename-only in change logs versus full paths in Sonar components), so every quantitative comparison lives at **basename × project** granularity with documented collision implications. Cleaning covers timezone-normalised timestamps, numeric coercion, duplicate suppression on natural keys, and cross-table joins sanity-checked against snapshot dates.

**4. Three-way labelling and how they work together.** The three labels answer **different notions** of “high risk” on the **same files**. **Consequence-oriented** risk scores normalised bug-fix cadence, churn, and fault-linked touches inside the six-month horizon (weights 0.5 / 0.3 / 0.2 reflecting signal reliability), then flags the **upper ~20%** per project—this is the thesis **prioritisation target**. **Severity** flags open Sonar **BLOCKER** or **CRITICAL** issues at **t**—a control showing how far tool severity diverges from forward-looking burden. **SZZ-style** positives flag files touched by fault-fixing commits in the window—a fault-history baseline comparable under identical splits. Together they triangulate reality: empirically they overlap weakly (pairwise agreement resembles “slight / none” on standard scales), so treating any single definition as ground truth would silently bake in a particular ideology of debt; running them in parallel exposes **which signals transfer when labels disagree**.

**5. Feature engineering (families and rationale).** Predictors are grouped conceptually as follows—each row states why the family belongs in a maintenance-risk study:

| Feature family | Technical role | Why it was selected |
|----------------|------------------|---------------------|
| Static Sonar aggregates @ *t* | File-level counts of issues, debt minutes, bugs/vulnerabilities, rule diversity | Captures immediate code-health portrait matching severity discourse |
| Historical Git process ≤ *t* | Churn, contributors, recency, ownership concentration, change-size moments | Literature shows **process** dominates **product** for forward-looking fault/TD proxies |
| Project Sonar context ≤ *t* | Whole-project complexity, coverage, duplication, SQALE metrics replicated per file | Controls for repository-wide climate inflating/deflating local readings |
| Co-change centralities ≤ *t* | Degree/strength/betweenness/closeness/PageRank on commit-co-touch graphs | Structural coupling beyond single-file metrics |
| Prior defect signals < *t* | Bug-fix counts in rolling windows, time since last bug-fix, JIRA bug density, inducing-commit tallies | Proxies historical fault proneness (trimmed when predicting SZZ to avoid trivial autocorrelation) |

After leakage trimming the consequence variant carries on the order of **80+** numeric descriptors per row; severity/SZZ variants drop a handful of columns each where those columns would duplicate the label semantics.

**6. Modelling formulation.** Each stream is **supervised binary classification**: minimise loss predicting high-risk membership from pre-*t* vectors. Training uses **stratified k-fold cross-validation** so rare positives appear proportionally in every fold; cross-project evaluation swaps entire projects into hold-out sets to mimic cold-start deployment; temporal splits train on an earlier chronological slice and test on a later slice inside the same repository to penalise subtle temporal leakage random folds ignore.

**7. Training (technical choices).** Candidate learners span **logistic regression** (linear separability baseline), **single decision trees**, **random forests** (variance reduction via bagging), **gradient boosting** (XGBoost + LightGBM; sequential residual fitting), and **RBF-kernel SVMs** for within-project experiments only—the quadratic memory footprint makes leave-one-project-out infeasible at corpus scale. Class imbalance is addressed primarily through **inverse-frequency class weights** in the loss so rare positives matter without inventing synthetic points by default.

**8. Optimisation and refinement (what changed).** Hyperparameters were searched with **Optuna** using a tree-structured Parzen estimator, optimising **mean precision–recall AUC on inner stratified folds**—the recommended objective under imbalance—before refitting on outer folds. For strong boosted models this typically lifts within-project **F1 on the consequence variant on the order of several percentage points** versus defaults (random forests and shallow trees gain even more depth/leaf regularisation). **Probability calibration** (Platt scaling + isotonic regression with nested cross-fitting) materially lowers **expected calibration error** on severity-style probabilities without harming ranking metrics, which matters if downstream tooling thresholds raw scores. **SMOTE vs class-weight** comparisons showed modest deltas on the headline boosted models—enough to justify retaining the simpler weighting as default while documenting where oversampling helped fringe cases.

**9. Evaluation metrics (why each matters).** I report a concise battery so readers cannot cherry-pick a single favourable number:

| Metric | Role |
|--------|------|
| Precision / Recall / **F1** | Harmonic trade-off for imbalance; standard comparability |
| **ROC-AUC** | Threshold-free ranking vs random ordering |
| **PR-AUC** | Stresses minority-class ranking—aligned with tuning objective |
| **MCC** | Balanced correlation coefficient (−1…1) resistant to unbalanced baselines |
| **CE@20** | Recall captured when inspecting only the **top 20%** of ranked files—direct operational analogue of fixed inspection budget |

Bootstrap confidence intervals and Bonferroni-corrected paired tests accompany headline comparisons so ranking claims survive multiplicity.

**10. Model choice and consequence weights — technical rationale.** Boosted trees won overall because they combine **nonlinear interactions**, **automatic feature selection pressure**, and efficient handling of mixed-scale inputs without manual basis expansions—exactly where logistic regression plateaus and shallow trees overfit. Random forests interpolate but rarely surpass well-tuned boosting on these tables. The consequence weights follow JIT defect-prediction guidance: **bug-fix intensity** receives the largest share because it most directly encodes corrective maintenance load; **churn** captures instability but confounds feature growth; **SZZ-linked counts** are informative jointly yet noisy alone due to linking errors—hence the smallest coefficient before percentile truncation.

**11. Robustness, interpretation, and validity.** Consequence models tolerate alternate three-/six-/twelve-month horizons and 10–30% quantile cuts with **stable ROC-AUC**, indicating ranking capability is not an artefact of one threshold. Global attribution (SHAP-style tree explanations plus permutation importance) shows **severity models leaning on static Sonar proxies**—consistent with their near-tautological relationship—while **consequence models lean on history + coupling**, aligning with the theoretical emphasis on process. Basename collisions remain the chief construct threat; grouping projects by collision severity shows **no systematic degradation** of cross-project scores in high-collision strata—consistent with symmetric noise biasing toward the mean rather than optimistic inflation.

**12. Replication.** The thesis companion ships one-command automation with pinned dependencies plus an optional **Google Colab** path that stages the database on fast local disk and mirrors checkpoints—useful if committee members repeat the experiment without cloning physical hardware.

Thank you for your continued guidance.

Respectfully,  
Abdulmajid Awol Seid

<!-- pandoc docs/ADVISOR_STATUS_UPDATE.md -o docs/ADVISOR_STATUS_UPDATE.docx -->
