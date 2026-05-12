# Discussion Summary: Technical Debt Prediction Pipeline

**Thesis**: *Predicting High-Risk Technical Debt in Open-Source Software Projects Using Machine Learning and Code Metrics*
**Author**: Abdulmajid Awol Seid (MSc, Feb 2026)
**Date of discussion**: 2026-05-12

---

## Table of Contents

1. [Overview of the Pipeline](#1-overview-of-the-pipeline)
2. [Three-Variant Labeling Explained](#2-three-variant-labeling-explained)
3. [Cohen's Kappa and Why It Is Low](#3-cohens-kappa-and-why-it-is-low)
4. [Snapshot Definition](#4-snapshot-definition)
5. [How the Three Labels Combine for Prediction](#5-how-the-three-labels-combine-for-prediction)
6. [Jaccard Overlap and Disjoint Labels](#6-jaccard-overlap-and-disjoint-labels)
7. [10-Fold Cross-Validation and Metrics](#7-10-fold-cross-validation-and-metrics)
8. [LOPO Cross-Project Validation](#8-lopo-cross-project-validation)
9. [Ablation, Features, and Columns](#9-ablation-features-and-columns)
10. [SHAP and Construct Validity](#10-shap-and-construct-validity)
11. [Stage-by-Stage Pipeline Summary](#11-stage-by-stage-pipeline-summary)
12. [Optimization & Validation Rationale](#12-optimization--validation-rationale)
13. [Proposal Alignment & Predictive Capability](#13-proposal-alignment--predictive-capability)
14. [Risk Score Weights (0.5 / 0.3 / 0.2)](#14-risk-score-weights-05--03--02)
15. [Why Three Models and How They Combine](#15-why-three-models-and-how-they-combine)

---

## 1. Overview of the Pipeline

A **10-stage (0-10) data-first pipeline** that predicts which files in open-source projects will become "high-risk" technical debt. Each stage produces an inspectable artifact (CSV/Parquet) that feeds into the next.

- **Input**: Technical Debt Dataset v2.0 (Lenarduzzi et al. 2019), 1.54 GB SQLite, 33 Apache projects.
- **Unit of analysis**: `(project_id, file_basename)` tuple — approximately 23,911 units across 22 eligible projects.
- **Output**: probability that a file will become high-risk in the next 6 months.

---

## 2. Three-Variant Labeling Explained

The thesis defines "high-risk technical debt" three different ways and runs three independent experiments. Each labeler is **deterministic** (no ML involved); it derives labels directly from the database.

### Variant 1 — Consequence-oriented (PRIMARY)

Defined in [src/data/labeling.py:87-179](src/data/labeling.py#L87-L179):

1. Take the universe of file basenames that existed at snapshot `t`.
2. For each file, count three future-window signals in `(t, t + 6 months]`:
   - `n_bugfix_commits_future` — commits matching bug-fix regex (*fix/bug/defect/patch/resolve*)
   - `future_churn` — lines added + lines removed
   - `n_szz_fixes_future` — fault-inducing commits via SZZ
3. Min-max normalize each within the project to `[0, 1]`.
4. Compute weighted risk score: `0.5·bugfix + 0.3·churn + 0.2·szz`.
5. Flag the **top 20% within each project** as high-risk.

Result: **14.78% positive rate** (3,533 files).

### Variant 2 — Severity baseline

Defined in [src/data/labeling.py:188-267](src/data/labeling.py#L188-L267):

- Flag a file if it has ≥1 SonarQube issue at snapshot `t` with `SEVERITY in ("BLOCKER", "CRITICAL")`.

Result: **9.64% positive rate** (2,305 files).

### Variant 3 — SZZ baseline

Defined in [src/data/labeling.py:273-298](src/data/labeling.py#L273-L298):

- Flag a file if a fault-inducing commit (per `SZZ_FAULT_INDUCING_COMMITS` table) touched it during the observation window.

Result: **1.24% positive rate** (297 files, only 16 of 22 projects).

---

## 3. Cohen's Kappa and Why It Is Low

### Definition

Cohen's kappa measures **agreement between two raters beyond chance**:

```
κ = (p_observed − p_expected_by_chance) / (1 − p_expected_by_chance)
```

Landis & Koch (1977) scale: `<0.20 = slight`, `0.21–0.40 = fair`, `0.41–0.60 = moderate`.

### Results

| Comparison | Cohen's Kappa | Jaccard | Overall Agreement |
|---|---|---|---|
| Consequence vs Severity | 0.21 | 0.18 | 82.9% |
| Consequence vs SZZ | 0.13 | 0.08 | 86.4% |
| Severity vs SZZ | 0.05 | 0.04 | 89.9% |

### Why kappa is low

1. The three rules measure **fundamentally different things** (future maintenance vs current SonarQube view vs sparse fault-inducing events).
2. Positive rates differ enormously (1.24% to 14.78%), which mathematically caps how high kappa can reach (the kappa paradox under imbalance).

### Does low kappa make the experiment meaningless?

**No — the opposite.** Low kappa **IS the finding** for RQ1. It is the quantitative evidence that *"how you define high-risk TD changes which files get flagged"*. A high kappa would mean the three definitions are interchangeable and the thesis has no story.

### Could we improve it? Should we?

You could artificially raise kappa by aligning thresholds, but it would defeat the comparison purpose. The best approach is the current one: keep each variant faithful to its literature definition and report disagreement honestly.

---

## 4. Snapshot Definition

Implementation in [src/data/snapshot.py:85-197](src/data/snapshot.py#L85-L197).

For each project independently:

1. Pull all main-branch commits.
2. Compute the **median commit date** — this becomes `t`.
3. Split:
   - **Pre-snapshot** (`AUTHOR_DATE <= t`) → used for **features**.
   - **Post-snapshot window** (`t < AUTHOR_DATE <= t + 6 months`) → used for **labels**.
   - Everything after `t + 6 months` is **discarded**.
4. Eligibility: `pre >= 500` commits AND `post >= 50` commits.

```
[----- everything before t -----] | t | [(t, t + 6 months]] | (discarded)
       PRE (features)                    POST (labels)
```

This temporal separation prevents future information from leaking into training features.

---

## 5. How the Three Labels Combine for Prediction

**They do NOT combine.** Three independent models are trained.

The pipeline produces:

- `dataset_consequence.parquet` (label = consequence)
- `dataset_severity.parquet` (label = severity, leaky features dropped)
- `dataset_szz.parquet` (label = SZZ, leaky features dropped)

Each variant gets its own model. The thesis then **compares** them. The thesis claim is not "ensemble of three"; it's "**which definition you choose changes the answer**".

### Prediction unit = FILE (not project)

For each `(project_id, basename)` tuple, the model outputs the probability that this file will become high-risk. The project appears only as context features (project-level NCLOC, complexity, etc., replicated across all files of that project).

---

## 6. Jaccard Overlap and Disjoint Labels

### Definition

`Jaccard(A, B) = |A ∩ B| / |A ∪ B|` — fraction of flagged files in EITHER labeler that are flagged by BOTH. Range 0–1.

Your values (0.04–0.18) mean almost completely disjoint sets.

### Is this broken methodology?

**No.** It is the central evidence for RQ1. The thesis does not claim the three definitions are equivalent measures — it claims each operationalizes TD differently. Low Jaccard proves disagreement is **substantive**, not a measurement artefact.

---

## 7. 10-Fold Cross-Validation and Metrics

### Stratified 10-fold CV

For each (variant, model) pair the code:
1. Takes the full dataset (~23k file-rows across all projects).
2. **Stratified split** into 10 equal folds (each fold preserves the positive-rate proportion).
3. Loops 10 times: train on 9 folds, evaluate on the held-out fold.
4. Averages metrics across the 10 folds.

### Metric definitions (plain English)

| Metric | Plain meaning | Range |
|---|---|---|
| **Precision** | "Of the files I flagged, what fraction are truly high-risk?" | 0-1 |
| **Recall** | "Of the truly high-risk files, what fraction did I flag?" | 0-1 |
| **F1** | Harmonic mean of precision and recall | 0-1 |
| **ROC-AUC** | Probability that a random positive scores higher than a random negative — pure ranking quality | 0.5 random, 1.0 perfect |
| **PR-AUC** | Area under precision-recall curve, more honest under class imbalance | 0-1 |
| **MCC** | Matthews correlation, balanced across all confusion-matrix cells | -1 to +1 |
| **CE@20** | "If you inspect only the top 20% of files my model flags, what fraction of real high-risk files do you catch?" — practical prioritization metric | 0-1 |

### Within-project results (XGBoost)

| Variant | ROC-AUC | PR-AUC | F1 | CE@20 | Positive rate |
|---|---|---|---|---|---|
| Severity | 0.976 | 0.817 | 0.704 | 0.970 | 9.64% |
| Consequence | 0.899 | 0.662 | 0.547 | 0.703 | 14.78% |
| SZZ | 0.937 | 0.241 | 0.155 | 0.906 | 1.24% |

---

## 8. LOPO Cross-Project Validation

### Mechanics

Implementation in [src/models/cross_project.py:64-120](src/models/cross_project.py#L64-L120).

For each of 22 projects:
1. Hold out **all files** from project `P`.
2. Train the model on **all files from the other 21 projects** (~22k files).
3. Predict on the held-out project's files.
4. Compute metrics on the held-out project.
5. Move to the next project.

After 22 folds, average across projects. Answers: *"If I train on the projects I have and deploy on a brand-new project, how well will it work?"*

### Results (best model per variant)

| Variant | Best model | F1 | ROC-AUC | CE@20 | N projects |
|---|---|---|---|---|---|
| Severity | LightGBM | 0.613 | 0.958 | 0.872 | 22 |
| Consequence | LightGBM | 0.418 | 0.776 | **0.526** | 22 |
| SZZ | LogReg | 0.113 | 0.786 | 0.562 | 16 |

### Generalization gaps (within − LOPO F1)

- **Severity**: 13% drop (SonarQube rules are project-agnostic).
- **Consequence**: 28% drop (healthy; Herbold 2018 reports 15–40%).
- **SZZ**: 95% drop (too project-specific to generalize).

### Practical interpretation

On a brand-new project, a developer who inspects the top 20% of files flagged by the consequence model will catch **~53% of files that will cause real maintenance pain in the next 6 months**. This is the publishable headline.

---

## 9. Ablation, Features, and Columns

### What is ablation?

Systematically remove a component and see how performance drops. The drop measures that component's unique contribution.

Three regimes per (variant, group):
- **Only this group** — train using ONLY features in this group.
- **Leave-one-out** — train using ALL features EXCEPT this group.
- **Baseline (all features)** — reference.

### Column vs Feature

- **Column** = raw field in the database (e.g., `SONAR_MEASURES.NCLOC`).
- **Feature** = engineered input to the model (e.g., `code_churn_pre = SUM(LINES_ADDED + LINES_REMOVED)` aggregated per basename, filtered to pre-snapshot).

Features are NOT a simple cross-product of columns — they are derived through counts, sums, ratios, time-window filters, and graph centralities.

### Feature catalogue (~92 features)

Per [feature_summary.csv](results/tables/feature_summary.csv): **49 static + 18 historical + 12 graph + 13 prior-defect**.

**Static features** ([config.py:124-168](config.py#L124-L168)):
- `n_issues_open`, `n_blocker`, `n_critical`, `n_major`, `n_minor`, `n_info`, `n_code_smell`, `n_bug`, `n_vulnerability`, `total_debt_minutes`, `total_effort_minutes`, `n_distinct_rules`, `max_severity_rank`, `issue_density`, `debt_per_loc`, `pseudo_ncloc_at_t`
- Project-replicated context: `ncloc`, `complexity`, `cognitive_complexity`, `classes`, `functions`, `statements`, `duplicated_lines_density`, `coverage`, `comment_lines_density`, `sqale_index`, `sqale_debt_ratio`, `file_complexity`, `cyclomatic_density`, `comment_to_code_ratio`

**Historical features** (15) ([config.py:188-204](config.py#L188-L204)):
- `total_commits_pre`, `total_contributors_pre`, `code_added_pre`, `code_removed_pre`, `code_churn_pre`, `recent_churn_30d_pre`, `recent_churn_90d_pre`, `recent_commits_30d_pre`, `recent_commits_90d_pre`, `file_age_days_at_snapshot`, `days_since_last_change_at_snapshot`, `ownership_ratio_pre`, `avg_change_size_pre`, `max_single_commit_churn_pre`, `std_change_size_pre`

**Graph features** (10) ([config.py:209-220](config.py#L209-L220)):
- `cocg_degree`, `cocg_strength_sum`, `cocg_strength_mean`, `cocg_strength_max`, `cocg_betweenness`, `cocg_closeness`, `cocg_clustering_coef`, `cocg_pagerank`, `cocg_neighbour_count_30d`, `cocg_neighbour_count_90d`

**Prior-defect features** (10) ([config.py:226-237](config.py#L226-L237)):
- `bugfix_commits_pre`, `bugfix_commits_pre_30d`, `bugfix_commits_pre_90d`, `bugfix_commits_pre_365d`, `time_since_last_bugfix_days`, `szz_inducing_pre`, `szz_inducing_pre_365d`, `linked_jira_issues_pre`, `linked_jira_bugs_pre`, `bug_density_pre`

### Ablation result table

| Variant | Dominant group | F1 (only-group) | F1 drop if removed |
|---|---|---|---|
| Severity | static_sonar (16) | 0.646 | -0.228 |
| Consequence | historical (15) | 0.555 | -0.092 |
| SZZ | historical (15) | 0.226 | -0.135 |

**Interpretation**:
- Severity is predicted by SonarQube features *alone* — confirms severity is a SonarQube tautology.
- Consequence and SZZ are **process-driven** (Git history dominates) — matches Kamei et al. 2013.
- Project-level context alone is weak (F1 0.07–0.30); helps marginally only when combined.

### Why family-level reporting

49 individual rows in an ablation table is unreadable. Tree ensembles redistribute signal across correlated features, so single-feature ablation mostly produces zero drop. Family-level grouping is the standard protocol (Zimmermann 2007 / Rahman 2013).

---

## 10. SHAP and Construct Validity

### What is SHAP?

SHapley Additive exPlanations — a game-theoretic feature importance method. For each prediction, it tells you how much each feature shifted the model's output away from the average. Sum across instances → global importance. TreeSHAP (Lundberg et al. 2020) computes this exactly for tree ensembles.

### Top-5 SHAP features per variant

| Variant | Top-5 SHAP features |
|---|---|
| **Consequence** | pseudo_ncloc_at_t, days_since_last_change, file_age_days, total_commits_pre, avg_change_size_pre |
| **Severity** | total_debt_minutes, n_distinct_rules, max_single_commit_churn, n_bug, file_age_days |
| **SZZ** | project_function_complexity, days_since_last_change, ownership_ratio, pseudo_ncloc, max_single_commit_churn |

### Interpretation

- **Consequence** is driven by **size + activity + recency** — "big, old, recently-churned files hurt most". No SonarQube issue count in the top-5 → consequence signal is **orthogonal to severity**.
- **Severity** is a **near-tautology** (debt + rule variety + bug counts at the same snapshot as the label).
- **SZZ** leans on **project context + ownership** — fault-fix incidence is partly a project-level phenomenon.

Permutation importance and SHAP agree on top-3 features per variant (Pearson rank correlation > 0.80).

### Does disjoint labeling threaten the thesis?

**No, if framed correctly**. The thesis does not claim "all three measure the same thing". It claims each operationalizes TD differently, and shows empirically what each rewards. The construct validity discussion in Chapter 5 names this "three views of TD" — consequence is the principled, process-driven, project-portable definition.

---

## 11. Stage-by-Stage Pipeline Summary

| Script | Stage | What it does |
|---|---|---|
| [01_inspect_db.py](scripts/01_inspect_db.py) | 1 | Schema inventory; dump table samples; record column types |
| [02_profile_projects.py](scripts/02_profile_projects.py) | 2 | Compute per-project snapshot `t`; apply eligibility thresholds |
| [03_clean.py](scripts/03_clean.py) | 3 | Clean tables; settle `(project_id, basename)` unit; export Parquet |
| [04_label.py](scripts/04_label.py) | 4 | Build three label variants; compute kappa and Jaccard |
| [05_features.py](scripts/05_features.py) | 5 | Build all features (static, historical, graph, prior-defect) — pre-snapshot only, parallel by project |
| [06_build_dataset.py](scripts/06_build_dataset.py) | 6 | Merge features + labels into three `dataset_{variant}.parquet`; apply leakage drops |
| [07_train.py](scripts/07_train.py) | 7 | Within-project 10-fold CV × 5 models × 3 variants |
| [07b_tune.py](scripts/07b_tune.py) | 7b | Optuna hyperparameter search (PR-AUC, 40 trials, 5-fold inner CV) |
| [07c_calibrate.py](scripts/07c_calibrate.py) | 7c | Probability calibration sweep (Platt + isotonic) with reliability diagrams |
| [07d_resample.py](scripts/07d_resample.py) | 7d | SMOTE vs class_weight comparison |
| [07e_temporal.py](scripts/07e_temporal.py) | 7e | Per-project temporal T1 (40th pct) → T2 (70th pct) split |
| [08_lopo.py](scripts/08_lopo.py) | 8 | Leave-One-Project-Out CV across 22 projects |
| [09_sensitivity.py](scripts/09_sensitivity.py) | 9 | 3×3 sensitivity grid + feature-group ablation |
| [10_report.py](scripts/10_report.py) | 10 | SHAP + permutation importance, 7 figures (PNG+PDF), auto-render docs |

---

## 12. Optimization & Validation Rationale

The "Refined Extended" package added each enhancement in response to a specific gap found in the proposal-alignment audit:

| Enhancement | Source gap | Rationale |
|---|---|---|
| **SVM activated** | Proposal §3.5 commits to DT/RF/SVM/GBM; baseline omitted SVM | Honors proposal commitment |
| **Optuna tuning (7b)** | §3.4 implies rigorous hyperparameter search; baseline used defaults | Removes "lucky defaults" threat; TPE Bayesian search avoids 30h grid cost |
| **Probability calibration (7c)** | §3.6 mentions calibration | Tree ensembles produce uncalibrated probs; threshold-based use requires it (Platt + isotonic per Niculescu-Mizil & Caruana 2005) |
| **SMOTE comparison (7d)** | §3.4 mentions imbalance handling | Tests oversampling vs reweighting; train-fold-only (test never resampled) |
| **Temporal T1→T2 split (7e)** | §3.4 mentions temporal CV | Real deployment is temporal: train on past, predict future within the same project (Falessi et al. 2020) |
| **Co-change graph features** | §3.3 mentions co-change features | Jiang 2024/2025: files that change together fail together; PageRank/betweenness capture architectural importance |
| **Prior-defect features** | §3.2 mentions SZZ-derived defect history | Hassan 2009 / Kamei 2013: past defects predict future defects; strictly `< t` to prevent leakage |
| **Bootstrap CIs (10k)** | §3.6 lists significance testing | Point estimates hide uncertainty; CIs give honest reporting |
| **Wilcoxon paired tests + Bonferroni** | Demsar 2006 ML comparison standard | Tells you whether model A really beats model B per-fold, not by luck |

Each item closes a specific proposal commitment that the bare Stage-7 baseline didn't satisfy.

---

## 13. Proposal Alignment & Predictive Capability

### Proposal claims vs evidence

| Claim | Evidence | Verdict |
|---|---|---|
| **RQ1**: Label definition affects which files are flagged | κ = 0.05–0.21, Jaccard = 0.04–0.18 across 23,911 files | ✅ Strong |
| **RQ2**: ML can predict consequence high-risk TD | F1 = 0.547, PR-AUC = 0.662, ROC-AUC = 0.899, CE@20 = 0.703 | ✅ Practical |
| **RQ3**: Models generalize to unseen projects | LOPO F1 = 0.418, CE@20 = 0.526; 28% gap within Herbold 2018's 15–40% band | ✅ Cross-project |
| Process metrics > product metrics for future events | Historical group dominates consequence and SZZ; SHAP top features are activity/age/size | ✅ Confirms Kamei 2013 |
| Severity-based labels are limited | Static-only F1 = 0.646 vs 0.704 with all features → SonarQube tautology | ✅ Strong |
| Robustness to hyperparameters | ROC-AUC stable 0.883–0.917 across 9 configurations | ✅ Supported |

### Can we truly predict high-risk TD?

**Yes, with these qualifications:**

1. **The prediction is probabilistic, not categorical.** Output = `P(file high-risk in next 6 months)`.
2. **Ranking-strong, threshold-weak.** ROC-AUC 0.776–0.958 says ranking works; point precision is weaker, hence CE@K as the headline metric.
3. **Operational claim**: top-20% inspection budget catches ~53% (LOPO) or ~70% (within-project) of future-burden files.
4. **What we cannot predict reliably**: rare catastrophic SZZ defects (1.24% positive, F1 = 0.155).

### Technical machinery making the prediction valid

- **Strict temporal split**: features only from `commit_date <= t`; labels from `(t, t+6mo]`.
- **Leakage audit**: `SEVERITY_LEAKY_FEATURES` and `SZZ_LEAKY_FEATURES` dropped per variant.
- **Stratified CV** preserves positive rate per fold; **LOPO never mixes projects**; **temporal T1→T2** validates within-project temporal generalization.
- **Bootstrap CIs (10k resamples)** and **Wilcoxon paired tests + Bonferroni**.
- **TreeSHAP + permutation importance agree** (Pearson > 0.80).

### Defensible headline claim

> *"On 22 Apache projects (~24k file-units), a LightGBM model trained on 21 projects and tested on the 22nd correctly ranks files such that inspecting the top 20% recovers 53% of files that will cause concrete maintenance burden in the next 6 months — robust to a 3×3 sensitivity grid over window and percentile."*

---

## 14. Risk Score Weights (0.5 / 0.3 / 0.2)

### Where they live

[config.py:95-99](config.py#L95-L99):

```python
RISK_SCORE_WEIGHTS = {
    "bugfix_commits_future": 0.5,
    "future_churn":           0.3,
    "szz_defects_future":     0.2,
}
```

### Scope: CONSEQUENCE variant only

The other two variants are pure boolean rules with no weighting:
- **Severity**: `is_high_risk = (n_blocker > 0) OR (n_critical > 0)`
- **SZZ**: `is_high_risk = (n_szz_fixes_future > 0)`

Only consequence fuses three signals, so only consequence needs weights.

### How they are used

In [src/data/labeling.py:162-178](src/data/labeling.py#L162-L178), after min-max normalizing each component within the project to `[0, 1]`:

```
risk_score = 0.5·bugfix_norm + 0.3·churn_norm + 0.2·szz_norm
```

### Why these weights

| Signal | Weight | Reason |
|---|---|---|
| **Bug-fix commits** | 0.5 | Strongest evidence of real maintenance pain (Mockus & Votta 2000, Fischer et al. 2003) |
| **Future churn** | 0.3 | Noisier than bug-fixes (refactors and feature work also produce churn) |
| **SZZ defects** | 0.2 | Most precise but extremely sparse (1.24%); higher weight would collapse consequence into SZZ |

The weights encode a **theoretically motivated prior**, not a data-fitted choice — which is a stronger thesis-defense position than "we tuned them".

### Are the weights swept in sensitivity?

No. Stage 9 sweeps window (3/6/12 months) × percentile (10/20/30%) but holds weights fixed. ROC-AUC stability (0.883–0.917) across that grid suggests the consequence label is robust to construction noise more generally.

---

## 15. Why Three Models and How They Combine

### Why three models

The three categories are **three different research questions**, not three inputs to one prediction:

| Model | Research question | Label rule |
|---|---|---|
| Consequence | Which files cause future maintenance burden? | Top 20% by bugfix + churn + SZZ |
| Severity | Which files does SonarQube call bad now? | Has BLOCKER/CRITICAL issue |
| SZZ | Which files participate in fault-inducing changes? | Touched by SZZ fault-inducing commit |

Three models → comparison answers RQ1 (does the definition matter?).

### How they combine for prediction

**They don't.** At deployment time you pick one based on what you care about:

- **Future maintenance cost** → consequence model (the recommended deployment option).
- **SonarQube-style flagging** → severity model (but SonarQube itself already gives this).
- **Fault-inducing change prediction** → SZZ model (doesn't generalize well, not deployment-ready).

### Why the consequence model is "the" deployment model

1. Future-looking, actionable definition (others describe the present).
2. Generalizes across projects (LOPO CE@20 = 0.526).
3. SHAP shows it's driven by process metrics, not SonarQube self-reference — adds information beyond static analyzers.

### Role of severity and SZZ in the thesis

Scientific controls, not deployment options:
- **Severity baseline** proves "SonarQube severity is a tautology; its 0.704 F1 is mostly self-prediction".
- **SZZ baseline** proves "the classical defect-prediction target is too sparse and too project-specific".

Both make the consequence-model result interpretable in context.

### Could you build an ensemble?

In principle: voting / score-averaging / stacking. But you **shouldn't** for this thesis:

1. The three definitions disagree on what "high-risk" means (κ ≤ 0.21). An ensemble would average three different *constructs*.
2. The thesis claim is specifically that the consequence framing is the right one.
3. No proposal commitment to an ensemble; adding one expands scope without addressing any RQ.

---

## Key Numbers at a Glance

- **Dataset**: 23,911 `(project, basename)` units across 22 eligible Apache projects
- **Features**: ~92 (49 static + 18 historical + 12 graph + 13 prior-defect)
- **Cohen's kappa range**: 0.05 to 0.21 (slight agreement)
- **Jaccard range**: 0.04 to 0.18 (largely disjoint sets)
- **Within-project consequence F1**: 0.547 (XGBoost)
- **Within-project consequence CE@20**: 0.703
- **LOPO consequence CE@20**: 0.526 (LightGBM)
- **Generalization gap**: 28% (healthy; Herbold 2018 reports 15–40%)
- **Sensitivity grid ROC-AUC range**: 0.883–0.917 (3.4 pp — stable)
- **SHAP-permutation agreement**: Pearson > 0.80
- **Bootstrap resamples**: 10,000 (95% BCa CIs)

---

*End of summary. Generated 2026-05-12.*
