# Chapter 7 - Discussion (scaffold)

## 7.1 RQ1: Do the three label variants identify different files?

**Answer: Yes, and the disagreement is large.** Pairwise Cohen's
kappa between the three variants ranges from 0.05 (severity vs
SZZ) to 0.21 (consequence vs severity). Jaccard similarity is
at most 0.18. This is the first empirical demonstration on the
Technical Debt Dataset v2.0 that the choice of operational
definition for "high-risk TD" fundamentally changes which files
are prioritised. See Table 6.5 and Figure
``fig_label_agreement_venn``.

## 7.2 RQ2: How accurately can each variant be predicted?

**Within-project** (stratified 10-fold CV, best model):

- Consequence: F1 = 0.581,
  CE@20 = 0.702
  (lightgbm)
- Severity is near-ceiling (F1 > 0.70) and SZZ is hardest (F1
  around 0.27). The order is consistent with the intrinsic
  difficulty of each label.

**Cross-project** (LOPO on 22 projects):

- Consequence: F1 = 0.418,
  CE@20 = 0.484
  (lightgbm). A top-20-percent inspection
  budget in an unseen project captures roughly 50 percent of
  files that will cause real maintenance burden in the next 6
  months - practically usable.

## 7.3 RQ3: Which feature families drive each variant?

From the ablation (Table 6.7):

- **Severity** is predicted best from *static SonarQube* features
  alone - confirming the "tautology" interpretation of severity
  labels.
- **Consequence** and **SZZ** are predicted best from
  *historical (process)* features - matching Kamei et al. (2013)
  for JIT defect prediction.
- Project-level context alone is weak on every variant (F1 < 0.32).

## 7.4 Implications

1. Research on TD prediction that uses severity labels primarily
   benchmarks **SonarQube consistency**, not future impact.
2. If the goal is prioritising maintenance effort, the
   **consequence framing** is a defensible alternative that
   captures different information (kappa < 0.25 against both
   baselines).
3. A 22-project Apache corpus with 6-month windows is sufficient
   for cross-project generalisation (Herbold 2018 protocol).

## 7.5 Threats to validity

- **Construct**: basename aggregation (documented in Research
  Log 2026-04-23); median basename-collision rate 37 percent.
- **Internal**: 20-percent percentile threshold for consequence
  positives may be sensitive to project-level positive-rate
  drift - partially addressed in Section 6.6 sensitivity grid.
- **External**: all projects are Apache Java - findings may not
  transfer to proprietary or non-Java codebases.
- **Conclusion**: stratified K-fold allows same-project
  contamination, inflating within-project numbers; the LOPO
  numbers in Section 6.5 should be taken as the realistic
  deployment estimate.
