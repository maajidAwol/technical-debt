# 03. High-Risk Technical Debt Labeling

*Paper-ready source for Methodology Chapter 3.4.*

## Temporal Protocol

For each project, a snapshot time `t` is chosen. Features are computed from
`AUTHOR_DATE <= t` data only; labels are derived from the observation window
`t < AUTHOR_DATE <= t + W` where `W` is the observation window (6 months
primary).

This strict separation prevents label leakage: the model is trained to
predict *future* maintenance risk using *only* information available at
decision time `t`.

## Variant 1 - Consequence-Oriented (PRIMARY)

High-Risk TD = files in the top `HIGH_RISK_PERCENTILE` of the within-project
maintenance-risk score.

### Risk Score Components

Each component is computed in the post-snapshot observation window:

| Component | Formula | Weight |
|-----------|---------|--------|
| `bugfix_commits_future` | Count of commits touching file matching bug-fix keyword OR Jira-linked Bug | 0.5 |
| `future_churn` | Sum of `LINES_ADDED + LINES_REMOVED` | 0.3 |
| `szz_defects_future` | Count of fault-inducing commits touching the file | 0.2 |

### Normalization

Each component is min-max normalized **within project** to [0, 1]:
`x_norm = (x - min_project) / (max_project - min_project)`

### Score and Threshold

`score = 0.5 * bugfix_commits_norm + 0.3 * future_churn_norm + 0.2 * szz_defects_norm`

Label = 1 if `score` >= within-project 80th percentile; 0 otherwise.

## Variant 2 - Severity Baseline

Label = 1 if the file has at least one SonarQube issue with severity
in {`BLOCKER`, `CRITICAL`} as of the snapshot.

Severity-count features are excluded from the feature vector for this
variant to prevent trivial label leakage.

## Variant 3 - SZZ Baseline

Label = 1 if the file was modified by a commit that appears in
`SZZ_FAULT_INDUCING_COMMITS` within the observation window.

## Sensitivity Analysis

- Percentile thresholds: {10%, 20%, 30%} (variant 1)
- Observation windows: {3, 6, 12} months (variants 1 and 3)

Results reported in `results/tables/sensitivity.csv`.

## Label Agreement Analysis

Cohen's kappa and Jaccard coefficient computed pairwise between
variants at the file level (per project and pooled). Results in
`results/tables/label_agreement.csv`. A Venn diagram
(`results/figures/label_overlap_venn.png`) visualizes overlap.

This label-agreement analysis is one of the primary novelty claims of
this thesis: if the three variants disagree substantially, then prior
work using severity-only labels was measuring something different from
what maintainers actually need.
