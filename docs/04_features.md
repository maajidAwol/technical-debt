# 04. Features

*Paper-ready source for Methodology Chapter 3.3 and Table 1.*

All features are computed from data with `AUTHOR_DATE <= t` (static features
use the closest `SONAR_MEASURES` row with `SNAPSHOT_DATE <= t`).

## Static Code Metrics (from SONAR_MEASURES)

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `ncloc` | Non-comment lines of code | Basic size; correlates with defect/maintenance risk |
| `complexity` | Cyclomatic complexity | High complexity increases change effort and error-proneness |
| `cognitive_complexity` | Cognitive complexity | Human-perceived complexity |
| `classes` | Number of classes | Structural size indicator |
| `functions` | Number of methods/functions | Design scale indicator |
| `statements` | Number of statements | Code density |
| `duplicated_lines_density` | % duplicated code | Duplication is linked to propagated fixes |
| `coverage` | Test coverage ratio | Lower coverage increases latent defect risk |
| `comment_lines_density` | Comment ratio | Documentation indicator |
| `sqale_index` | TD remediation minutes | Maintainability indicator |
| `sqale_debt_ratio` | TD ratio | Normalized maintainability |
| `file_complexity` | SonarQube's per-file complexity | Alternative complexity view |

## Derived Static Features

| Feature | Formula |
|---------|---------|
| `cyclomatic_density` | complexity / max(ncloc, 1) |
| `has_coverage` | Indicator: 1 if coverage is reported, 0 if imputed |

## Rule Violation Counts (from SONAR_ISSUES)

| Feature | Description |
|---------|-------------|
| `code_smells_nonsevere` | Code smells excluding BLOCKER/CRITICAL |
| `bugs_nonsevere` | Bug issues excluding BLOCKER/CRITICAL |
| `vulnerabilities_nonsevere` | Vulnerability issues excluding BLOCKER/CRITICAL |
| `major_issues` | Count of MAJOR severity issues |
| `minor_issues` | Count of MINOR severity issues |
| `info_issues` | Count of INFO severity issues |

**Leakage protection for severity variant**: `code_smells_nonsevere`,
`bugs_nonsevere`, `vulnerabilities_nonsevere`, `blocker_issues`,
`critical_issues`, and `major_issues` are DROPPED when training on
severity-baseline labels.

## Historical Change Metrics (from GIT_COMMITS + GIT_COMMITS_CHANGES up to t)

| Feature | Description |
|---------|-------------|
| `total_commits_pre` | Commits touching file up to t |
| `total_contributors_pre` | Distinct authors up to t |
| `code_churn_pre` | Total (added + removed) lines up to t |
| `recent_churn_30d_pre` | Churn in 30 days before t |
| `recent_churn_90d_pre` | Churn in 90 days before t |
| `recent_commits_30d_pre` | Commits in 30 days before t |
| `recent_commits_90d_pre` | Commits in 90 days before t |
| `file_age_days_at_snapshot` | Days since first commit, measured at t |
| `days_since_last_change_at_snapshot` | Recency measured at t |
| `ownership_ratio_pre` | Share of changes by dominant contributor |
| `avg_change_size_pre` | Avg lines per commit |
| `add_count_pre` / `modify_count_pre` / `delete_count_pre` | Change-type distribution |

## Feature Transformations

- For tree-based models (RF, XGBoost, LightGBM, DT): no scaling
- For SVM: `StandardScaler` fit on training folds only, with `log1p` on skewed
  counts (`ncloc`, churn, commit counts)
- Missing `coverage` imputed as 0 (and `has_coverage` set to 0)
- Missing `cognitive_complexity` imputed from `complexity` where available

## Correlation / Redundancy

Correlation analysis is performed in Stage 6. Feature pairs with
|Pearson r| > 0.95 are flagged, and one of each highly-correlated pair is
dropped (documented in `results/tables/feature_catalog.csv`).
