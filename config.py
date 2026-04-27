"""
Configuration for the Technical Debt Prediction Research pipeline.

Aligned with the approved MSc research proposal (Updated, Feb 2026) by
Abdulmajid Awol Seid. This module centralizes paths, temporal-split
parameters, three-variant labeling settings, feature catalogues, and
machine-learning model specifications.

Sections
--------
1. Paths
2. Dataset
3. Temporal split and snapshot policy
4. Labeling (three variants + sensitivity grid)
5. Feature catalogues (Proposal Table 1)
6. Modeling (models, hyperparameter grids, imbalance handling)
7. Evaluation (metrics, CV folds, randomness)
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

DOCS_DIR = PROJECT_ROOT / "docs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Ensure the output subtrees exist at import time (idempotent).
for _d in (PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, FIGURES_DIR, TABLES_DIR, DOCS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------
# Technical Debt Dataset v2.0 (Lenarduzzi et al. 2019).
# Download: https://github.com/clowee/The-Technical-Debt-Dataset/releases
TD_DATASET_PATH = RAW_DATA_DIR / "td_V2.db"

# Java source files only; exclude tests, generated code, build artefacts.
SOURCE_FILE_EXTENSIONS = (".java",)
PATH_EXCLUSION_PATTERNS = (
    "/test/",
    "/tests/",
    "/generated/",
    "/generated-sources/",
    "/target/",
    "/build/",
)

# ---------------------------------------------------------------------------
# 3. Temporal split and snapshot policy
# ---------------------------------------------------------------------------
# Snapshot selection strategy per project. "median" is the primary choice:
# the median commit date of the project's master-branch history. Alternative
# strategies can be activated by overriding this in a script if needed.
SNAPSHOT_STRATEGY = "median"  # one of: "median", "release", "fixed"

# Primary observation window (months after snapshot) used to derive labels.
OBSERVATION_WINDOW_MONTHS = 6

# Sensitivity analysis windows (months) to assess robustness of labeling.
SENSITIVITY_WINDOWS = (3, 6, 12)

# Minimum history requirements for a project to be usable.
# `min_post=50` chosen to maximize LOPO fold count while preserving label
# signal quality (expected top-20% positives per project: 20-50 files).
# Rationale documented in RESEARCH_LOG.md, 2026-04-23 entry.
MIN_PRE_SNAPSHOT_COMMITS = 500
MIN_POST_SNAPSHOT_COMMITS = 50

# ---------------------------------------------------------------------------
# 4. Labeling (three variants + sensitivity)
# ---------------------------------------------------------------------------
# Primary labeling: consequence-oriented. High-Risk TD = top P% by maintenance
# risk score computed over the post-snapshot observation window.
HIGH_RISK_PERCENTILE = 20  # top 20%

# Sensitivity analysis thresholds.
SENSITIVITY_PERCENTILES = (10, 20, 30)

# Weights for the consequence risk score components. See Proposal Section 3.4.
# All components are min-max normalized within project before weighting.
RISK_SCORE_WEIGHTS = {
    "bugfix_commits_future": 0.5,
    "future_churn": 0.3,
    "szz_defects_future": 0.2,
}

# Bug-fix keyword regex (case-insensitive). Used on commit messages.
# Based on Mockus & Votta 2000 and Fischer et al. 2003.
BUG_FIX_KEYWORDS = (
    r"\bfix(?:es|ed|ing)?\b",
    r"\bbug(?:s|fix|fixes)?\b",
    r"\bdefect(?:s)?\b",
    r"\berror(?:s)?\b",
    r"\bpatch(?:es|ed)?\b",
    r"\bresolve(?:d|s)?\b",
    r"\bissue\s*#?\d+",
    r"\bclose(?:s|d)?\s*#?\d+",
)

# Jira issue-key regex to extract links from commit messages (e.g. "HBASE-1234").
JIRA_ISSUE_KEY_PATTERN = r"\b([A-Z][A-Z0-9_]+)-(\d+)\b"

# Severity levels considered "high-risk" for the severity baseline variant.
SEVERITY_BASELINE_LEVELS = ("BLOCKER", "CRITICAL")

# ---------------------------------------------------------------------------
# 5. Feature catalogues (Proposal Table 1)
# ---------------------------------------------------------------------------
# Static code metrics extracted from SONAR_MEASURES at snapshot time.
STATIC_METRICS = [
    "ncloc",
    "complexity",
    "cognitive_complexity",
    "classes",
    "functions",
    "statements",
    "duplicated_lines_density",
    "coverage",
    "comment_lines_density",
    "sqale_index",
    "sqale_debt_ratio",
    "file_complexity",
]

# Derived static features computed from the raw metrics above.
STATIC_DERIVED_FEATURES = [
    "cyclomatic_density",            # complexity / ncloc
    "comment_to_code_ratio",         # comment_lines_density-based
]

# Rule-violation count features from SONAR_ISSUES (at snapshot time).
# These are used as predictors for the consequence and SZZ variants, and
# EXCLUDED for the severity variant to avoid leakage.
ISSUE_COUNT_FEATURES = [
    "code_smells_nonsevere",
    "bugs_nonsevere",
    "vulnerabilities_nonsevere",
    "major_issues",
    "minor_issues",
    "info_issues",
]

# Features to DROP when training on severity-baseline labels to prevent
# label leakage. The severity label is ``(n_blocker + n_critical) > 0``
# (per ``SEVERITY_BASELINE_LEVELS``); any feature that carries that signal
# at inference time must be removed.
SEVERITY_LEAKY_FEATURES = [
    "n_blocker",
    "n_critical",
    "n_major",
    "n_minor",
    "n_info",
    "max_severity_rank",
]

# Features to DROP when training on SZZ-baseline labels. The SZZ label is
# ``(n_szz_fixes_future > 0)``; ``n_szz_inducing_past`` carries strongly
# correlated signal (files previously caught by SZZ are usually caught
# again), so we drop it to keep SZZ predictions non-trivial.
SZZ_LEAKY_FEATURES: list[str] = []  # features module currently carries no SZZ features

# Historical metrics from GIT_COMMITS and GIT_COMMITS_CHANGES up to snapshot t.
# Note: TD Dataset v2.0 does not expose a change-type column, so ADD / MODIFY /
# DELETE counts are replaced with distributional churn statistics (max, std)
# that capture volatility without requiring change-type annotations.
HISTORICAL_METRICS = [
    "total_commits_pre",
    "total_contributors_pre",
    "code_added_pre",
    "code_removed_pre",
    "code_churn_pre",
    "recent_churn_30d_pre",
    "recent_churn_90d_pre",
    "recent_commits_30d_pre",
    "recent_commits_90d_pre",
    "file_age_days_at_snapshot",
    "days_since_last_change_at_snapshot",
    "ownership_ratio_pre",
    "avg_change_size_pre",
    "max_single_commit_churn_pre",
    "std_change_size_pre",
]

# ---------------------------------------------------------------------------
# 6. Modeling
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

MODELS = {
    "decision_tree": {
        "class": "DecisionTreeClassifier",
        "params": {"random_state": RANDOM_STATE, "class_weight": "balanced"},
    },
    "random_forest": {
        "class": "RandomForestClassifier",
        "params": {
            "n_estimators": 200,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
            "n_jobs": -1,
        },
    },
    "svm": {
        "class": "SVC",
        "params": {
            "kernel": "rbf",
            "probability": True,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        },
    },
    "xgboost": {
        "class": "XGBClassifier",
        "params": {
            "n_estimators": 200,
            "random_state": RANDOM_STATE,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "n_jobs": -1,
        },
    },
    "lightgbm": {
        "class": "LGBMClassifier",
        "params": {
            "n_estimators": 200,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
            "n_jobs": -1,
            "verbose": -1,
        },
    },
}

PARAM_GRIDS = {
    "xgboost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.1, 0.2],
        "scale_pos_weight": [1, 3, 5, 10],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "lightgbm": {
        "n_estimators": [100, 200, 300],
        "max_depth": [-1, 5, 10, 20],
        "learning_rate": [0.01, 0.1, 0.2],
        "num_leaves": [15, 31, 63],
    },
}

# ---------------------------------------------------------------------------
# 7. Evaluation
# ---------------------------------------------------------------------------
CV_FOLDS = 10                       # within-project stratified K-fold
COST_EFFECTIVENESS_AT = 0.20        # CE @ top-20% (prioritization metric)

# Variant keys used across pipeline scripts.
LABEL_VARIANTS = ("consequence", "severity", "szz")
