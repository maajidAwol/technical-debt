"""
Configuration for the simplified Technical Debt prediction pipeline.

Single dual-signal label, 27 features in 5 families, 4 models
(LR / RF / XGB / LGBM). Threshold optimisation per fold;
LOPO with similarity-weighted training; SHAP on best model.

Sections
--------
1. Paths
2. Dataset
3. Snapshot policy
4. Labeling (dual-signal combined-weight)
5. Feature catalogue (27 features in 5 families)
6. Modeling (4 models)
7. Evaluation
8. Parallelism
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for _d in (PROCESSED_DATA_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR, DOCS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------
TD_DATASET_PATH = RAW_DATA_DIR / "td_V2.db"

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
# 3. Snapshot policy
# ---------------------------------------------------------------------------
# t = median commit date per project. Equal proportional history across
# projects for LOPO comparability (Tsoukalas 2020, Jiang 2024). The
# 6-month surrogate window for weight derivation follows Kamei 2013.
SNAPSHOT_STRATEGY = "median"
OBSERVATION_WINDOW_MONTHS = 6
LABEL_SURROGATE_WINDOW_MONTHS = 6

MIN_PRE_SNAPSHOT_COMMITS = 500
MIN_POST_SNAPSHOT_COMMITS = 50

# ---------------------------------------------------------------------------
# 4. Labeling (dual-signal combined-weight)
# ---------------------------------------------------------------------------
# Canonical bug-fix regex. Used for S4 (label signal),
# bugfix_commits_pre features, and the surrogate.
BUGFIX_REGEX = r"\b(fix|bug|defect|patch|resolve|repair)\b"

# Jira issue-key regex (e.g. "HBASE-1234"), used to link commits -> Jira.
JIRA_ISSUE_KEY_PATTERN = r"\b([A-Z][A-Z0-9_]+)-(\d+)\b"

# Theoretically motivated baseline weights (Kamei TSE 2013 style).
# Compared against empirically derived weights; the pipeline uses
# theoretical if |derived - theoretical| <= 0.05 on every signal, else
# uses derived. See src/data/labeling.py.
THEORETICAL_WEIGHTS = {
    "S1_severity": 0.30,
    "S4_bugfix": 0.25,
    "S2_debt": 0.20,
    "S5_churn": 0.15,
    "S3_smells": 0.05,
    "S6_contributors": 0.05,
}

LABEL_RISK_THRESHOLD = 0.50
LABEL_THRESHOLD_FALLBACKS = (0.45, 0.40)  # if per-project positives < 5
LABEL_MIN_POSITIVES_PER_PROJECT = 5

SEVERITY_LABEL_LEVELS = ("BLOCKER", "CRITICAL")

# ---------------------------------------------------------------------------
# 5. Feature catalogue (27 features in 5 families)
# ---------------------------------------------------------------------------
SIZE_COMPLEXITY_FEATURES = [
    "ncloc",
    "complexity",
    "cognitive_complexity",
    "functions",
    "classes",
]

STATIC_DEBT_FEATURES = [
    "n_code_smells",
    "n_bugs",
    "total_debt_minutes",
    "issue_density",
    "duplicated_lines_density",
]

HISTORICAL_FEATURES = [
    "total_commits_pre",
    "code_churn_pre",
    "recent_churn_90d",
    "commit_frequency_30d",
    "file_age_days",
    "days_since_last_change",
    "contributor_count",
    "ownership_ratio",
]

GRAPH_FEATURES = [
    "cocg_degree",
    "cocg_pagerank",
    "cocg_betweenness",
    "cocg_entropy",
]

PRIOR_DEFECT_FEATURES = [
    "bugfix_commits_pre",
    "bugfix_commits_90d",
    "bug_density_pre",
    "n_jira_bugs_pre",
    "jira_blocker_flag",
]

FEATURE_FAMILIES = {
    "size_complexity": SIZE_COMPLEXITY_FEATURES,
    "static_debt": STATIC_DEBT_FEATURES,
    "historical": HISTORICAL_FEATURES,
    "graph": GRAPH_FEATURES,
    "prior_defect": PRIOR_DEFECT_FEATURES,
}

ALL_FEATURES = (
    SIZE_COMPLEXITY_FEATURES
    + STATIC_DEBT_FEATURES
    + HISTORICAL_FEATURES
    + GRAPH_FEATURES
    + PRIOR_DEFECT_FEATURES
)
assert len(ALL_FEATURES) == 27, f"expected 27 features, got {len(ALL_FEATURES)}"

# Columns log1p-transformed in scripts/06_build_dataset.py
# (heavy-tailed counts and durations).
LOG1P_FEATURES = [
    "ncloc",
    "complexity",
    "total_commits_pre",
    "code_churn_pre",
    "recent_churn_90d",
    "file_age_days",
    "total_debt_minutes",
    "bugfix_commits_pre",
    "n_jira_bugs_pre",
]

# Permutation-importance threshold for post-tuning feature selection.
FEATURE_SELECTION_THRESHOLD = 0.001

# ---------------------------------------------------------------------------
# 6. Modeling (4 models)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

MODELS = {
    "logistic_regression": {
        "class": "LogisticRegression",
        "params": {
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": RANDOM_STATE,
        },
    },
    "random_forest": {
        "class": "RandomForestClassifier",
        "params": {
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
        },
    },
    "xgboost": {
        "class": "XGBClassifier",
        "params": {
            "eval_metric": "aucpr",
            "random_state": RANDOM_STATE,
            "verbosity": 0,
            # scale_pos_weight is set dynamically per fit (n_neg/n_pos).
        },
    },
    "lightgbm": {
        "class": "LGBMClassifier",
        "params": {
            "class_weight": "balanced",
            "verbose": -1,
            "random_state": RANDOM_STATE,
        },
    },
}

MODEL_ORDER = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]

# ---------------------------------------------------------------------------
# 7. Evaluation
# ---------------------------------------------------------------------------
CV_FOLDS = 10
TUNING_TRIALS = 30
TUNING_INNER_CV_FOLDS = 5
TUNING_OBJECTIVE = "pr_auc"

COST_EFFECTIVENESS_AT = 0.20

# Threshold sweep for find_optimal_threshold(y_true, y_prob).
THRESHOLD_SWEEP_MIN = 0.05
THRESHOLD_SWEEP_MAX = 0.80
THRESHOLD_SWEEP_STEP = 0.01
THRESHOLD_DEFAULT = 0.30
THRESHOLD_MIN_POSITIVES = 3

# ---------------------------------------------------------------------------
# 8. Parallelism
# ---------------------------------------------------------------------------
def _resolve_n_jobs(default: int | None = None) -> int:
    env = os.environ.get("TD_N_JOBS")
    if env is not None:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    if default is not None:
        return default
    cpu = os.cpu_count() or 4
    return max(1, cpu // 2)


STAGE5_N_JOBS = _resolve_n_jobs()
