"""Generator for td_pipeline_colab.ipynb.

Reads source files from src/ and scripts/, strips the `sys.path.append` /
`from config import ...` / `from src... import ...` lines (we hoist
constants and submodules into Section 0), and emits an nbformat v4
notebook whose cells run the pipeline end-to-end in Colab.

Run from the repo root:
    python notebooks/_build_notebook.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "td_pipeline_colab.ipynb"


# ---------------------------------------------------------------------------
# Helpers for inlining
# ---------------------------------------------------------------------------
_STRIP_PATTERNS = [
    re.compile(r"^from __future__ import annotations\s*$"),
    re.compile(r"^sys\.path\.append\(.*\)\s*$"),
    re.compile(r"^from config import .*$"),
    re.compile(r"^from src\.[A-Za-z0-9_.]+ import .*$"),
]


def _is_paren_open_no_close(stripped_no_comment: str) -> bool:
    """True if the line opens a paren that doesn't close on the same line."""
    return "(" in stripped_no_comment and ")" not in stripped_no_comment


def read_source(rel: str, *, drop_main_guard: bool = True, rename_main: str | None = None) -> str:
    """Return the file's source minus disallowed lines.

    If ``rename_main`` is given, every occurrence of ``main`` defined or
    called inside the script is renamed (so two scripts inlined in the same
    notebook don't shadow each other).
    """
    path = REPO / rel
    raw = path.read_text(encoding="utf-8").splitlines()
    out = []
    skip_until_close_paren = False
    for line in raw:
        if skip_until_close_paren:
            # Drop the closing-paren line too. A real ')' is what we want;
            # anything containing it ends the import block.
            if ")" in line:
                skip_until_close_paren = False
            continue
        if any(p.match(line) for p in _STRIP_PATTERNS):
            # Strip trailing comment before deciding if this opens a multi-line import.
            no_comment = re.sub(r"#.*$", "", line).rstrip()
            if _is_paren_open_no_close(no_comment):
                skip_until_close_paren = True
            continue
        out.append(line)
    src = "\n".join(out)
    if drop_main_guard:
        src = re.sub(
            r"\n\s*if __name__ == [\"']__main__[\"']:\s*\n\s*main\(\)\s*\n?",
            "\n",
            src,
        )
    if rename_main is not None:
        src = re.sub(r"\bdef main\b", f"def {rename_main}", src)
        src = re.sub(r"(?<![A-Za-z0-9_])main\s*\(\)", f"{rename_main}()", src)
    # Collapse runs of blank lines (cosmetic)
    src = re.sub(r"\n{4,}", "\n\n\n", src).rstrip() + "\n"
    return src


# ---------------------------------------------------------------------------
# Notebook scaffolding
# ---------------------------------------------------------------------------
_CELL_COUNTER = [0]


def _next_id() -> str:
    _CELL_COUNTER[0] += 1
    return f"cell-{_CELL_COUNTER[0]:03d}"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": _next_id(), "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": _next_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    }


# ---------------------------------------------------------------------------
# Section 0 — Setup
# ---------------------------------------------------------------------------
SEC0_MD = """\
# Technical Debt Prediction — Standalone Colab Pipeline

End-to-end reproduction of the LightGBM high-risk technical-debt classifier
on 22 Apache Java projects. The only external input is `td_V2.db` in
Google Drive.

## Section 0 — Setup

Mount Drive, configure paths, install non-pre-installed packages, and verify
the environment. **Edit `DRIVE_BASE` in cell 0.1 if your Drive layout differs.**"""


SEC0_1 = """\
# 0.1 — Mount Drive and copy the raw database to local disk
from google.colab import drive
drive.mount('/content/drive')

# >>> Edit this one line if your Drive layout differs <<<
DRIVE_BASE = '/content/drive/MyDrive/td_thesis'

import os, shutil
from pathlib import Path

PROJECT_ROOT = Path('/content')
os.chdir(PROJECT_ROOT)
(PROJECT_ROOT / 'data' / 'raw').mkdir(parents=True, exist_ok=True)

src_db = Path(DRIVE_BASE) / 'data' / 'raw' / 'td_V2.db'
dst_db = PROJECT_ROOT / 'data' / 'raw' / 'td_V2.db'
if not src_db.exists():
    raise FileNotFoundError(f'Place td_V2.db at: {src_db}')

# Always recopy from Drive — partial copies on free-tier Colab silently
# truncate large files. Delete any existing local copy first.
dst_db.unlink(missing_ok=True)
shutil.copy(src_db, dst_db)

src_size = src_db.stat().st_size
dst_size = dst_db.stat().st_size
if dst_size != src_size:
    raise RuntimeError(
        f'Copy truncated: source={src_size:,} bytes, local={dst_size:,} bytes. '
        'Re-run this cell.'
    )
print(f'DB ready at {dst_db}  ({dst_size / 1e9:.2f} GB, matches Drive)')
"""


SEC0_2 = '''\
# 0.2 — Project layout, global config (constants from config.py), shared helpers
import os, sys, json, time, shutil, warnings, sqlite3, re, math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, List, Tuple, Dict
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ---- paths ----
PROJECT_ROOT = Path('/content')
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
TABLES_DIR = RESULTS_DIR / 'tables'
MODELS_DIR = PROJECT_ROOT / 'models'
for _d in (PROCESSED_DATA_DIR, FIGURES_DIR, TABLES_DIR, TABLES_DIR / 'db_samples', MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
TD_DATASET_PATH = RAW_DATA_DIR / 'td_V2.db'

# ---- dataset filters ----
SOURCE_FILE_EXTENSIONS = ('.java',)
PATH_EXCLUSION_PATTERNS = ('/test/', '/tests/', '/generated/', '/generated-sources/', '/target/', '/build/')

# ---- snapshot ----
SNAPSHOT_STRATEGY = 'median'
OBSERVATION_WINDOW_MONTHS = 6
LABEL_SURROGATE_WINDOW_MONTHS = 6
MIN_PRE_SNAPSHOT_COMMITS = 500
MIN_POST_SNAPSHOT_COMMITS = 50

# ---- labeling ----
BUGFIX_REGEX = r"\\b(fix|bug|defect|patch|resolve|repair)\\b"
JIRA_ISSUE_KEY_PATTERN = r"\\b([A-Z][A-Z0-9_]+)-(\\d+)\\b"
THEORETICAL_WEIGHTS = {
    'S1_severity': 0.30, 'S4_bugfix': 0.25, 'S2_debt': 0.20,
    'S5_churn': 0.15, 'S3_smells': 0.05, 'S6_contributors': 0.05,
}
LABEL_RISK_THRESHOLD = 0.50
LABEL_THRESHOLD_FALLBACKS = (0.45, 0.40)
LABEL_MIN_POSITIVES_PER_PROJECT = 5
SEVERITY_LABEL_LEVELS = ('BLOCKER', 'CRITICAL')

# ---- feature catalogue (27 features, 5 families) ----
SIZE_COMPLEXITY_FEATURES = ['ncloc','complexity','cognitive_complexity','functions','classes']
STATIC_DEBT_FEATURES = ['n_code_smells','n_bugs','total_debt_minutes','issue_density','duplicated_lines_density']
HISTORICAL_FEATURES = ['total_commits_pre','code_churn_pre','recent_churn_90d','commit_frequency_30d',
                       'file_age_days','days_since_last_change','contributor_count','ownership_ratio']
GRAPH_FEATURES = ['cocg_degree','cocg_pagerank','cocg_betweenness','cocg_entropy']
PRIOR_DEFECT_FEATURES = ['bugfix_commits_pre','bugfix_commits_90d','bug_density_pre','n_jira_bugs_pre','jira_blocker_flag']
FEATURE_FAMILIES = {
    'size_complexity': SIZE_COMPLEXITY_FEATURES,
    'static_debt': STATIC_DEBT_FEATURES,
    'historical': HISTORICAL_FEATURES,
    'graph': GRAPH_FEATURES,
    'prior_defect': PRIOR_DEFECT_FEATURES,
}
ALL_FEATURES = SIZE_COMPLEXITY_FEATURES + STATIC_DEBT_FEATURES + HISTORICAL_FEATURES + GRAPH_FEATURES + PRIOR_DEFECT_FEATURES
assert len(ALL_FEATURES) == 27, f'expected 27 features, got {len(ALL_FEATURES)}'
LOG1P_FEATURES = ['ncloc','complexity','total_commits_pre','code_churn_pre','recent_churn_90d',
                  'file_age_days','total_debt_minutes','bugfix_commits_pre','n_jira_bugs_pre']
FEATURE_SELECTION_THRESHOLD = 0.001

# ---- modeling ----
RANDOM_STATE = 42
MODEL_ORDER = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
CV_FOLDS = 10
TUNING_TRIALS = 30
TUNING_INNER_CV_FOLDS = 5
TUNING_OBJECTIVE = 'pr_auc'
COST_EFFECTIVENESS_AT = 0.20
THRESHOLD_SWEEP_MIN = 0.05
THRESHOLD_SWEEP_MAX = 0.80
THRESHOLD_SWEEP_STEP = 0.01
THRESHOLD_DEFAULT = 0.30
THRESHOLD_MIN_POSITIVES = 3

# Lower parallelism in Colab to avoid pickling overhead.
STAGE5_N_JOBS = min(2, os.cpu_count() or 2)

# ---- display helpers ----
from IPython.display import Image, display

def show_figure(rel_path):
    p = Path(rel_path)
    if not p.exists():
        print(f'(figure not found: {p})'); return
    display(Image(filename=str(p)))

def show_table(csv_path, highlight_col=None, top=None):
    df = pd.read_csv(csv_path)
    if top is not None:
        df = df.head(top)
    num = df.select_dtypes('number').columns
    if len(num):
        df[num] = df[num].round(4)
    if highlight_col and highlight_col in df.columns:
        display(df.style.highlight_max(subset=[highlight_col], color='lightgreen'))
    else:
        display(df)

print('Paths + config ready.  PROJECT_ROOT =', PROJECT_ROOT)
'''


SEC0_3 = """\
# 0.3 — Install only the packages not pre-installed in Colab
!pip install -q xgboost>=2.0.0 lightgbm>=4.0.0 optuna>=3.5.0 shap>=0.44.0 \\
    pyarrow>=14.0.0 pydriller>=2.5 imbalanced-learn>=0.11.0 \\
    matplotlib-venn>=0.11.9 tabulate>=0.9.0 networkx>=3.2 igraph>=0.11
print('Dependencies installed.')
"""


SEC0_4 = """\
# 0.4 — Verify environment
import platform
import xgboost, lightgbm, optuna, shap
print(f'Python      : {platform.python_version()}')
print(f'xgboost     : {xgboost.__version__}')
print(f'lightgbm    : {lightgbm.__version__}')
print(f'optuna      : {optuna.__version__}')
print(f'shap        : {shap.__version__}')

with sqlite3.connect(str(TD_DATASET_PATH)) as conn:
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;", conn)
print(f'DB tables    : {len(tables)}')
print(tables['name'].tolist())
"""


# ---------------------------------------------------------------------------
# Section 1 — Database Inspection
# ---------------------------------------------------------------------------
SEC1_MD = """\
## Section 1 — Database Inspection

Inspect the raw SQLite database schema and confirm all 10 tables are present.
Writes `results/tables/db_schema.csv`, `db_table_counts.csv`, and per-table
sample CSVs."""

SEC1_CODE_PIPELINE = read_source("src/data/load_data.py")
SEC1_CODE_SCRIPT = read_source("scripts/01_inspect_db.py", rename_main="run_stage_1")
SEC1_RUN = """\
run_stage_1()
print()
print('--- db_table_counts.csv ---')
show_table('results/tables/db_table_counts.csv')
"""


# ---------------------------------------------------------------------------
# Section 2 — Project Profiling
# ---------------------------------------------------------------------------
SEC2_MD = """\
## Section 2 — Project Profiling

Profile 33 Apache projects and select the 22 eligible ones using the
500-pre / 50-post commit thresholds. Writes `project_stats.csv` and
`corpus_summary.csv`."""

SEC2_CODE_PIPELINE = read_source("src/data/snapshot.py")
SEC2_CODE_SCRIPT = read_source("scripts/02_profile_projects.py", rename_main="run_stage_2")
SEC2_RUN = """\
run_stage_2()
print()
print('--- corpus_summary.csv (22 eligible) ---')
show_table('results/tables/corpus_summary.csv')
"""


# ---------------------------------------------------------------------------
# Section 3 — Data Cleaning and Collision Resolution
# ---------------------------------------------------------------------------
SEC3_MD = """\
## Section 3 — Data Cleaning and Collision Resolution

Resolve basename path collisions and filter every table to clean
file-level instances. Writes the six `clean_*.parquet` files and
`collision_report.csv`."""

SEC3_CODE_PIPELINE = read_source("src/data/clean.py")
SEC3_CODE_SCRIPT = read_source("scripts/03_clean.py", rename_main="run_stage_3")
SEC3_RUN = """\
run_stage_3()
print()
print('--- collision_report.csv ---')
df = pd.read_csv('data/processed/collision_report.csv')
display(df.style.background_gradient(subset=['drop_rate_pct'], cmap='Reds'))
"""


# ---------------------------------------------------------------------------
# Section 4 — High-Risk TD Labeling
# ---------------------------------------------------------------------------
SEC4_MD = """\
## Section 4 — High-Risk TD Labeling

Compute the combined-weight high-risk TD label using six binary signals
(three static SonarQube signals + three git-history signals). Empirical
weights derived via point-biserial correlation; falls back to theoretical
Kamei 2013 weights if within 0.05. Writes `labels.parquet`,
`derived_weights.json`, `label_summary.csv`."""

SEC4_CODE_PIPELINE = read_source("src/data/labeling.py")
SEC4_CODE_SZZ = read_source("src/data/szz.py")
SEC4_CODE_SCRIPT = read_source("scripts/04_label.py", rename_main="run_stage_4")
SEC4_RUN = """\
run_stage_4()
print()
print('--- derived_weights.json ---')
print(json.dumps(json.loads(Path('data/processed/derived_weights.json').read_text()), indent=2))
print()
print('--- label_summary.csv ---')
show_table('results/tables/label_summary.csv')

# fig_01/02/03 are produced by Stage 10's report renderer (Section 12);
# they will not exist at this point. show_figure prints a notice and
# moves on rather than raising.
print()
show_figure('results/figures/fig_01_positive_rates.png')
show_figure('results/figures/fig_02_label_signal_breakdown.png')
show_figure('results/figures/fig_03_risk_score_distribution.png')
"""

# Note: figures 01/02/03 are generated by 10_report.py — Section 4 produces
# their inputs but the PNGs only appear after Section 12 runs. We still call
# show_figure() here; if the PNG doesn't exist yet the helper prints a tiny
# notice rather than raising. The same images are shown again where the
# spec asks for them.


# ---------------------------------------------------------------------------
# Section 5 — Feature Engineering
# ---------------------------------------------------------------------------
SEC5_MD = """\
## Section 5 — Feature Engineering

Extract 27 features in 5 families (size/complexity, static debt, historical,
co-change graph, prior defect) at the project/basename snapshot. Writes
four `features_*.parquet` files + `feature_summary.csv`."""

SEC5_CODE_STATIC = read_source("src/features/static_features.py")
SEC5_CODE_HIST = read_source("src/features/historical_features.py")
SEC5_CODE_GRAPH = read_source("src/features/graph_features.py")
SEC5_CODE_PRIOR = read_source("src/features/priordefect_features.py")
SEC5_CODE_SCRIPT = read_source("scripts/05_features.py", rename_main="run_stage_5")
SEC5_RUN = """\
run_stage_5()
print()
print('--- feature_summary.csv ---')
show_table('results/tables/feature_summary.csv')
"""


# ---------------------------------------------------------------------------
# Section 6 — Dataset Assembly
# ---------------------------------------------------------------------------
SEC6_MD = """\
## Section 6 — Dataset Assembly

Merge features and labels into `dataset_final.parquet`. Apply `log1p` to
heavy-tailed columns and assert no leakage (raw severity counts and the
weight-derivation surrogate are absent). Writes the dataset and
`feature_catalog.csv`."""

SEC6_CODE_SCRIPT = read_source("scripts/06_build_dataset.py", rename_main="run_stage_6")
SEC6_RUN = """\
run_stage_6()
print()
ds = pd.read_parquet('data/processed/dataset_final.parquet')
print(f'shape: {ds.shape}   positive rate: {100*ds["is_high_risk"].mean():.2f}%')

# fig_04 is produced later by 10_report.py (Section 12); show if available.
show_figure('results/figures/fig_04_feature_correlation_heatmap.png')

# Drive checkpoint
shutil.copytree('data/processed', f'{DRIVE_BASE}/data/processed', dirs_exist_ok=True)
print('\\nDataset checkpointed to Drive.')
"""


# ---------------------------------------------------------------------------
# Section 7 — Default Model Training
# ---------------------------------------------------------------------------
SEC7_MD = """\
## Section 7 — Default Model Training

Train all 4 models with default hyperparameters under stratified 10-fold
within-project CV as a pre-tuning baseline. Writes `default_cv_results.csv`.
processed data from Drive if the local copy is missing."""

SEC7_RESUME = """\
# Resume guard — load processed data from Drive if missing locally
if not Path('data/processed/dataset_final.parquet').exists():
    print('Local data missing. Loading from Drive...')
    shutil.copytree(f'{DRIVE_BASE}/data/processed', 'data/processed', dirs_exist_ok=True)
    print('Loaded from Drive.')
else:
    print('Local data found. Proceeding.')
"""

SEC7_CODE_TRAIN = read_source("src/models/train.py")

# 07_train.py uses argparse. Patch the inlined source so main() accepts a
# tuned kwarg, dropping the argparse boilerplate; everything else stays.
_raw_07 = read_source("scripts/07_train.py", rename_main="run_stage_7")
_raw_07 = _raw_07.replace(
    "import argparse\n", ""
).replace(
    'def run_stage_7() -> None:\n'
    '    parser = argparse.ArgumentParser(description=__doc__)\n'
    '    parser.add_argument("--tuned", action="store_true", help="Use tuned hyperparams from 07b_tune.py")\n'
    '    args = parser.parse_args()\n',
    'def run_stage_7(tuned: bool = False) -> None:\n'
    '    class _A: pass\n'
    '    args = _A(); args.tuned = tuned\n',
)
SEC7_CODE_SCRIPT = _raw_07

SEC7_RUN = """\
run_stage_7(tuned=False)
print()
print('--- default_cv_results.csv (per-fold summary) ---')
df = pd.read_csv('results/tables/default_cv_results.csv')
summary = df.groupby('model')[['f1','roc_auc','pr_auc','ce_at_20']].mean().round(4).reset_index()
display(summary.style.highlight_max(subset=['f1'], color='lightgreen'))
"""


# ---------------------------------------------------------------------------
# Section 8 — Hyperparameter Tuning
# ---------------------------------------------------------------------------
SEC8_MD = """\
## Section 8 — Hyperparameter Tuning

Tune all 4 models with Optuna (30 trials each, PR-AUC objective, 5-fold
inner CV); logistic regression uses GridSearchCV over `C`. Writes
`tuned_hyperparameters.csv`.
step in the pipeline. Do not close the browser tab during tuning."""

SEC8_CODE_TUNING = read_source("src/models/tuning.py")
SEC8_CODE_SCRIPT = read_source("scripts/07b_tune.py", rename_main="run_stage_7b")
SEC8_RUN = """\
run_stage_7b()
print()
print('--- tuned_hyperparameters.csv ---')
show_table('results/tables/tuned_hyperparameters.csv')
"""


# ---------------------------------------------------------------------------
# Section 9 — Tuned Within-Project Validation
# ---------------------------------------------------------------------------
SEC9_MD = """\
## Section 9 — Tuned Within-Project Validation

Re-evaluate all 4 tuned models under stratified 10-fold within-project CV.
Writes `within_project_results.csv`."""

SEC9_RUN = """\
# Reuses run_stage_7 defined in Section 7
run_stage_7(tuned=True)
print()
print('--- within_project_results.csv (mean per model) ---')
df = pd.read_csv('results/tables/within_project_results.csv')
summary = df.groupby('model')[['f1','roc_auc','pr_auc','ce_at_20']].mean().round(4).reset_index()
display(summary.style.highlight_max(subset=['f1'], color='lightgreen'))

# fig_06 is produced later by 10_report.py (Section 12); show if available.
show_figure('results/figures/fig_06_model_comparison_f1.png')
"""


# ---------------------------------------------------------------------------
# Section 10 — LOPO Cross-Project Validation
# ---------------------------------------------------------------------------
SEC10_MD = """\
## Section 10 — LOPO Cross-Project Validation

Leave-One-Project-Out validation across 22 projects with similarity-weighted
training (org.apache:daemon excluded from test, kept in train). This is the
primary evaluation — the model is tested on projects it has never seen.

Writes `lopo_results.csv`, `lopo_per_project.csv`, `model_comparison.csv`."""

SEC10_RESUME = """\
# Resume guard
if not Path('data/processed/dataset_final.parquet').exists():
    print('Local data missing. Loading from Drive...')
    shutil.copytree(f'{DRIVE_BASE}/data/processed', 'data/processed', dirs_exist_ok=True)
"""

SEC10_CODE_CROSS = read_source("src/models/cross_project.py")
SEC10_CODE_SCRIPT = read_source("scripts/08_lopo.py", rename_main="run_stage_8")
SEC10_RUN = """\
run_stage_8()
print()
print('--- lopo_results.csv ---')
df = pd.read_csv('results/tables/lopo_results.csv')
num = df.select_dtypes('number').columns
df[num] = df[num].round(4)
display(df.style.highlight_max(subset=['f1_mean'], color='lightgreen'))

print()
print('--- model_comparison.csv ---')
mc = pd.read_csv('results/tables/model_comparison.csv')
num = mc.select_dtypes('number').columns
mc[num] = mc[num].round(4)
display(mc)

best = mc[mc['is_best']].iloc[0]
print(f"\\n{best['model'].upper()} (best)")
print(f"  LOPO F1     = {best['lopo_f1']:.4f}")
print(f"  LOPO CE@20  = {best['lopo_ce_at_20']:.4f}")
print(f"  ({100*best['lopo_ce_at_20']:.1f}% of high-risk files found by reviewing top 20%)")

# fig_07 / fig_08 are produced later by 10_report.py (Section 12); show if available.
show_figure('results/figures/fig_07_lopo_per_project_f1.png')
show_figure('results/figures/fig_08_lopo_ce20_distribution.png')

# Drive checkpoint
shutil.copytree('results', f'{DRIVE_BASE}/results', dirs_exist_ok=True)
print('\\nResults checkpointed to Drive.')
"""


# ---------------------------------------------------------------------------
# Section 11 — Feature Family Ablation
# ---------------------------------------------------------------------------
SEC11_MD = """\
## Section 11 — Feature Family Ablation

Systematically remove each feature family (and isolate each family alone)
to measure which groups carry the most predictive signal. 11-row table:
1 baseline + 5 families × 2 modes. Writes `ablation_results.csv`."""

SEC11_CODE_ABL = read_source("src/analysis/ablation.py")
SEC11_CODE_SCRIPT = read_source("scripts/09_ablation.py", rename_main="run_stage_9")
SEC11_RUN = """\
run_stage_9()
print()
print('--- ablation_results.csv ---')
show_table('results/tables/ablation_results.csv')

# fig_10 is produced later by 10_report.py (Section 12); show if available.
show_figure('results/figures/fig_10_ablation.png')
"""


# ---------------------------------------------------------------------------
# Section 12 — Feature Importance and Report Figures
# ---------------------------------------------------------------------------
SEC12_MD = """\
## Section 12 — Feature Importance and Report Figures

SHAP analysis on the best model (top 15 features → fig_11) and permutation
importance for all 4 models (top 10 each → `feature_importance.csv`).
Pooled LOPO predictions across the 4 models produce fig_12 ROC/PR curves.
Also generates fig_05 and fig_09. All 12 thesis figures are present after
this section."""

SEC12_CODE_FIGURES = read_source("src/reporting/figures.py")
# figures.py has `from src.models.train import ...` indented inside function
# bodies (fig_09, fig_11). The module-level strip regex misses these.
SEC12_CODE_FIGURES = re.sub(
    r"^\s*from src\.models\.train import .*$",
    "",
    SEC12_CODE_FIGURES,
    flags=re.MULTILINE,
)
SEC12_CODE_SCRIPT = read_source("scripts/10_report.py", rename_main="run_stage_10")
# 10_report.py uses `F.fig_XX` via `from src.reporting import figures as F`,
# which our stripper removed. figures.py is inlined in the same cell, so the
# functions are already module-level — rewrite `F.fig_` -> `fig_`.
SEC12_CODE_SCRIPT = re.sub(r"\bF\.fig_", "fig_", SEC12_CODE_SCRIPT)
SEC12_RUN = """\
run_stage_10()
print()
# All 12 figures are now generated. Show the three explicitly requested for this section.
show_figure('results/figures/fig_09_threshold_curve.png')
show_figure('results/figures/fig_11_shap_best_model.png')
show_figure('results/figures/fig_12_roc_pr_curves.png')

print()
print('--- feature_importance.csv (top 10) ---')
show_table('results/tables/feature_importance.csv', top=10)
"""


# ---------------------------------------------------------------------------
# Section 13 — Model Persistence
# ---------------------------------------------------------------------------
SEC13_MD = """\
## Section 13 — Model Persistence

Persist the best model, scaler, threshold (0.5), feature names, scoring
helper, and model card to `models/` and to Drive."""

SEC13_CODE_SCRIPT = read_source("scripts/11_persist.py", rename_main="run_stage_11")
SEC13_RUN = """\
run_stage_11()
shutil.copytree('models', f'{DRIVE_BASE}/models', dirs_exist_ok=True)
print('\\nModels checkpointed to Drive.')

card = json.loads(Path('models/model_card.json').read_text())
print('\\n--- model_card.json ---')
print(json.dumps(card, indent=2))
"""


# ---------------------------------------------------------------------------
# Section 14 — Score a New Project (Demo)
# ---------------------------------------------------------------------------
SEC14_MD = """\
## Section 14 — Score a New Project (Demo)

Demonstrate the persisted model scoring files in a project it was not
trained on — using saved features from one held-out project
(`org.apache:zookeeper`, 170 files, 17.9% positive rate). Displays the
top-10 highest-risk files and a top-20% review summary."""

SEC14_RUN = """\
import joblib

DEMO_PROJECT = 'org.apache:zookeeper'

best_model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/feature_scaler.pkl')
threshold = float(Path('models/optimal_threshold.txt').read_text())
feat_names = pd.read_csv('models/feature_names.csv')['feature'].tolist()

ds = pd.read_parquet('data/processed/dataset_final.parquet')
demo = ds[ds['project_id'] == DEMO_PROJECT].copy()
print(f'Demo project: {DEMO_PROJECT}  files={len(demo)}  positive_rate={100*demo["is_high_risk"].mean():.2f}%')

X = demo[feat_names].fillna(0).astype(float).copy()
# dataset_final.parquet already has log1p applied, so we do NOT log1p again here.
X_scaled = scaler.transform(X.values)
probs = best_model.predict_proba(X_scaled)[:, 1]

scored = demo[['basename', 'is_high_risk']].copy()
scored['risk_score'] = probs
scored['predicted_high_risk'] = (probs >= threshold).astype(int)
scored = scored.sort_values('risk_score', ascending=False).reset_index(drop=True)

print('\\nTop 10 highest-risk files:')
display(scored[['basename', 'risk_score', 'predicted_high_risk']].head(10).style.format({'risk_score': '{:.4f}'}))

# Top-20% CE summary
n = len(scored)
top_k = max(1, int(round(0.20 * n)))
top = scored.head(top_k)
n_pos_total = int(scored['is_high_risk'].sum())
n_pos_found = int(top['is_high_risk'].sum())
ce_at_20 = n_pos_found / n_pos_total if n_pos_total else 0.0
print(f'\\nReviewed top 20% of files ({top_k} of {n} files).')
print(f'Would find {n_pos_found} of {n_pos_total} truly high-risk files.')
print(f'CE@20 = {ce_at_20:.4f}')
"""


# ---------------------------------------------------------------------------
# Section 15 — Analyze Any Apache Java Project from GitHub
# ---------------------------------------------------------------------------
SEC15_MD = """\
## Section 15 — Analyze Any Apache Java Project from GitHub

Given a GitHub repository URL of any Apache Java project, this section clones
the repo, computes the 27 features from git history, scores all Java files
using the persisted model, and outputs a ranked risk report. SonarQube
features default to zero if not available — the model still runs using the
14 git-based features only, with slightly reduced accuracy."""

SEC15_INPUT = """\
# STEP 1 — User input. Edit this URL to analyze any Apache Java project.
GITHUB_REPO_URL = 'https://github.com/apache/commons-lang'
REPO_NAME = 'commons-lang'
CLONE_DEPTH = 500  # ~500 most recent commits; lower if the repo is huge
"""

SEC15_CLONE = """\
# STEP 2 — Clone the repository
# Shallow clone first for speed, then unshallow so PyDriller can diff every
# commit against its real parent (a grafted-parent boundary in a shallow
# clone breaks `git diff-tree` and crashes PyDriller).
import subprocess

clone_dir = f'/content/repos/{REPO_NAME}'
Path('/content/repos').mkdir(parents=True, exist_ok=True)
if not Path(clone_dir).exists():
    subprocess.run(
        ['git', 'clone', f'--depth={CLONE_DEPTH}', GITHUB_REPO_URL, clone_dir],
        check=True,
    )
    print(f'Cloned {REPO_NAME} into {clone_dir}')
else:
    print(f'Already cloned: {clone_dir}')

# Unshallow if needed so PyDriller can diff every commit against its parent.
is_shallow = (Path(clone_dir) / '.git' / 'shallow').exists()
if is_shallow:
    print('Unshallowing clone (needed for PyDriller diffs) ...')
    subprocess.run(['git', '-C', clone_dir, 'fetch', '--unshallow'], check=True)

n_commits = int(subprocess.check_output(
    ['git', '-C', clone_dir, 'rev-list', '--count', 'HEAD']
).decode().strip())
print(f'Available commits: {n_commits}')
"""

SEC15_FEATURES = '''\
# STEP 3 — Extract git features (Families 3, 4, 5) with PyDriller.
# Families 1 and 2 (SonarQube) default to 0 — the model still runs but
# accuracy is reduced (~80% of the full pipeline). This is noted below.
from pydriller import Repository

_BUGFIX_RE = re.compile(BUGFIX_REGEX, re.IGNORECASE)

print(f'Traversing {REPO_NAME} commits with PyDriller ...')
file_records = {}  # basename -> dict of per-file aggregates
authors_per_file = {}  # basename -> {author: commit_count}
cochange_edges = Counter()  # (a, b) sorted -> co-change weight
all_basenames = set()

t_now = pd.Timestamp.utcnow()
t_30d = t_now - pd.Timedelta(days=30)
t_90d = t_now - pd.Timedelta(days=90)

# Cap traversal to the most recent CLONE_DEPTH commits to keep runtime
# bounded on large repos even after unshallowing.
head_rev_list = subprocess.check_output(
    ['git', '-C', clone_dir, 'rev-list', f'--max-count={CLONE_DEPTH}', 'HEAD']
).decode().splitlines()
target_hashes = set(head_rev_list)
print(f'Traversing latest {len(target_hashes)} commits ...')

n_seen = 0
n_skipped = 0
for commit in Repository(clone_dir).traverse_commits():
    if commit.hash not in target_hashes:
        continue
    n_seen += 1
    commit_date = pd.Timestamp(commit.author_date).tz_convert('UTC') if commit.author_date.tzinfo else pd.Timestamp(commit.author_date, tz='UTC')
    msg = commit.msg or ''
    is_bugfix = bool(_BUGFIX_RE.search(msg))
    author = commit.author.name if commit.author else 'unknown'
    touched_in_commit = []
    try:
        modified_files = commit.modified_files
    except Exception as _e:
        # Grafted-parent boundary or any other diff failure — skip this commit.
        n_skipped += 1
        continue
    for mf in modified_files:
        path = mf.new_path or mf.old_path or ''
        if not path.endswith('.java'):
            continue
        p = path.replace('\\\\', '/').lower()
        if any(bad in p for bad in PATH_EXCLUSION_PATTERNS):
            continue
        bn = path.rsplit('/', 1)[-1]
        added = mf.added_lines or 0
        removed = mf.deleted_lines or 0
        rec = file_records.setdefault(bn, {
            'total_commits_pre': 0,
            'code_churn_pre': 0,
            'recent_churn_90d': 0,
            'commit_frequency_30d': 0,
            'first_commit': commit_date,
            'last_commit': commit_date,
            'bugfix_commits_pre': 0,
            'bugfix_commits_90d': 0,
        })
        rec['total_commits_pre'] += 1
        rec['code_churn_pre'] += added + removed
        if commit_date >= t_90d:
            rec['recent_churn_90d'] += added + removed
            if is_bugfix:
                rec['bugfix_commits_90d'] += 1
        if commit_date >= t_30d:
            rec['commit_frequency_30d'] += 1
        if commit_date < rec['first_commit']:
            rec['first_commit'] = commit_date
        if commit_date > rec['last_commit']:
            rec['last_commit'] = commit_date
        if is_bugfix:
            rec['bugfix_commits_pre'] += 1
        authors_per_file.setdefault(bn, Counter())[author] += 1
        touched_in_commit.append(bn)
        all_basenames.add(bn)
    if len(touched_in_commit) > 1:
        for a, b in combinations(sorted(set(touched_in_commit)), 2):
            cochange_edges[(a, b)] += 1

print(f'Traversed {n_seen} commits ({n_skipped} skipped); collected {len(all_basenames)} Java basenames.')

# Co-change graph features (igraph for parity with the training pipeline)
import igraph as ig
node_list = sorted(all_basenames)
node_idx = {n: i for i, n in enumerate(node_list)}
edges = [(node_idx[a], node_idx[b]) for (a, b) in cochange_edges]
weights = list(cochange_edges.values())
G = ig.Graph(n=len(node_list), edges=edges, directed=False)
if len(node_list) > 2 and edges:
    bc_raw = G.betweenness(directed=False)
    denom = (len(node_list) - 1) * (len(node_list) - 2) / 2.0
    bc_norm = [b / denom for b in bc_raw]
else:
    bc_norm = [0.0] * len(node_list)
try:
    pr = G.pagerank(damping=0.85, weights=weights if weights else None)
except Exception:
    pr = G.pagerank(damping=0.85)
deg = G.degree()

# Per-node entropy over normalised neighbour edge weights
neighbour_weights = {n: [] for n in node_list}
for (a, b), w in cochange_edges.items():
    neighbour_weights[a].append(w)
    neighbour_weights[b].append(w)
entropy = {}
for n in node_list:
    ws = neighbour_weights[n]
    s = sum(ws)
    if s <= 0:
        entropy[n] = 0.0
        continue
    h = 0.0
    for w in ws:
        p = w / s
        if p > 0:
            h -= p * math.log2(p)
    entropy[n] = h

# Assemble the 27-column matrix
rows = []
for bn in node_list:
    rec = file_records[bn]
    age_days = max(0, (rec['last_commit'] - rec['first_commit']).days)
    days_since = max(0, (t_now - rec['last_commit']).days)
    authors = authors_per_file.get(bn, Counter())
    n_authors = len(authors)
    max_author = max(authors.values()) if authors else 0
    ownership = (max_author / rec['total_commits_pre']) if rec['total_commits_pre'] else 1.0
    bug_density = (rec['bugfix_commits_pre'] / rec['total_commits_pre']) if rec['total_commits_pre'] else 0.0
    i = node_idx[bn]
    row = {
        'basename': bn,
        # Family 1: size/complexity — SonarQube features default to 0
        'ncloc': 0.0, 'complexity': 0.0, 'cognitive_complexity': 0.0,
        'functions': 0.0, 'classes': 0.0,
        # Family 2: static debt — SonarQube features default to 0
        'n_code_smells': 0, 'n_bugs': 0, 'total_debt_minutes': 0.0,
        'issue_density': 0.0, 'duplicated_lines_density': 0.0,
        # Family 3: historical
        'total_commits_pre': rec['total_commits_pre'],
        'code_churn_pre': rec['code_churn_pre'],
        'recent_churn_90d': rec['recent_churn_90d'],
        'commit_frequency_30d': rec['commit_frequency_30d'],
        'file_age_days': age_days,
        'days_since_last_change': days_since,
        'contributor_count': n_authors,
        'ownership_ratio': ownership,
        # Family 4: co-change graph
        'cocg_degree': deg[i],
        'cocg_pagerank': pr[i],
        'cocg_betweenness': bc_norm[i],
        'cocg_entropy': entropy[bn],
        # Family 5: prior defect
        'bugfix_commits_pre': rec['bugfix_commits_pre'],
        'bugfix_commits_90d': rec['bugfix_commits_90d'],
        'bug_density_pre': bug_density,
        'n_jira_bugs_pre': 0,
        'jira_blocker_flag': 0,
    }
    rows.append(row)

feat_df = pd.DataFrame(rows)
print(f'Built feature matrix: {feat_df.shape}')
print('\\nStatic features unavailable without SonarQube. Using git-based features only.')
print('Expected accuracy: ~80% of full model.')
'''

SEC15_SCORE = """\
# STEP 4 — Score files using the persisted model
import joblib

best_model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/feature_scaler.pkl')
feat_names = pd.read_csv('models/feature_names.csv')['feature'].tolist()
threshold = float(Path('models/optimal_threshold.txt').read_text())

# Align to the trained feature order; missing columns become 0
X = feat_df.reindex(columns=feat_names, fill_value=0).fillna(0).astype(float)
assert list(X.columns) == feat_names, 'Feature column mismatch'

# Apply log1p to the same 9 skewed columns the training pipeline did
for col in LOG1P_FEATURES:
    if col in X.columns:
        X[col] = np.log1p(X[col])

X_scaled = scaler.transform(X.values)
probs = best_model.predict_proba(X_scaled)[:, 1]

scored = pd.DataFrame({
    'basename': feat_df['basename'],
    'risk_score': probs,
    'predicted_high_risk': (probs >= threshold).astype(int),
}).sort_values('risk_score', ascending=False).reset_index(drop=True)

print(f'Scored {len(scored)} Java files.')
"""

SEC15_DISPLAY = """\
# STEP 5 — Display results
import matplotlib.pyplot as plt

print(f'\\nTop 20 highest-risk files in {REPO_NAME}:')
display(scored.head(20).style.format({'risk_score': '{:.4f}'}))

n = len(scored)
k = max(1, int(round(0.20 * n)))
n_high = int(scored['predicted_high_risk'].sum())
print(f'\\nTotal Java files analyzed   : {n}')
print(f'Files in top 20% review budget: {k}')
print(f'High-risk files predicted    : {n_high}')
print(f'\\nTo prioritize maintenance effort, focus on the top {k} files listed above.')

top20 = scored.head(20)
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top20['basename'][::-1], top20['risk_score'][::-1], color='#d62728')
ax.set_xlabel('Predicted risk score')
ax.set_title(f'Top 20 high-risk files: {REPO_NAME}')
fig.tight_layout()
plt.show()
"""

SEC15_SAVE = """\
# STEP 6 — Save report to Drive
out_path = f'{DRIVE_BASE}/risk_report_{REPO_NAME}.csv'
scored.to_csv(out_path, index=False)
print(f'Full report saved to Drive: risk_report_{REPO_NAME}.csv')
"""

SEC15_FOOTER_MD = """\
## Taking This Further

This scoring logic can be integrated into a GitHub Actions workflow to
automatically flag high-risk files on every pull request or release. The
workflow would:

1. Checkout the repository
2. Install PyDriller and the model artifacts
3. Run the feature extraction and scoring
4. Comment on the PR with the top high-risk files, or fail the check if any
   file exceeds a defined risk threshold

The persisted model artifacts (saved to `models/` in Section 13) are all that
is needed to run inference on any new project."""


# ---------------------------------------------------------------------------
# Build notebook
# ---------------------------------------------------------------------------
def build():
    cells = []

    # Section 0
    cells.append(md(SEC0_MD))
    cells.append(code(SEC0_1))
    cells.append(code(SEC0_2))
    cells.append(code(SEC0_3))
    cells.append(code(SEC0_4))

    # Section 1
    cells.append(md(SEC1_MD))
    cells.append(code("# Pipeline code\n\n"
                      + SEC1_CODE_PIPELINE + "\n\n" + SEC1_CODE_SCRIPT))
    cells.append(code(SEC1_RUN))

    # Section 2
    cells.append(md(SEC2_MD))
    cells.append(code("# Pipeline code\n\n"
                      + SEC2_CODE_PIPELINE + "\n\n" + SEC2_CODE_SCRIPT))
    cells.append(code(SEC2_RUN))

    # Section 3
    cells.append(md(SEC3_MD))
    cells.append(code("# Pipeline code\n\n"
                      + SEC3_CODE_PIPELINE + "\n\n" + SEC3_CODE_SCRIPT))
    cells.append(code(SEC3_RUN))

    # Section 4
    cells.append(md(SEC4_MD))
    cells.append(code("# Pipeline code\n\n"
                      + SEC4_CODE_SZZ + "\n\n" + SEC4_CODE_PIPELINE + "\n\n" + SEC4_CODE_SCRIPT))
    cells.append(code(SEC4_RUN))

    # Section 5 (szz already loaded in Section 4)
    cells.append(md(SEC5_MD))
    cells.append(code(
        "# Pipeline code\n\n"
        + "from joblib import Parallel, delayed\n\n"
        + SEC5_CODE_STATIC + "\n\n"
        + SEC5_CODE_HIST + "\n\n"
        + SEC5_CODE_GRAPH + "\n\n"
        + SEC5_CODE_PRIOR + "\n\n"
        + SEC5_CODE_SCRIPT
    ))
    cells.append(code(SEC5_RUN))

    # Section 6
    cells.append(md(SEC6_MD))
    cells.append(code("# Pipeline code\n\n" + SEC6_CODE_SCRIPT))
    cells.append(code(SEC6_RUN))

    # Section 7
    cells.append(md(SEC7_MD))
    cells.append(code(SEC7_RESUME))
    cells.append(code("# Pipeline code\n\n"
                      + SEC7_CODE_TRAIN + "\n\n" + SEC7_CODE_SCRIPT))
    cells.append(code(SEC7_RUN))

    # Section 8
    cells.append(md(SEC8_MD))
    cells.append(code("# Pipeline code\n\n"
                      + SEC8_CODE_TUNING + "\n\n" + SEC8_CODE_SCRIPT))
    cells.append(code(SEC8_RUN))

    # Section 9 — reuses Section 7's main with tuned=True
    cells.append(md(SEC9_MD))
    cells.append(code(SEC9_RUN))

    # Section 10
    cells.append(md(SEC10_MD))
    cells.append(code(SEC10_RESUME))
    cells.append(code("# Pipeline code\n\n"
                      + SEC10_CODE_CROSS + "\n\n" + SEC10_CODE_SCRIPT))
    cells.append(code(SEC10_RUN))

    # Section 11
    cells.append(md(SEC11_MD))
    cells.append(code("# Pipeline code\n\n"
                      + SEC11_CODE_ABL + "\n\n" + SEC11_CODE_SCRIPT))
    cells.append(code(SEC11_RUN))

    # Section 12
    cells.append(md(SEC12_MD))
    cells.append(code("# Pipeline code\n\n"
                      + SEC12_CODE_FIGURES + "\n\n" + SEC12_CODE_SCRIPT))
    cells.append(code(SEC12_RUN))

    # Section 13
    cells.append(md(SEC13_MD))
    cells.append(code("# Pipeline code\n\n" + SEC13_CODE_SCRIPT))
    cells.append(code(SEC13_RUN))

    # Section 14
    cells.append(md(SEC14_MD))
    cells.append(code(SEC14_RUN))

    # Section 15 — Analyze any Apache Java project from GitHub
    cells.append(md(SEC15_MD))
    cells.append(code(SEC15_INPUT))
    cells.append(code(SEC15_CLONE))
    cells.append(code(SEC15_FEATURES))
    cells.append(code(SEC15_SCORE))
    cells.append(code(SEC15_DISPLAY))
    cells.append(code(SEC15_SAVE))
    cells.append(md(SEC15_FOOTER_MD))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {OUT}  ({len(cells)} cells, {OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
