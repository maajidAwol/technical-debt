"""
Generator for the Google Colab replication notebooks.

Reads every src/ module verbatim from disk and assembles two Colab-ready
notebooks that reproduce Stages 1-10 of the technical-debt prediction
pipeline:

    notebooks/td_pipeline_colab.ipynb       full 22-project, stages 1-10
    notebooks/td_pipeline_colab_demo.ipynb  5-project subset, stages 1-7

A third "standalone" variant (src/ inlined, no %%writefile) is also
produced for ad-hoc experimentation.

Run from the repository root:

    python notebooks/_build_notebook.py                  # all outputs
    python notebooks/_build_notebook.py --mode full      # td_pipeline_colab.ipynb only
    python notebooks/_build_notebook.py --mode demo      # td_pipeline_colab_demo.ipynb only
    python notebooks/_build_notebook.py --mode standalone

After building, this script can be deleted; the .ipynb files are
self-contained.

Hardening applied to both notebooks:
  * True copy (not symlink) of td_V2.db onto Colab local SSD with
    SHA-256 verification and 3-attempt FUSE retry - root-cause fix
    for the runtime disconnections seen on the 1.54 GB SQLite file
    when read straight off Drive.
  * Idempotent stage-skip with checkpoints rsynced to Drive after
    every stage - reconnect-and-rerun is safe.
  * Colab tier detection (free / pro / pro+) drives TD_N_JOBS so
    the pipeline does not oversubscribe a 2-core / 12 GB free VM.
  * tqdm.auto progress bars throughout.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_FULL = REPO / "notebooks" / "td_pipeline_colab.ipynb"
OUT_DEMO = REPO / "notebooks" / "td_pipeline_colab_demo.ipynb"
OUT_STANDALONE = REPO / "notebooks" / "td_pipeline_colab_standalone.ipynb"
OUT = OUT_FULL  # legacy alias used by the existing build flow

DEMO_PROJECTS = [
    "org.apache:archiva",
    "org.apache:codec",
    "org.apache:commons-jexl",
    "org.apache:configuration",
    "org.apache:httpclient",
]


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------
def _split_keep_lines(text: str) -> list[str]:
    """Convert a multi-line string to nbformat ``source`` (list of lines, each
    keeping its trailing newline except the last)."""
    if not text:
        return []
    lines = text.splitlines(keepends=True)
    return lines


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _split_keep_lines(text),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split_keep_lines(text),
    }


def writefile_cell(target_path: str, file_content: str) -> dict:
    """%%writefile cell that materialises a file on the Colab filesystem."""
    body = f"%%writefile {target_path}\n{file_content}"
    return code(body)


# ---------------------------------------------------------------------------
# Read source modules from disk
# ---------------------------------------------------------------------------
def read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


SRC_MODULES: list[tuple[str, str, str]] = [
    # (target path on Colab, repo-relative source, one-line purpose)
    ("/content/src/data/load_data.py",        "src/data/load_data.py",        "Thin SQL readers for the SQLite dataset."),
    ("/content/src/data/snapshot.py",         "src/data/snapshot.py",         "Per-project snapshot date selection."),
    ("/content/src/data/clean.py",            "src/data/clean.py",            "Path normalisation, bug-fix tagging, type coercion."),
    ("/content/src/data/szz.py",              "src/data/szz.py",              "SZZ / bug-fix / Jira helpers at basename granularity."),
    ("/content/src/data/labeling.py",         "src/data/labeling.py",         "Three-variant label computation + agreement metrics."),
    ("/content/src/features/static_features.py", "src/features/static_features.py", "Per-basename SonarQube + project-context features at t."),
    ("/content/src/features/historical_features.py", "src/features/historical_features.py", "Per-basename Git process features (pre-snapshot)."),
    ("/content/src/features/graph_features.py", "src/features/graph_features.py", "Co-change graph centrality features (Stage 5 enhancement)."),
    ("/content/src/features/priordefect_features.py", "src/features/priordefect_features.py", "Pre-snapshot bugfix / SZZ / Jira defect-history features."),
    ("/content/src/models/train.py",          "src/models/train.py",          "Within-project stratified K-fold + calibration + SMOTE."),
    ("/content/src/models/cross_project.py",  "src/models/cross_project.py",  "Leave-One-Project-Out cross-project validation."),
    ("/content/src/models/tuning.py",         "src/models/tuning.py",         "Optuna hyperparameter tuning (Stage 7b)."),
    ("/content/src/models/temporal.py",       "src/models/temporal.py",       "Temporal within-project T1->T2 split (Stage 7e)."),
    ("/content/src/analysis/sensitivity.py",  "src/analysis/sensitivity.py",  "Sensitivity grid for consequence labelling."),
    ("/content/src/analysis/ablation.py",     "src/analysis/ablation.py",     "Feature-group ablation."),
    ("/content/src/analysis/importance.py",   "src/analysis/importance.py",   "SHAP + permutation importance."),
    ("/content/src/analysis/significance.py", "src/analysis/significance.py", "Bootstrap CIs + Wilcoxon paired tests (Stage 7 inspect)."),
    ("/content/src/reporting/figures.py",     "src/reporting/figures.py",     "Publication-ready figures (PNG + PDF)."),
    ("/content/src/reporting/render.py",      "src/reporting/render.py",      "Render docs/06_results.md and 07_discussion.md from artefacts."),
]


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------
cells: list[dict] = []


# ===== 1. Front matter =====================================================
cells.append(md(
    "# TD Prediction Pipeline - Replication Notebook\n"
    "\n"
    "**Purpose.** Reproduces Stages 1-10 of the experiments described in the accompanying MSc thesis on High-Risk Technical Debt prediction. Given a copy of the Technical Debt Dataset v2.0 (`td_V2.db`) on Google Drive, running this notebook top-to-bottom regenerates every experimental artefact: cleaned parquet tables, label / feature / dataset matrices, within-project and cross-project results, sensitivity and ablation tables, SHAP-based feature-importance, all seven thesis figures, and the auto-generated `docs/06_results.md` / `docs/07_discussion.md`.\n"
    "\n"
    "**Author.** Abdulmajid Awol Seid  \n"
    "**Version.** 1.0 (2026-04-26)  \n"
    "**Scope.** Experimentation only. Hand-written thesis prose chapters are *not* produced or copied by this notebook.\n"
))

cells.append(md(
    "## Pipeline overview\n"
    "\n"
    "```mermaid\n"
    "flowchart TB\n"
    "    Front[Front matter and reproducibility setup] --> Cfg[Config cell] --> Mods[src/ modules writefile]\n"
    "    Mods --> S1[Stage 1: DB inspection]\n"
    "    S1 --> S2[Stage 2: Snapshot selection]\n"
    "    S2 --> S3[Stage 3: Clean and basename normalise]\n"
    "    S3 --> S4[Stage 4: Three-variant labelling and agreement]\n"
    "    S4 --> S5[Stage 5: Static and historical features]\n"
    "    S5 --> S6[Stage 6: Dataset assembly and leakage audit]\n"
    "    S6 --> S7[Stage 7: Within-project 10-fold CV]\n"
    "    S7 --> S8[Stage 8: LOPO cross-project CV]\n"
    "    S8 --> S9[Stage 9: Sensitivity grid and ablation]\n"
    "    S9 --> S10[Stage 10: SHAP, figures, results.md, discussion.md]\n"
    "    S10 --> Final[Runtime receipt and zip output to Drive]\n"
    "```\n"
    "\n"
    "### Runtime expectations (default Colab CPU runtime)\n"
    "\n"
    "| Stage | Description                                  | Wall-clock | Peak RAM |\n"
    "|-------|----------------------------------------------|------------|----------|\n"
    "| 1     | DB schema inventory                          | ~10 s      | < 1 GB   |\n"
    "| 2     | Per-project snapshot selection               | ~30 s      | < 1 GB   |\n"
    "| 3     | Clean and persist as parquet                 | ~3 min     | ~3 GB    |\n"
    "| 4     | Three-variant labelling                      | ~45 s      | ~2 GB    |\n"
    "| 5     | Static + historical feature extraction       | ~30 s      | ~2 GB    |\n"
    "| 6     | Dataset assembly + leakage audit             | ~5 s       | ~1 GB    |\n"
    "| 7     | Within-project 10-fold CV (5 models x 3)     | ~2 min     | ~2 GB    |\n"
    "| 8     | LOPO cross-project CV (5 models x 3)         | ~3 min     | ~2 GB    |\n"
    "| 9     | Sensitivity grid + ablation                  | ~3 min     | ~2 GB    |\n"
    "| 10    | SHAP + figures + render markdown reports     | ~3 min     | ~3 GB    |\n"
    "| **Total** |                                          | **~15 min** |          |\n"
))

cells.append(md(
    "## Data citation\n"
    "\n"
    "Cite the underlying dataset (Lenarduzzi, Saarimaki and Taibi 2019) when reusing any output of this notebook:\n"
    "\n"
    "```bibtex\n"
    "@inproceedings{Lenarduzzi2019TDDataset,\n"
    "  author    = {Lenarduzzi, Valentina and Saarim{\\\"a}ki, Nyyti and Taibi, Davide},\n"
    "  title     = {The Technical Debt Dataset},\n"
    "  booktitle = {Proceedings of the 15th International Conference on Predictive Models and Data Analytics in Software Engineering (PROMISE)},\n"
    "  year      = {2019},\n"
    "  doi       = {10.1145/3345629.3345630},\n"
    "  publisher = {ACM}\n"
    "}\n"
    "```\n"
    "\n"
    "Dataset URL: <https://github.com/clowee/The-Technical-Debt-Dataset/releases>\n"
    "\n"
    "## Table of contents\n"
    "\n"
    "1. [Reproducibility setup](#repro-setup)\n"
    "2. [Configuration](#configuration)\n"
    "3. [Source modules](#source-modules)\n"
    "4. [Stage 1 - Database inspection](#stage-1)\n"
    "5. [Stage 2 - Snapshot selection](#stage-2)\n"
    "6. [Stage 3 - Clean and basename-normalise](#stage-3)\n"
    "7. [Stage 4 - Three-variant labelling](#stage-4)\n"
    "8. [Stage 5 - Feature extraction](#stage-5)\n"
    "9. [Stage 6 - Dataset assembly](#stage-6)\n"
    "10. [Stage 7 - Within-project 10-fold CV](#stage-7)\n"
    "11. [Stage 8 - LOPO cross-project CV](#stage-8)\n"
    "12. [Collision analysis (construct validity)](#collision-analysis)\n"
    "13. [Stage 9 - Sensitivity and ablation](#stage-9)\n"
    "14. [Stage 10 - SHAP, figures and reports](#stage-10)\n"
    "15. [Finalisation - runtime, zip, copy to Drive](#finalisation)\n"
))


# ===== 2. Reproducibility setup ============================================
cells.append(md(
    '<a id="repro-setup"></a>\n'
    "## Reproducibility setup\n"
    "\n"
    "Five cells: (A) detect Colab and mount Drive; (B) verify the SQLite database; (C) install pinned dependencies; (D) set deterministic seeds and capture an environment receipt; (E) scaffold `/content/` and symlink the database into `data/raw/`.\n"
    "\n"
    "All cells are idempotent - re-running them does not duplicate work."
))

cells.append(code(
    "# Cell A: detect Colab and mount Drive (idempotent)\n"
    "import sys, os\n"
    "from pathlib import Path\n"
    "\n"
    "IN_COLAB = 'google.colab' in sys.modules\n"
    "if IN_COLAB:\n"
    "    from google.colab import drive\n"
    "    drive.mount('/content/drive', force_remount=False)\n"
    "    DRIVE_ROOT = Path('/content/drive/MyDrive/td_pipeline')\n"
    "else:\n"
    "    # Local fallback so the notebook is testable outside Colab\n"
    "    DRIVE_ROOT = Path.cwd() / 'td_pipeline_drive'\n"
    "DRIVE_ROOT.mkdir(parents=True, exist_ok=True)\n"
    "print('IN_COLAB =', IN_COLAB)\n"
    "print('DRIVE_ROOT =', DRIVE_ROOT)\n"
))

cells.append(code(
    "# Cell B: assert the SQLite database is on Drive and record an integrity\n"
    "# fingerprint (SHA-256 of the first 64 MiB - full hash on a 1.5 GB file is\n"
    "# unnecessarily expensive; the prefix is a sufficient sanity check).\n"
    "import hashlib\n"
    "\n"
    "DB_PATH = DRIVE_ROOT / 'td_V2.db'\n"
    "assert DB_PATH.exists(), (\n"
    "    f'Place td_V2.db in {DRIVE_ROOT} before running this notebook.\\n'\n"
    "    'Download from https://github.com/clowee/The-Technical-Debt-Dataset/releases'\n"
    ")\n"
    "size_mb = DB_PATH.stat().st_size / (1024 ** 2)\n"
    "print(f'td_V2.db size: {size_mb:.1f} MB')\n"
    "\n"
    "h = hashlib.sha256()\n"
    "with open(DB_PATH, 'rb') as f:\n"
    "    bytes_hashed = 0\n"
    "    while bytes_hashed < 64 * 1024 * 1024:\n"
    "        chunk = f.read(1024 * 1024)\n"
    "        if not chunk:\n"
    "            break\n"
    "        h.update(chunk)\n"
    "        bytes_hashed += len(chunk)\n"
    "DB_FINGERPRINT = h.hexdigest()[:16]\n"
    "print(f'SHA256 prefix (first {bytes_hashed // (1024*1024)} MiB): {DB_FINGERPRINT}')\n"
))

cells.append(code(
    "# Cell C: install pinned dependencies. Skipped silently when each package\n"
    "# is already importable, so re-runs are nearly instant.\n"
    "import importlib.util\n"
    "\n"
    "REQUIRED = [\n"
    "    ('pandas', 'pandas>=2.0.0'),\n"
    "    ('numpy', 'numpy>=1.24.0'),\n"
    "    ('sklearn', 'scikit-learn>=1.3.0'),\n"
    "    ('xgboost', 'xgboost>=2.0.0'),\n"
    "    ('lightgbm', 'lightgbm>=4.0.0'),\n"
    "    ('imblearn', 'imbalanced-learn>=0.11.0'),\n"
    "    ('matplotlib', 'matplotlib>=3.7.0'),\n"
    "    ('seaborn', 'seaborn>=0.12.0'),\n"
    "    ('pyarrow', 'pyarrow>=14.0.0'),\n"
    "    ('shap', 'shap>=0.44.0'),\n"
    "    ('matplotlib_venn', 'matplotlib-venn>=0.11.9'),\n"
    "    ('tabulate', 'tabulate>=0.9.0'),\n"
    "    ('scipy', 'scipy>=1.11.0'),\n"
    "    ('tqdm', 'tqdm>=4.65.0'),\n"
    "    # Graph centralities. Use the modern PyPI name 'igraph' (NOT 'python-igraph',\n"
    "    # which is the deprecated name and fails on Colab's pip).\n"
    "    ('igraph', 'igraph>=0.11'),\n"
    "    ('networkx', 'networkx>=3.2'),\n"
    "    ('joblib', 'joblib>=1.3.0'),\n"
    "    ('optuna', 'optuna>=3.5.0'),\n"
    "]\n"
    "missing = [pin for mod, pin in REQUIRED if importlib.util.find_spec(mod) is None]\n"
    "if missing:\n"
    "    print('Installing missing packages:', missing)\n"
    "    import subprocess\n"
    "    subprocess.run(\n"
    "        [sys.executable, '-m', 'pip', 'install', '-q', *missing],\n"
    "        check=True,\n"
    "    )\n"
    "else:\n"
    "    print('All dependencies already satisfied; skipping pip install.')\n"
))

cells.append(code(
    "# Cell D: deterministic seeds + environment receipt for the appendix.\n"
    "import json, platform, random, datetime\n"
    "import numpy as np\n"
    "import importlib.metadata as md_meta\n"
    "\n"
    "RANDOM_STATE = 42\n"
    "random.seed(RANDOM_STATE)\n"
    "np.random.seed(RANDOM_STATE)\n"
    "os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)\n"
    "\n"
    "_pkgs = ['pandas', 'numpy', 'scikit-learn', 'xgboost', 'lightgbm', 'shap',\n"
    "         'matplotlib', 'seaborn', 'pyarrow', 'tabulate', 'matplotlib-venn',\n"
    "         'imbalanced-learn', 'tqdm', 'scipy']\n"
    "_versions = {}\n"
    "for p in _pkgs:\n"
    "    try:\n"
    "        _versions[p] = md_meta.version(p)\n"
    "    except md_meta.PackageNotFoundError:\n"
    "        _versions[p] = None\n"
    "\n"
    "ENV_RECEIPT = {\n"
    "    'timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',\n"
    "    'python_version': sys.version.split()[0],\n"
    "    'platform': platform.platform(),\n"
    "    'in_colab': IN_COLAB,\n"
    "    'random_state': RANDOM_STATE,\n"
    "    'db_fingerprint_sha256_prefix': DB_FINGERPRINT,\n"
    "    'db_size_mb': round(size_mb, 1),\n"
    "    'package_versions': _versions,\n"
    "}\n"
    "print(json.dumps(ENV_RECEIPT, indent=2))\n"
))

cells.append(code(
    "# Cell E: scaffold /content and copy the SQLite database off Drive.\n"
    "#\n"
    "# Why a true copy and not a symlink: when the SQLite file is read through\n"
    "# Drive's FUSE mount every page request becomes a network round-trip.\n"
    "# On a 1.54 GB DB this triggers Drive's I/O quota and frequently kills\n"
    "# the Colab runtime mid-stage (the 'runtime disconnected' symptom).\n"
    "# A true copy moves the file onto Colab's local SSD, where reads are\n"
    "# 30-50 times faster and never touch the network again. The copy is\n"
    "# idempotent: if a previous run already wrote the file with a matching\n"
    "# SHA-256 prefix, the cell skips the copy.\n"
    "import shutil, hashlib, os\n"
    "from tqdm.auto import tqdm\n"
    "\n"
    "CONTENT = Path('/content') if IN_COLAB else (Path.cwd() / 'colab_workdir')\n"
    "CONTENT.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "for sub in [\n"
    "    'src/data', 'src/features', 'src/models', 'src/analysis', 'src/reporting',\n"
    "    'tools',\n"
    "    'data/raw', 'data/processed', 'data/external',\n"
    "    'results/tables', 'results/figures', 'results/run_logs',\n"
    "    'docs',\n"
    "]:\n"
    "    (CONTENT / sub).mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "# Empty package markers so `from src.data import ...` works\n"
    "for pkg_dir in ['src', 'src/data', 'src/features', 'src/models', 'src/analysis', 'src/reporting']:\n"
    "    init = CONTENT / pkg_dir / '__init__.py'\n"
    "    if not init.exists():\n"
    "        init.write_text('', encoding='utf-8')\n"
    "\n"
    "DB_LOCAL = CONTENT / 'data' / 'raw' / 'td_V2.db'\n"
    "\n"
    "def _sha256_prefix(path, n_mib=64):\n"
    "    h = hashlib.sha256()\n"
    "    target = n_mib * 1024 * 1024\n"
    "    read = 0\n"
    "    with open(path, 'rb') as fh:\n"
    "        while read < target:\n"
    "            chunk = fh.read(1024 * 1024)\n"
    "            if not chunk:\n"
    "                break\n"
    "            h.update(chunk); read += len(chunk)\n"
    "    return h.hexdigest()[:16]\n"
    "\n"
    "def _copy_with_progress(src, dst, attempt):\n"
    "    total = src.stat().st_size\n"
    "    chunk = 16 * 1024 * 1024  # 16 MiB; FUSE copes well with this size\n"
    "    with open(src, 'rb') as fin, open(dst, 'wb') as fout, \\\n"
    "         tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024,\n"
    "              desc=f'Copy DB attempt {attempt}', leave=False) as bar:\n"
    "        while True:\n"
    "            buf = fin.read(chunk)\n"
    "            if not buf:\n"
    "                break\n"
    "            fout.write(buf); bar.update(len(buf))\n"
    "\n"
    "if DB_LOCAL.exists() and _sha256_prefix(DB_LOCAL) == DB_FINGERPRINT:\n"
    "    print(f'DB already copied locally at {DB_LOCAL} (SHA matches Drive copy).')\n"
    "else:\n"
    "    last_exc = None\n"
    "    for _attempt in range(1, 4):\n"
    "        try:\n"
    "            _copy_with_progress(DB_PATH, DB_LOCAL, _attempt)\n"
    "            local_sha = _sha256_prefix(DB_LOCAL)\n"
    "            if local_sha != DB_FINGERPRINT:\n"
    "                raise OSError(f'SHA mismatch after copy: {local_sha} vs {DB_FINGERPRINT}')\n"
    "            print(f'DB copied to local SSD at {DB_LOCAL}; SHA prefix verified.')\n"
    "            break\n"
    "        except (OSError, FileNotFoundError) as _exc:\n"
    "            last_exc = _exc\n"
    "            print(f'  copy attempt {_attempt} failed: {_exc}; retrying after 5s...')\n"
    "            import time as _time\n"
    "            _time.sleep(5)\n"
    "    else:\n"
    "        raise RuntimeError(\n"
    "            f'Failed to copy {DB_PATH} to {DB_LOCAL} after 3 attempts: {last_exc}'\n"
    "        )\n"
    "\n"
    "# config.py resolves to RAW_DATA_DIR / 'td_V2.db' which is exactly DB_LOCAL,\n"
    "# so no env-var override is needed.\n"
    "\n"
    "if str(CONTENT) not in sys.path:\n"
    "    sys.path.insert(0, str(CONTENT))\n"
    "\n"
    "os.environ['TD_REPO_ROOT'] = str(CONTENT.resolve())\n"
    "\n"
    "# Persist the env receipt now that results/ exists\n"
    "(CONTENT / 'results' / 'env_receipt.json').write_text(\n"
    "    json.dumps(ENV_RECEIPT, indent=2), encoding='utf-8'\n"
    ")\n"
    "\n"
    "# Wall-clock accumulator used by every stage\n"
    "RUNTIMES: dict[str, float] = {}\n"
    "RECOMPUTE = False  # set True to force re-execution of cached stages\n"
    "\n"
    "print('Working tree at:', CONTENT)\n"
))

cells.append(md(
    "## Runtime tier detection and parallelism budget\n"
    "\n"
    "Colab offers three free/paid tiers with very different memory and CPU budgets:\n"
    "\n"
    "| Tier        | RAM   | CPUs | Recommended `TD_N_JOBS` |\n"
    "|-------------|------:|-----:|------------------------:|\n"
    "| Free (T4)   | 12 GB | 2    | 1                       |\n"
    "| Pro         | 25 GB | 4    | 2                       |\n"
    "| Pro+ / High-RAM | 51 GB | 8 | -1 (use all)            |\n"
    "\n"
    "The next cell auto-detects the tier and sets `TD_N_JOBS` so the pipeline does not oversubscribe a 2-core free VM. Stage 5 (graph features) and Stage 7e (temporal) honour this budget at the project-loop level; inner model fits stay at `n_jobs=1` to avoid CPU thrash."
))

cells.append(code(
    "# Cell F: tier-aware parallelism budget for stages 5 and 7e.\n"
    "try:\n"
    "    import psutil\n"
    "    _ram_gb = psutil.virtual_memory().total / (1024 ** 3)\n"
    "except ImportError:\n"
    "    # /proc/meminfo fallback\n"
    "    try:\n"
    "        with open('/proc/meminfo') as _mf:\n"
    "            for _line in _mf:\n"
    "                if _line.startswith('MemTotal:'):\n"
    "                    _ram_kb = int(_line.split()[1]); _ram_gb = _ram_kb / (1024 ** 2); break\n"
    "            else:\n"
    "                _ram_gb = 0.0\n"
    "    except FileNotFoundError:\n"
    "        _ram_gb = 0.0\n"
    "_n_cpu = os.cpu_count() or 2\n"
    "if _ram_gb >= 40 and _n_cpu >= 8:\n"
    "    _tier = 'pro+'; _jobs = -1\n"
    "elif _ram_gb >= 20 and _n_cpu >= 4:\n"
    "    _tier = 'pro'; _jobs = 2\n"
    "else:\n"
    "    _tier = 'free'; _jobs = 1\n"
    "os.environ['TD_N_JOBS'] = str(_jobs)\n"
    "print(f'Detected tier: {_tier:<5}  RAM={_ram_gb:.1f} GB  CPUs={_n_cpu}  TD_N_JOBS={_jobs}')\n"
    "if _tier == 'free':\n"
    "    print('NOTE: free tier uses TD_N_JOBS=1 to avoid OOM on the largest project (hive, 5307 basenames).')\n"
    "    print('      Stages 8 (LOPO) and 9 (sensitivity grid) remain enabled but will be slower.')\n"
))

cells.append(md(
    "## Drive checkpointing helper\n"
    "\n"
    "Every stage cell starts by checking whether its output artefacts already exist either locally or in `MyDrive/td_pipeline/checkpoints/<stageNN>/`. If found, the cell prints a *skipping* banner and copies the cached artefacts into `/content/` so downstream stages can use them.\n"
    "\n"
    "After every successful stage, the cell rsync's its outputs back to Drive. The net effect: a runtime disconnect at, say, stage 7c only loses stage-7c work; reconnecting and re-running the notebook resumes from stage 7c with no data loss.\n"
))

cells.append(code(
    "# Cell G: stage checkpoint helpers (idempotent stage-skip + Drive backup).\n"
    "CHECKPOINTS = DRIVE_ROOT / 'checkpoints'\n"
    "CHECKPOINTS.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "def stage_outputs_exist(*paths):\n"
    "    \"\"\"True iff every relative-to-CONTENT path exists either locally or in the\n"
    "    Drive checkpoint folder for that stage. Local paths are restored from\n"
    "    Drive when missing.\"\"\"\n"
    "    all_present = True\n"
    "    for rel in paths:\n"
    "        local = CONTENT / rel\n"
    "        if local.exists():\n"
    "            continue\n"
    "        backup = CHECKPOINTS / rel\n"
    "        if backup.exists():\n"
    "            local.parent.mkdir(parents=True, exist_ok=True)\n"
    "            shutil.copy2(backup, local)\n"
    "            print(f'  restored from Drive: {rel}')\n"
    "        else:\n"
    "            all_present = False\n"
    "    return all_present\n"
    "\n"
    "def checkpoint_to_drive(*paths):\n"
    "    \"\"\"Mirror local stage outputs back to Drive so a runtime disconnect cannot\n"
    "    discard them.\"\"\"\n"
    "    for rel in paths:\n"
    "        local = CONTENT / rel\n"
    "        if not local.exists():\n"
    "            continue\n"
    "        backup = CHECKPOINTS / rel\n"
    "        backup.parent.mkdir(parents=True, exist_ok=True)\n"
    "        shutil.copy2(local, backup)\n"
    "    print(f'  checkpointed {len(paths)} artefact(s) to Drive: {CHECKPOINTS}')\n"
    "\n"
    "print(f'Checkpoint store: {CHECKPOINTS}')\n"
))

cells.append(md(
    "## Anti-disconnect keep-alive (optional, recommended for the full notebook)\n"
    "\n"
    "Colab disconnects idle browser tabs after about 90 minutes; if your wall-clock for the full pipeline exceeds that, the runtime will be recycled and you must reconnect manually. Two mitigations:\n"
    "\n"
    "1. **Browser-side keep-alive (preferred for the *full* notebook).** Open the browser developer console (F12 -> Console) and paste this snippet, which clicks the Connect button every 60 seconds:\n"
    "\n"
    "   ```javascript\n"
    "   function _td_keepAlive() {\n"
    "     const btn = document.querySelector('colab-connect-button');\n"
    "     if (btn) btn.click();\n"
    "   }\n"
    "   setInterval(_td_keepAlive, 60 * 1000);\n"
    "   ```\n"
    "\n"
    "2. **Notebook-side ping (works without dev console).** The next cell schedules an in-kernel ping every five minutes, which is sufficient for most idle-disconnect scenarios.\n"
    "\n"
    "If the runtime *does* disconnect, simply reconnect and re-run from the top - every previously completed stage is restored from Drive and skipped automatically thanks to the checkpoint helper above."
))

cells.append(code(
    "# Cell H: notebook-side keep-alive ping (optional). Comment out for short runs.\n"
    "from IPython.display import Javascript, display\n"
    "if IN_COLAB:\n"
    "    display(Javascript('''\n"
    "        if (typeof window._tdKeepAliveTimer === 'undefined') {\n"
    "          window._tdKeepAliveTimer = setInterval(function() {\n"
    "            const btn = document.querySelector('colab-connect-button');\n"
    "            if (btn) btn.click();\n"
    "          }, 5 * 60 * 1000);\n"
    "          console.log('TD pipeline keep-alive installed (5 min interval).');\n"
    "        }\n"
    "    '''))\n"
    "    print('Keep-alive installed.')\n"
    "else:\n"
    "    print('Local run; keep-alive not needed.')\n"
))


# ===== 3. Configuration ====================================================
cells.append(md(
    '<a id="configuration"></a>\n'
    "## Configuration\n"
    "\n"
    "The next cell writes `config.py` to `/content/`. It is the single source of truth for paths, snapshot policy, labelling parameters, feature catalogues and model definitions.\n"
    "\n"
    "Reviewers wishing to vary the experiment without editing source files should change one of the parameters below in the **config cell itself** and re-run from Stage 4 onwards (Stages 1-3 do not depend on these):\n"
    "\n"
    "- `OBSERVATION_WINDOW_MONTHS` (primary 6) - length of the post-snapshot window for label derivation.\n"
    "- `HIGH_RISK_PERCENTILE` (primary 20) - top P% by risk score flagged as positive.\n"
    "- `MIN_PRE_SNAPSHOT_COMMITS` (500) and `MIN_POST_SNAPSHOT_COMMITS` (50) - eligibility thresholds.\n"
    "- `SENSITIVITY_WINDOWS` and `SENSITIVITY_PERCENTILES` - grids used by Stage 9.\n"
    "- `RISK_SCORE_WEIGHTS` - weighting between bug-fix commits, future churn and SZZ events in the consequence risk score.\n"
    "- `MODELS` and `RANDOM_STATE` - controlled training settings.\n"
    "\n"
    "After editing, save and continue running the notebook; the change propagates because every downstream cell imports from this `config.py`.\n"
))

cells.append(writefile_cell("/content/config.py", read("config.py")))


# ===== 4. Source modules ===================================================
cells.append(md(
    '<a id="source-modules"></a>\n'
    "## Source modules\n"
    "\n"
    "The next 14 cells materialise the production code modules under `/content/src/` using `%%writefile`. Module contents are *verbatim* copies of the repository sources - they are the same code that the local `scripts/01..10` use, ensuring byte-identical experimental results.\n"
    "\n"
    "Each cell is preceded by a one-line caption stating the file path and purpose. After all 14 modules are written, a final cell imports them so subsequent stage cells can call them directly.\n"
))

for target, rel, purpose in SRC_MODULES:
    cells.append(md(f"**`{rel}`** - {purpose}"))
    cells.append(writefile_cell(target, read(rel)))

cells.append(code(
    "# Final modules cell - import everything so stage drivers can use it.\n"
    "import importlib\n"
    "\n"
    "# Force a fresh import in case the cell is re-run after editing a writefile.\n"
    "for name in list(sys.modules):\n"
    "    if name == 'config' or name.startswith('src.'):\n"
    "        del sys.modules[name]\n"
    "\n"
    "import config  # noqa: F401\n"
    "from src.data import load_data, snapshot, clean, szz, labeling  # noqa: F401\n"
    "from src.features import (  # noqa: F401\n"
    "    static_features, historical_features, graph_features, priordefect_features,\n"
    ")\n"
    "from src.models import train, cross_project, tuning, temporal  # noqa: F401\n"
    "from src.analysis import sensitivity, ablation, importance, significance  # noqa: F401\n"
    "from src.reporting import figures, render  # noqa: F401\n"
    "\n"
    "print('All src/ modules imported.')\n"
    "print('  config.PROCESSED_DATA_DIR =', config.PROCESSED_DATA_DIR)\n"
    "print('  config.TD_DATASET_PATH    =', config.TD_DATASET_PATH)\n"
))


# ===== 5. Stage 1 ===========================================================
cells.append(md(
    '<a id="stage-1"></a>\n'
    "## Stage 1 - Database inspection\n"
    "\n"
    "Open `td_V2.db`, list every table, record per-column metadata via `PRAGMA table_info`, and persist a 3-row sample per table. This is the canonical schema record referenced by every subsequent stage. No data transformation happens here.\n"
    "\n"
    "Outputs:\n"
    "- `results/tables/db_schema.csv` - one row per (table, column).\n"
    "- `results/tables/db_table_counts.csv` - per-table row counts.\n"
    "- `results/tables/db_samples/<table>.csv` - first 3 rows per table.\n"
))

cells.append(code(
    "import time\n"
    "import pandas as pd\n"
    "from src.data.load_data import (\n"
    "    get_connection, get_row_count, get_sample_rows,\n"
    "    get_table_names, get_table_schema,\n"
    ")\n"
    "from config import TABLES_DIR, TD_DATASET_PATH\n"
    "\n"
    "_t0 = time.time()\n"
    "_samples_dir = TABLES_DIR / 'db_samples'\n"
    "_samples_dir.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "with get_connection() as conn:\n"
    "    _tables = get_table_names(conn)\n"
    "    print(f'[Stage 1] Tables found: {len(_tables)}')\n"
    "    _schema_rows, _count_rows = [], []\n"
    "    for _t in _tables:\n"
    "        _count = get_row_count(conn, _t)\n"
    "        _count_rows.append({'table': _t, 'row_count': _count})\n"
    "        _schema = get_table_schema(conn, _t)\n"
    "        _sample = get_sample_rows(conn, _t, 3)\n"
    "        _sample.to_csv(_samples_dir / f'{_t}.csv', index=False)\n"
    "        _first = _sample.iloc[0] if len(_sample) else pd.Series(dtype=object)\n"
    "        for _, _col in _schema.iterrows():\n"
    "            _example = _first.get(_col['name'], None) if len(_first) else None\n"
    "            if isinstance(_example, str) and len(_example) > 120:\n"
    "                _example = _example[:117] + '...'\n"
    "            _schema_rows.append({\n"
    "                'table': _t, 'column': _col['name'], 'type': _col['type'],\n"
    "                'notnull': bool(_col['notnull']), 'pk': bool(_col['pk']),\n"
    "                'example_value': _example,\n"
    "            })\n"
    "\n"
    "_schema_df = pd.DataFrame(_schema_rows)\n"
    "_counts_df = pd.DataFrame(_count_rows).sort_values('row_count', ascending=False)\n"
    "_schema_df.to_csv(TABLES_DIR / 'db_schema.csv', index=False)\n"
    "_counts_df.to_csv(TABLES_DIR / 'db_table_counts.csv', index=False)\n"
    "\n"
    "RUNTIMES['stage_01_inspect'] = round(time.time() - _t0, 2)\n"
    "print(f'[Stage 1] Elapsed: {RUNTIMES[\"stage_01_inspect\"]} s')\n"
))

cells.append(code(
    "# Inspect Stage 1 outputs.\n"
    "from IPython.display import display, Markdown\n"
    "\n"
    "_counts = pd.read_csv(TABLES_DIR / 'db_table_counts.csv')\n"
    "display(Markdown('**Row counts per table:**'))\n"
    "display(_counts)\n"
    "\n"
    "_schema = pd.read_csv(TABLES_DIR / 'db_schema.csv')\n"
    "display(Markdown(f'**Schema: {_schema[\"table\"].nunique()} tables, '\n"
    "                 f'{len(_schema)} columns total.** First 12 columns shown:'))\n"
    "display(_schema.head(12))\n"
))


# ===== 6. Stage 2 ===========================================================
cells.append(md(
    '<a id="stage-2"></a>\n'
    "## Stage 2 - Per-project snapshot selection\n"
    "\n"
    "For every project, the master-branch commit history is loaded and a snapshot date `t` is selected as the median commit date. Pre-snapshot history feeds features; the post-snapshot 6-month window feeds labels. Projects without enough history on either side are flagged ineligible.\n"
    "\n"
    "Eligibility thresholds: `MIN_PRE_SNAPSHOT_COMMITS = 500`, `MIN_POST_SNAPSHOT_COMMITS = 50`.\n"
    "\n"
    "Outputs:\n"
    "- `data/processed/project_snapshots.parquet` - full per-project metadata.\n"
    "- `results/tables/project_stats.csv` - same data as CSV.\n"
))

cells.append(code(
    "from src.data.load_data import get_connection\n"
    "from src.data.snapshot import compute_all_snapshots\n"
    "from config import PROCESSED_DATA_DIR, TABLES_DIR, OBSERVATION_WINDOW_MONTHS, SNAPSHOT_STRATEGY\n"
    "\n"
    "_snap_path = PROCESSED_DATA_DIR / 'project_snapshots.parquet'\n"
    "if _snap_path.exists() and not RECOMPUTE:\n"
    "    print(f'[Stage 2] Cached: {_snap_path} (set RECOMPUTE=True to regenerate)')\n"
    "    snapshots_df = pd.read_parquet(_snap_path)\n"
    "    RUNTIMES['stage_02_snapshot'] = 0.0\n"
    "else:\n"
    "    _t0 = time.time()\n"
    "    print(f'[Stage 2] Strategy={SNAPSHOT_STRATEGY}, window={OBSERVATION_WINDOW_MONTHS} mo')\n"
    "    with get_connection() as conn:\n"
    "        snapshots_df = compute_all_snapshots(conn)\n"
    "    snapshots_df.to_parquet(_snap_path, index=False)\n"
    "    snapshots_df.to_csv(TABLES_DIR / 'project_stats.csv', index=False)\n"
    "    RUNTIMES['stage_02_snapshot'] = round(time.time() - _t0, 2)\n"
    "    print(f'[Stage 2] Elapsed: {RUNTIMES[\"stage_02_snapshot\"]} s')\n"
))

cells.append(code(
    "# Inspect snapshot eligibility.\n"
    "_eligible = snapshots_df[snapshots_df['eligible']].copy()\n"
    "_excluded = snapshots_df[~snapshots_df['eligible']].copy()\n"
    "display(Markdown(\n"
    "    f'**Eligibility:** {len(_eligible)} of {len(snapshots_df)} projects.  '\n"
    "    f'Excluded: {len(_excluded)} (reasons: '\n"
    "    f\"{sorted(_excluded['exclusion_reason'].dropna().unique().tolist())}).\"\n"
    "))\n"
    "_disp_cols = [\n"
    "    'project_id', 'first_commit', 'last_commit', 'snapshot_date',\n"
    "    'total_commits', 'pre_snapshot_commits', 'post_snapshot_commits',\n"
    "    'distinct_files_pre', 'distinct_authors_pre',\n"
    "]\n"
    "display(_eligible[_disp_cols].sort_values('total_commits', ascending=False).reset_index(drop=True))\n"
))


# ===== 7. Stage 3 ===========================================================
cells.append(md(
    '<a id="stage-3"></a>\n'
    "## Stage 3 - Clean and basename-normalise\n"
    "\n"
    "Convert the raw SQLite tables into tidy parquet files restricted to eligible projects and Java source files (excluding tests, generated code and build artefacts). Path normalisation extracts the **basename** from `SONAR_ISSUES.COMPONENT` and `GIT_COMMITS_CHANGES.FILE` so the two tables can join on `(project_id, basename)` (the canonical unit of analysis - documented in the thesis Threats to Construct Validity).\n"
    "\n"
    "Outputs:\n"
    "- 6 cleaned parquets in `data/processed/clean_*.parquet`.\n"
    "- `results/tables/path_overlap_report.csv` - per-project basename coverage and collision-rate sanity check.\n"
))

cells.append(code(
    "from src.data.clean import (\n"
    "    clean_git_commits, clean_git_commits_changes, clean_jira_issues,\n"
    "    clean_sonar_issues, clean_sonar_measures_with_dates, clean_szz_with_dates,\n"
    ")\n"
    "from src.data.snapshot import load_snapshots\n"
    "\n"
    "_required = [\n"
    "    'clean_git_commits.parquet', 'clean_git_commits_changes.parquet',\n"
    "    'clean_sonar_issues.parquet', 'clean_sonar_measures.parquet',\n"
    "    'clean_szz.parquet', 'clean_jira_issues.parquet',\n"
    "]\n"
    "_all_present = all((PROCESSED_DATA_DIR / p).exists() for p in _required)\n"
    "if _all_present and not RECOMPUTE:\n"
    "    print('[Stage 3] Cached cleaned parquets present; skipping.')\n"
    "    RUNTIMES['stage_03_clean'] = 0.0\n"
    "else:\n"
    "    _t0 = time.time()\n"
    "    _snaps = load_snapshots(PROCESSED_DATA_DIR / 'project_snapshots.parquet')\n"
    "    _projects = _snaps[_snaps['eligible']]['project_id'].tolist()\n"
    "    print(f'[Stage 3] Cleaning {len(_projects)} eligible projects ...')\n"
    "    with get_connection() as conn:\n"
    "        gc = clean_git_commits(conn, _projects)\n"
    "        print(f'  GIT_COMMITS:           {len(gc):>12,}')\n"
    "        gcc = clean_git_commits_changes(conn, _projects, java_only=True)\n"
    "        print(f'  GIT_COMMITS_CHANGES:   {len(gcc):>12,} (Java only)')\n"
    "        si = clean_sonar_issues(conn, _projects, java_only=True)\n"
    "        print(f'  SONAR_ISSUES:          {len(si):>12,} (Java only)')\n"
    "        sm = clean_sonar_measures_with_dates(conn, _projects)\n"
    "        print(f'  SONAR_MEASURES (dates):{len(sm):>12,}')\n"
    "        szz_df = clean_szz_with_dates(conn, _projects, gc)\n"
    "        print(f'  SZZ (dates):           {len(szz_df):>12,}')\n"
    "        ji = clean_jira_issues(conn, _projects)\n"
    "        print(f'  JIRA_ISSUES:           {len(ji):>12,}')\n"
    "\n"
    "    gc.to_parquet(PROCESSED_DATA_DIR / 'clean_git_commits.parquet', index=False)\n"
    "    gcc.to_parquet(PROCESSED_DATA_DIR / 'clean_git_commits_changes.parquet', index=False)\n"
    "    si.to_parquet(PROCESSED_DATA_DIR / 'clean_sonar_issues.parquet', index=False)\n"
    "    sm.to_parquet(PROCESSED_DATA_DIR / 'clean_sonar_measures.parquet', index=False)\n"
    "    szz_df.to_parquet(PROCESSED_DATA_DIR / 'clean_szz.parquet', index=False)\n"
    "    ji.to_parquet(PROCESSED_DATA_DIR / 'clean_jira_issues.parquet', index=False)\n"
    "\n"
    "    # Basename-overlap sanity check\n"
    "    _gcc_paths = gcc.groupby('PROJECT_ID')['basename'].agg(set).rename('git_basenames')\n"
    "    _si_paths = si.groupby('PROJECT_ID')['basename'].agg(set).rename('sonar_basenames')\n"
    "    _overlap = pd.concat([_gcc_paths, _si_paths], axis=1)\n"
    "    for _c in ('git_basenames', 'sonar_basenames'):\n"
    "        _overlap[_c] = _overlap[_c].apply(lambda x: x if isinstance(x, set) else set())\n"
    "    _overlap['n_sonar_basenames'] = _overlap['sonar_basenames'].map(len)\n"
    "    _overlap['n_git_basenames'] = _overlap['git_basenames'].map(len)\n"
    "    _overlap['n_intersection'] = _overlap.apply(\n"
    "        lambda r: len(r['sonar_basenames'] & r['git_basenames']), axis=1)\n"
    "    _overlap['sonar_coverage_pct'] = (\n"
    "        _overlap['n_intersection'] / _overlap['n_sonar_basenames'].where(_overlap['n_sonar_basenames'] > 0)\n"
    "    ).round(4) * 100\n"
    "    _overlap = _overlap[['n_sonar_basenames', 'n_git_basenames', 'n_intersection', 'sonar_coverage_pct']]\n"
    "    _coll_rows = []\n"
    "    for _pid, _sub in si.groupby('PROJECT_ID'):\n"
    "        _files_per_base = _sub.groupby('basename')['file_path'].nunique()\n"
    "        _coll_rows.append({\n"
    "            'PROJECT_ID': _pid,\n"
    "            'sonar_full_paths': int(_sub['file_path'].nunique()),\n"
    "            'sonar_basenames': int(_files_per_base.shape[0]),\n"
    "            'max_files_per_basename': int(_files_per_base.max()) if len(_files_per_base) else 0,\n"
    "            'basename_collision_pct': (\n"
    "                round((1 - _files_per_base.shape[0] / _sub['file_path'].nunique()) * 100, 1)\n"
    "                if _sub['file_path'].nunique() else 0.0\n"
    "            ),\n"
    "        })\n"
    "    _collisions = pd.DataFrame(_coll_rows).set_index('PROJECT_ID')\n"
    "    _overlap = _overlap.join(_collisions)\n"
    "    _overlap.reset_index().to_csv(TABLES_DIR / 'path_overlap_report.csv', index=False)\n"
    "\n"
    "    RUNTIMES['stage_03_clean'] = round(time.time() - _t0, 2)\n"
    "    print(f'[Stage 3] Elapsed: {RUNTIMES[\"stage_03_clean\"]} s')\n"
))

cells.append(code(
    "# Inspect Stage 3 cleaned tables and basename overlap.\n"
    "_overlap = pd.read_csv(TABLES_DIR / 'path_overlap_report.csv')\n"
    "_min = _overlap['sonar_coverage_pct'].min()\n"
    "_avg = _overlap['sonar_coverage_pct'].mean()\n"
    "_med_coll = _overlap['basename_collision_pct'].median()\n"
    "_max_coll = _overlap['basename_collision_pct'].max()\n"
    "display(Markdown(\n"
    "    '**Sonar -> Git basename coverage:** '\n"
    "    f'min={_min:.1f}% / avg={_avg:.1f}%.  '\n"
    "    f'**Basename collision (multiple full-paths -> same basename):** '\n"
    "    f'median={_med_coll:.1f}% / max={_max_coll:.1f}%.'\n"
    "))\n"
    "display(_overlap.sort_values('sonar_coverage_pct', ascending=False).reset_index(drop=True))\n"
    "\n"
    "_clean_summary = pd.DataFrame([\n"
    "    {'parquet': p, 'rows': len(pd.read_parquet(PROCESSED_DATA_DIR / p))}\n"
    "    for p in [\n"
    "        'clean_git_commits.parquet', 'clean_git_commits_changes.parquet',\n"
    "        'clean_sonar_issues.parquet', 'clean_sonar_measures.parquet',\n"
    "        'clean_szz.parquet', 'clean_jira_issues.parquet',\n"
    "    ]\n"
    "])\n"
    "display(Markdown('**Cleaned parquet row counts:**'))\n"
    "display(_clean_summary)\n"
))


# ===== 8. Stage 4 ===========================================================
cells.append(md(
    '<a id="stage-4"></a>\n'
    "## Stage 4 - Three-variant labelling\n"
    "\n"
    "Materialise three label variants per (project, basename):\n"
    "\n"
    "- **Consequence (primary)** - top 20% of basenames within each project by a weighted risk score combining future bug-fix commits (0.5), future churn (0.3) and SZZ events (0.2) within the 6-month observation window.\n"
    "- **Severity** - any open SonarQube `BLOCKER` or `CRITICAL` issue at `t`.\n"
    "- **SZZ** - touched by at least one fault-fixing commit in the 6-month window.\n"
    "\n"
    "Pairwise Cohen's kappa and Jaccard similarity are computed to quantify how disjoint the three variants are.\n"
    "\n"
    "Outputs: `data/processed/labels_{consequence,severity,szz}.parquet`, `results/tables/label_summary.csv`, `results/tables/label_agreement.csv`.\n"
))

cells.append(code(
    "from src.data.labeling import (\n"
    "    compute_consequence_labels, compute_severity_labels, compute_szz_labels, label_agreement,\n"
    ")\n"
    "from config import OBSERVATION_WINDOW_MONTHS, HIGH_RISK_PERCENTILE, SEVERITY_BASELINE_LEVELS\n"
    "from tqdm.auto import tqdm\n"
    "\n"
    "def _load_clean_for_labeling():\n"
    "    _commits = pd.read_parquet(PROCESSED_DATA_DIR / 'clean_git_commits.parquet')\n"
    "    _changes = pd.read_parquet(PROCESSED_DATA_DIR / 'clean_git_commits_changes.parquet')\n"
    "    _sonar_issues = pd.read_parquet(PROCESSED_DATA_DIR / 'clean_sonar_issues.parquet')\n"
    "    _szz = pd.read_parquet(PROCESSED_DATA_DIR / 'clean_szz.parquet')\n"
    "    _jira = pd.read_parquet(PROCESSED_DATA_DIR / 'clean_jira_issues.parquet')\n"
    "    for _df, _cols in (\n"
    "        (_commits, ['AUTHOR_DATE', 'COMMITTER_DATE']),\n"
    "        (_changes, ['DATE']),\n"
    "        (_sonar_issues, ['CREATION_DATE', 'CLOSE_DATE']),\n"
    "        (_szz, ['fix_date', 'induce_date']),\n"
    "        (_jira, ['CREATION_DATE', 'RESOLUTION_DATE', 'UPDATE_DATE', 'COMMIT_DATE']),\n"
    "    ):\n"
    "        for _c in _cols:\n"
    "            if _c in _df.columns and _df[_c].dtype.kind == 'M' and _df[_c].dt.tz is None:\n"
    "                _df[_c] = _df[_c].dt.tz_localize('UTC')\n"
    "    return {'commits': _commits, 'changes': _changes,\n"
    "            'sonar_issues': _sonar_issues, 'szz': _szz, 'jira': _jira}\n"
    "\n"
    "_required = ['labels_consequence.parquet', 'labels_severity.parquet', 'labels_szz.parquet']\n"
    "_have_all = all((PROCESSED_DATA_DIR / p).exists() for p in _required)\n"
    "if _have_all and not RECOMPUTE:\n"
    "    print('[Stage 4] Cached label parquets present; skipping.')\n"
    "    RUNTIMES['stage_04_label'] = 0.0\n"
    "else:\n"
    "    _t0 = time.time()\n"
    "    print(f'[Stage 4] window={OBSERVATION_WINDOW_MONTHS} mo, percentile=top {HIGH_RISK_PERCENTILE}%, '\n"
    "          f'severity={SEVERITY_BASELINE_LEVELS}')\n"
    "    _data = _load_clean_for_labeling()\n"
    "    _snaps = load_snapshots(PROCESSED_DATA_DIR / 'project_snapshots.parquet')\n"
    "    _eligible = _snaps[_snaps['eligible']].sort_values('project_id')\n"
    "    _cons_parts, _sev_parts, _szz_parts, _summary = [], [], [], []\n"
    "    for _, _row in tqdm(list(_eligible.iterrows()), desc='Labelling projects'):\n"
    "        _pid, _t = _row['project_id'], _row['snapshot_date']\n"
    "        _cons = compute_consequence_labels(_pid, _t, _data['commits'], _data['changes'],\n"
    "                                           _data['szz'], _data['jira'])\n"
    "        _sev = compute_severity_labels(_pid, _t, _data['sonar_issues'], _data['changes'])\n"
    "        _szzl = compute_szz_labels(_pid, _t, _data['changes'], _data['szz'])\n"
    "        _cons_parts.append(_cons); _sev_parts.append(_sev); _szz_parts.append(_szzl)\n"
    "        _summary.append({\n"
    "            'project_id': _pid, 'n_basenames': len(_cons),\n"
    "            'consequence_positives': int(_cons['is_high_risk'].sum()),\n"
    "            'consequence_rate_pct': round(100 * _cons['is_high_risk'].mean(), 2),\n"
    "            'severity_positives': int(_sev['is_high_risk'].sum()),\n"
    "            'severity_rate_pct': round(100 * _sev['is_high_risk'].mean(), 2),\n"
    "            'szz_positives': int(_szzl['is_high_risk'].sum()),\n"
    "            'szz_rate_pct': round(100 * _szzl['is_high_risk'].mean(), 2),\n"
    "        })\n"
    "    _cons_all = pd.concat(_cons_parts, ignore_index=True)\n"
    "    _sev_all = pd.concat(_sev_parts, ignore_index=True)\n"
    "    _szz_all = pd.concat(_szz_parts, ignore_index=True)\n"
    "    _cons_all.to_parquet(PROCESSED_DATA_DIR / 'labels_consequence.parquet', index=False)\n"
    "    _sev_all.to_parquet(PROCESSED_DATA_DIR / 'labels_severity.parquet', index=False)\n"
    "    _szz_all.to_parquet(PROCESSED_DATA_DIR / 'labels_szz.parquet', index=False)\n"
    "    pd.DataFrame(_summary).to_csv(TABLES_DIR / 'label_summary.csv', index=False)\n"
    "    _agr = label_agreement({'consequence': _cons_all, 'severity': _sev_all, 'szz': _szz_all})\n"
    "    _agr.to_csv(TABLES_DIR / 'label_agreement.csv', index=False)\n"
    "    RUNTIMES['stage_04_label'] = round(time.time() - _t0, 2)\n"
    "    print(f'[Stage 4] Elapsed: {RUNTIMES[\"stage_04_label\"]} s')\n"
))

cells.append(code(
    "# Inspect labelling outcomes.\n"
    "_summary = pd.read_csv(TABLES_DIR / 'label_summary.csv')\n"
    "_agr = pd.read_csv(TABLES_DIR / 'label_agreement.csv')\n"
    "display(Markdown('**Per-project positive rates (%):**'))\n"
    "display(_summary.style.format({\n"
    "    'consequence_rate_pct': '{:.2f}',\n"
    "    'severity_rate_pct': '{:.2f}',\n"
    "    'szz_rate_pct': '{:.2f}',\n"
    "}))\n"
    "display(Markdown('**Pairwise label agreement (Cohen kappa, Jaccard):**'))\n"
    "display(_agr)\n"
    "\n"
    "for _v in ('consequence', 'severity', 'szz'):\n"
    "    _df = pd.read_parquet(PROCESSED_DATA_DIR / f'labels_{_v}.parquet')\n"
    "    print(f'  labels_{_v:<11} rows={len(_df):>6,} positive_rate={100*_df[\"is_high_risk\"].mean():.2f}%')\n"
))


# ===== 9. Stage 5 ===========================================================
cells.append(md(
    '<a id="stage-5"></a>\n'
    "## Stage 5 - Static and historical features\n"
    "\n"
    "Build snapshot-aware features at `(project_id, basename)` granularity. Two families are produced:\n"
    "\n"
    "- **Static** - per-basename SonarQube issue aggregates open at `t`, plus project-level SonarQube context from the latest analysis with `analysis_date <= t`, plus a Git-derived size proxy.\n"
    "- **Historical (process)** - per-basename Git commit statistics restricted to commits with `AUTHOR_DATE <= t`: total commits, contributors, churn, recency windows (30/90 d), age, ownership ratio, distributional churn statistics.\n"
    "\n"
    "Outputs: `data/processed/features_{static,historical}.parquet`, `results/tables/feature_summary.csv`.\n"
))

cells.append(code(
    "from src.features.static_features import build_static_features_for_project\n"
    "from src.features.historical_features import build_historical_features_for_project\n"
    "from src.features.graph_features import build_graph_features_for_project\n"
    "from src.features.priordefect_features import build_priordefect_features_for_project\n"
    "\n"
    "def _load_clean_for_features():\n"
    "    _commits = pd.read_parquet(PROCESSED_DATA_DIR / 'clean_git_commits.parquet')\n"
    "    _changes = pd.read_parquet(PROCESSED_DATA_DIR / 'clean_git_commits_changes.parquet')\n"
    "    _issues = pd.read_parquet(PROCESSED_DATA_DIR / 'clean_sonar_issues.parquet')\n"
    "    _measures = pd.read_parquet(PROCESSED_DATA_DIR / 'clean_sonar_measures.parquet')\n"
    "    _szz_df = pd.read_parquet(PROCESSED_DATA_DIR / 'clean_szz.parquet')\n"
    "    _jira = pd.read_parquet(PROCESSED_DATA_DIR / 'clean_jira_issues.parquet')\n"
    "    for _df, _cols in (\n"
    "        (_commits, ['AUTHOR_DATE', 'COMMITTER_DATE']),\n"
    "        (_changes, ['DATE']),\n"
    "        (_issues, ['CREATION_DATE', 'CLOSE_DATE']),\n"
    "        (_measures, ['analysis_date']),\n"
    "        (_szz_df, ['fix_date', 'induce_date']),\n"
    "        (_jira, ['CREATION_DATE', 'RESOLUTION_DATE', 'UPDATE_DATE', 'COMMIT_DATE']),\n"
    "    ):\n"
    "        for _c in _cols:\n"
    "            if _c in _df.columns and _df[_c].dtype.kind == 'M' and _df[_c].dt.tz is None:\n"
    "                _df[_c] = _df[_c].dt.tz_localize('UTC')\n"
    "    return {'commits': _commits, 'changes': _changes,\n"
    "            'sonar_issues': _issues, 'sonar_measures': _measures,\n"
    "            'szz': _szz_df, 'jira': _jira}\n"
    "\n"
    "_required = ['features_static.parquet', 'features_historical.parquet',\n"
    "             'features_graph.parquet', 'features_priordefect.parquet']\n"
    "_have_all = all((PROCESSED_DATA_DIR / p).exists() for p in _required)\n"
    "if _have_all and not RECOMPUTE:\n"
    "    print('[Stage 5] Cached feature parquets present; skipping.')\n"
    "    RUNTIMES['stage_05_features'] = 0.0\n"
    "else:\n"
    "    _t0 = time.time()\n"
    "    _data = _load_clean_for_features()\n"
    "    _snaps = load_snapshots(PROCESSED_DATA_DIR / 'project_snapshots.parquet')\n"
    "    _eligible = _snaps[_snaps['eligible']].sort_values('project_id')\n"
    "    _static_parts, _hist_parts, _graph_parts, _prior_parts, _summary = [], [], [], [], []\n"
    "    for _, _row in tqdm(list(_eligible.iterrows()), desc='Building features'):\n"
    "        _pid, _t = _row['project_id'], _row['snapshot_date']\n"
    "        _ts = time.time()\n"
    "        _static_df = build_static_features_for_project(\n"
    "            _pid, _t, _data['sonar_issues'], _data['sonar_measures'], _data['changes'])\n"
    "        _hist_df = build_historical_features_for_project(\n"
    "            _pid, _t, _data['commits'], _data['changes'])\n"
    "        _graph_df = build_graph_features_for_project(_pid, _t, _data['changes'])\n"
    "        _prior_df = build_priordefect_features_for_project(\n"
    "            _pid, _t, _data['commits'], _data['changes'], _data['szz'], _data['jira'])\n"
    "        _static_parts.append(_static_df); _hist_parts.append(_hist_df)\n"
    "        _graph_parts.append(_graph_df); _prior_parts.append(_prior_df)\n"
    "        _summary.append({\n"
    "            'project_id': _pid, 'n_basenames': len(_static_df),\n"
    "            'static_cols': len(_static_df.columns),\n"
    "            'hist_cols': len(_hist_df.columns),\n"
    "            'graph_cols': len(_graph_df.columns),\n"
    "            'prior_cols': len(_prior_df.columns),\n"
    "            'mean_cocg_degree': round(float(_graph_df['cocg_degree'].mean()), 2)\n"
    "                if 'cocg_degree' in _graph_df.columns else 0.0,\n"
    "            'mean_bugfix_pre': round(float(_prior_df['bugfix_commits_pre'].mean()), 2)\n"
    "                if 'bugfix_commits_pre' in _prior_df.columns else 0.0,\n"
    "            'elapsed_s': round(time.time() - _ts, 2),\n"
    "        })\n"
    "    _static_all = pd.concat(_static_parts, ignore_index=True)\n"
    "    _hist_all = pd.concat(_hist_parts, ignore_index=True)\n"
    "    _graph_all = pd.concat(_graph_parts, ignore_index=True)\n"
    "    _prior_all = pd.concat(_prior_parts, ignore_index=True)\n"
    "    _static_all.to_parquet(PROCESSED_DATA_DIR / 'features_static.parquet', index=False)\n"
    "    _hist_all.to_parquet(PROCESSED_DATA_DIR / 'features_historical.parquet', index=False)\n"
    "    _graph_all.to_parquet(PROCESSED_DATA_DIR / 'features_graph.parquet', index=False)\n"
    "    _prior_all.to_parquet(PROCESSED_DATA_DIR / 'features_priordefect.parquet', index=False)\n"
    "    pd.DataFrame(_summary).to_csv(TABLES_DIR / 'feature_summary.csv', index=False)\n"
    "    RUNTIMES['stage_05_features'] = round(time.time() - _t0, 2)\n"
    "    print(f'[Stage 5] Elapsed: {RUNTIMES[\"stage_05_features\"]} s '\n"
    "          f'(static={len(_static_all):,}, hist={len(_hist_all):,}, '\n"
    "          f'graph={len(_graph_all):,}, prior={len(_prior_all):,})')\n"
))

cells.append(code(
    "# Inspect feature shapes and a sample row.\n"
    "_static = pd.read_parquet(PROCESSED_DATA_DIR / 'features_static.parquet')\n"
    "_hist = pd.read_parquet(PROCESSED_DATA_DIR / 'features_historical.parquet')\n"
    "display(Markdown(\n"
    "    f'**features_static**: {_static.shape[0]:,} rows x {_static.shape[1]} cols  -  '\n"
    "    f'**features_historical**: {_hist.shape[0]:,} rows x {_hist.shape[1]} cols.'\n"
    "))\n"
    "display(Markdown('**Feature summary per project:**'))\n"
    "display(pd.read_csv(TABLES_DIR / 'feature_summary.csv'))\n"
    "display(Markdown('**Sample static feature row:**'))\n"
    "display(_static.head(3))\n"
    "display(Markdown('**Sample historical feature row:**'))\n"
    "display(_hist.head(3))\n"
))


# ===== 10. Stage 6 ==========================================================
cells.append(md(
    '<a id="stage-6"></a>\n'
    "## Stage 6 - Dataset assembly with leakage audit\n"
    "\n"
    "Merge static and historical features with each label variant; for each variant, drop columns that would leak the label (e.g. severity counts for the severity variant). Missing project-level context is median-imputed and an indicator `has_project_context` is added.\n"
    "\n"
    "An audit row is written per variant verifying:\n"
    "- No leaky feature is retained.\n"
    "- The post-merge positive rate matches Stage 4.\n"
    "- The number of training features.\n"
    "\n"
    "Outputs: `data/processed/dataset_{consequence,severity,szz}.parquet`, `results/tables/dataset_summary.csv`.\n"
))

cells.append(code(
    "from config import SEVERITY_LEAKY_FEATURES, SZZ_LEAKY_FEATURES\n"
    "import numpy as np\n"
    "\n"
    "_KEY_COLS = ['project_id', 'basename']\n"
    "_LABEL_COLS = ['is_high_risk']\n"
    "\n"
    "def _load_features():\n"
    "    _static = pd.read_parquet(PROCESSED_DATA_DIR / 'features_static.parquet')\n"
    "    _hist = pd.read_parquet(PROCESSED_DATA_DIR / 'features_historical.parquet')\n"
    "    if 'snapshot_date' in _static.columns and 'snapshot_date' in _hist.columns:\n"
    "        _hist = _hist.drop(columns=['snapshot_date'])\n"
    "    return _static.merge(_hist, on=_KEY_COLS, how='outer')\n"
    "\n"
    "def _prepare_labels(kind):\n"
    "    _df = pd.read_parquet(PROCESSED_DATA_DIR / f'labels_{kind}.parquet')\n"
    "    _keep = _KEY_COLS + ['is_high_risk']\n"
    "    if kind == 'consequence' and 'risk_score' in _df.columns:\n"
    "        _keep.append('risk_score')\n"
    "    return _df[_keep]\n"
    "\n"
    "def _drop_leaky(df, variant):\n"
    "    if variant == 'severity':\n"
    "        return df.drop(columns=[c for c in SEVERITY_LEAKY_FEATURES if c in df.columns])\n"
    "    if variant == 'szz':\n"
    "        return df.drop(columns=[c for c in SZZ_LEAKY_FEATURES if c in df.columns])\n"
    "    return df\n"
    "\n"
    "def _fill_missing_context(df):\n"
    "    _proj_cols = [c for c in df.columns if c.startswith('project_') and c != 'project_id']\n"
    "    if not _proj_cols:\n"
    "        df['has_project_context'] = 1\n"
    "        return df\n"
    "    _ctx = df[_proj_cols].notna().any(axis=1)\n"
    "    df = df.assign(has_project_context=_ctx.astype('int64'))\n"
    "    _num = df[_proj_cols].select_dtypes(include='number').columns.tolist()\n"
    "    if _num:\n"
    "        df[_num] = df[_num].fillna(df[_num].median())\n"
    "    _non_num = [c for c in _proj_cols if c not in _num]\n"
    "    if _non_num:\n"
    "        df = df.drop(columns=_non_num)\n"
    "    return df\n"
    "\n"
    "_t0 = time.time()\n"
    "_features = _load_features()\n"
    "_audits = []\n"
    "for _variant in ('consequence', 'severity', 'szz'):\n"
    "    _labels = _prepare_labels(_variant)\n"
    "    _df = _features.merge(_labels, on=_KEY_COLS, how='inner')\n"
    "    _df = _drop_leaky(_df, _variant)\n"
    "    _df = _fill_missing_context(_df)\n"
    "    _forbidden = {'severity': set(SEVERITY_LEAKY_FEATURES),\n"
    "                  'szz': set(SZZ_LEAKY_FEATURES),\n"
    "                  'consequence': set()}[_variant]\n"
    "    _retained = sorted(set(_df.columns) & _forbidden)\n"
    "    _n_features = len(_df.columns) - len(_KEY_COLS) - len(_LABEL_COLS)\n"
    "    if _variant == 'consequence' and 'risk_score' in _df.columns:\n"
    "        _n_features -= 1\n"
    "    _audits.append({\n"
    "        'variant': _variant,\n"
    "        'rows': len(_df),\n"
    "        'columns_total': len(_df.columns),\n"
    "        'n_features': _n_features,\n"
    "        'positive_rate_pct': round(100 * float(_df['is_high_risk'].mean()), 2),\n"
    "        'positives': int(_df['is_high_risk'].sum()),\n"
    "        'retained_leaky_cols': _retained,\n"
    "        'cols_with_any_na': int((_df.isna().any()).sum()),\n"
    "    })\n"
    "    _df.to_parquet(PROCESSED_DATA_DIR / f'dataset_{_variant}.parquet', index=False)\n"
    "\n"
    "_summary = pd.DataFrame(_audits)\n"
    "_summary['retained_leaky_cols'] = _summary['retained_leaky_cols'].apply(\n"
    "    lambda xs: ';'.join(xs) if xs else '')\n"
    "_summary.to_csv(TABLES_DIR / 'dataset_summary.csv', index=False)\n"
    "RUNTIMES['stage_06_dataset'] = round(time.time() - _t0, 2)\n"
    "print(f'[Stage 6] Elapsed: {RUNTIMES[\"stage_06_dataset\"]} s')\n"
))

cells.append(code(
    "# Inspect leakage audit and dataset summaries.\n"
    "_summary = pd.read_csv(TABLES_DIR / 'dataset_summary.csv')\n"
    "display(Markdown('**Per-variant leakage audit:**'))\n"
    "display(_summary)\n"
    "\n"
    "_assert = _summary['retained_leaky_cols'].fillna('').eq('').all()\n"
    "display(Markdown(\n"
    "    f'**Leakage audit verdict:** '\n"
    "    f\"{'PASSED - no leaky features retained.' if _assert else 'FAILED - inspect summary above.'}\"\n"
    "))\n"
))


# ===== 11. Stage 7 ==========================================================
cells.append(md(
    '<a id="stage-7"></a>\n'
    "## Stage 7 - Within-project 10-fold cross-validation\n"
    "\n"
    "Stratified 10-fold CV on the combined dataset for each `(variant, model)` pair. Five model families are evaluated: logistic regression, decision tree, random forest, XGBoost and LightGBM. The metric battery is precision, recall, F1, ROC-AUC, PR-AUC, MCC, and CE@20 (Cost-Effectiveness at top-20%).\n"
    "\n"
    "Outputs: `results/tables/within_project_folds.csv`, `results/tables/within_project_summary.csv`.\n"
))

cells.append(code(
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "from src.models.train import (\n"
    "    fold_results_to_frame, load_variant_matrix, stratified_kfold_cv, summarize,\n"
    ")\n"
    "from src.analysis.significance import (\n"
    "    attach_ci_to_summary, bootstrap_confidence_intervals, pairwise_wilcoxon,\n"
    ")\n"
    "from config import CV_FOLDS, LABEL_VARIANTS\n"
    "\n"
    "_MODELS_W = ['logistic_regression', 'decision_tree', 'random_forest',\n"
    "             'xgboost', 'lightgbm', 'svm']\n"
    "_pred_path = TABLES_DIR / 'within_project_predictions.parquet'\n"
    "if _pred_path.exists():\n"
    "    _pred_path.unlink()\n"
    "_t0 = time.time()\n"
    "_all_folds = []\n"
    "for _variant in LABEL_VARIANTS:\n"
    "    _X, _y, _proj = load_variant_matrix(_variant)\n"
    "    print(f'[Stage 7] {_variant:<11} rows={len(_X):,} feats={_X.shape[1]} '\n"
    "          f'positives={int(_y.sum())} ({100*_y.mean():.2f}%)')\n"
    "    for _m in _MODELS_W:\n"
    "        _t1 = time.time()\n"
    "        _r = stratified_kfold_cv(\n"
    "            _variant, _m, _X, _y, n_splits=CV_FOLDS,\n"
    "            persist_predictions=True, project_id=_proj,\n"
    "        )\n"
    "        if not _r:\n"
    "            print(f'   {_m:<20} SKIPPED (not installed)')\n"
    "            continue\n"
    "        _df = fold_results_to_frame(_r)\n"
    "        _means = _df[['precision','recall','f1','roc_auc','pr_auc','mcc','ce_at_20']].mean()\n"
    "        print(f'   {_m:<20} F1={_means[\"f1\"]:.3f} ROC={_means[\"roc_auc\"]:.3f} '\n"
    "              f'PR={_means[\"pr_auc\"]:.3f} MCC={_means[\"mcc\"]:.3f} CE@20={_means[\"ce_at_20\"]:.3f} '\n"
    "              f'({time.time()-_t1:.1f}s)')\n"
    "        _all_folds.append(_df)\n"
    "\n"
    "_fold_df = pd.concat(_all_folds, ignore_index=True)\n"
    "_fold_df.to_csv(TABLES_DIR / 'within_project_folds.csv', index=False)\n"
    "_summary = summarize(_fold_df)\n"
    "_summary.to_csv(TABLES_DIR / 'within_project_summary.csv', index=False)\n"
    "\n"
    "try:\n"
    "    _ci = bootstrap_confidence_intervals(_fold_df)\n"
    "    _ci.to_csv(TABLES_DIR / 'within_project_ci.csv', index=False)\n"
    "    attach_ci_to_summary(_summary, _ci).to_csv(\n"
    "        TABLES_DIR / 'within_project_summary_with_ci.csv', index=False)\n"
    "    _sig_rows = []\n"
    "    for _metric in ('f1', 'pr_auc'):\n"
    "        _sig_rows.append(pairwise_wilcoxon(_fold_df,\n"
    "            pair_within='variant', contrast_col='model',\n"
    "            metric=_metric, paired_on=('fold',)))\n"
    "    pd.concat(_sig_rows, ignore_index=True).to_csv(\n"
    "        TABLES_DIR / 'pairwise_significance.csv', index=False)\n"
    "    print('[Stage 7] Significance: within_project_ci.csv + pairwise_significance.csv')\n"
    "except Exception as _exc:\n"
    "    print(f'[Stage 7] Significance step skipped: {_exc}')\n"
    "\n"
    "RUNTIMES['stage_07_within'] = round(time.time() - _t0, 2)\n"
    "print(f'[Stage 7] Elapsed: {RUNTIMES[\"stage_07_within\"]} s')\n"
))

cells.append(code(
    "# Inspect within-project summary (mean across folds, 3 d.p.).\n"
    "_w = pd.read_csv(TABLES_DIR / 'within_project_summary.csv')\n"
    "_mean_cols = [c for c in _w.columns if c.endswith('_mean')]\n"
    "_pretty = _w[['variant', 'model'] + _mean_cols].copy()\n"
    "_pretty[_mean_cols] = _pretty[_mean_cols].round(3)\n"
    "display(Markdown('**Within-project 10-fold CV (mean of folds):**'))\n"
    "display(_pretty)\n"
    "\n"
    "_ci_path = TABLES_DIR / 'within_project_ci.csv'\n"
    "if _ci_path.exists():\n"
    "    _ci = pd.read_csv(_ci_path)\n"
    "    _ci_cols = ['variant','model','metric','mean','ci_low','ci_high']\n"
    "    _ci_cols = [c for c in _ci_cols if c in _ci.columns]\n"
    "    display(Markdown('**Bootstrap 95% confidence intervals (within-project):**'))\n"
    "    display(_ci[_ci_cols].head(30))\n"
    "_sig_path = TABLES_DIR / 'pairwise_significance.csv'\n"
    "if _sig_path.exists():\n"
    "    _sig = pd.read_csv(_sig_path)\n"
    "    display(Markdown('**Pairwise Wilcoxon (Bonferroni-corrected, p < 0.05 highlighted):**'))\n"
    "    display(_sig.head(40))\n"
))


# ===== Stage 7b - Hyperparameter tuning ====================================
cells.append(md(
    "## Stage 7b - Hyperparameter tuning (Optuna)\n"
    "\n"
    "Run Optuna-driven hyperparameter search per (variant, model) using PR-AUC on inner stratified folds. The best parameters are then re-evaluated on the full 10-fold within-project CV and persisted alongside the baseline summary.\n"
    "\n"
    "Outputs: `results/tables/tuned_params.json`, `results/tables/within_project_summary_tuned.csv`.\n"
))

cells.append(code(
    "from src.models.train import fold_results_to_frame, load_variant_matrix, stratified_kfold_cv, summarize\n"
    "from src.models.tuning import tune_model\n"
    "from config import LABEL_VARIANTS, TUNING_TRIALS, CV_FOLDS\n"
    "import json as _json\n"
    "\n"
    "_MODELS_TUNE = ['logistic_regression', 'decision_tree', 'random_forest',\n"
    "                'xgboost', 'lightgbm']\n"
    "_t0 = time.time()\n"
    "_tuned_records = []\n"
    "_tuned_folds = []\n"
    "for _variant in LABEL_VARIANTS:\n"
    "    _X, _y, _ = load_variant_matrix(_variant)\n"
    "    print(f'[Stage 7b] {_variant} (n_trials={TUNING_TRIALS})')\n"
    "    for _m in _MODELS_TUNE:\n"
    "        _t1 = time.time()\n"
    "        try:\n"
    "            _rec = tune_model(_variant, _m, _X, _y, n_trials=TUNING_TRIALS)\n"
    "        except Exception as _exc:\n"
    "            print(f'   {_m:<20} tuning failed: {_exc}')\n"
    "            continue\n"
    "        _tuned_records.append(_rec)\n"
    "        print(f'   {_m:<20} best_PR_AUC={_rec[\"best_pr_auc\"]:.3f} '\n"
    "              f'trials={_rec[\"n_trials_completed\"]}/{_rec[\"n_trials_requested\"]} '\n"
    "              f'({_rec[\"elapsed_s\"]}s)')\n"
    "        _r = stratified_kfold_cv(_variant, _m, _X, _y, n_splits=CV_FOLDS,\n"
    "                                 params=_rec['best_params'])\n"
    "        if _r:\n"
    "            _df = fold_results_to_frame(_r)\n"
    "            _means = _df[['f1','pr_auc','mcc','ce_at_20']].mean()\n"
    "            print(f'   {_m:<20} -> outer F1={_means[\"f1\"]:.3f} PR={_means[\"pr_auc\"]:.3f} '\n"
    "                  f'CE@20={_means[\"ce_at_20\"]:.3f} ({time.time()-_t1:.1f}s)')\n"
    "            _tuned_folds.append(_df)\n"
    "\n"
    "(TABLES_DIR / 'tuned_params.json').write_text(\n"
    "    _json.dumps(_tuned_records, indent=2, default=str), encoding='utf-8')\n"
    "if _tuned_folds:\n"
    "    _tdf = pd.concat(_tuned_folds, ignore_index=True)\n"
    "    _tdf.to_csv(TABLES_DIR / 'within_project_folds_tuned.csv', index=False)\n"
    "    summarize(_tdf).to_csv(TABLES_DIR / 'within_project_summary_tuned.csv', index=False)\n"
    "RUNTIMES['stage_07b_tune'] = round(time.time() - _t0, 2)\n"
    "print(f'[Stage 7b] Elapsed: {RUNTIMES[\"stage_07b_tune\"]} s')\n"
))

cells.append(code(
    "_tp = TABLES_DIR / 'within_project_summary_tuned.csv'\n"
    "if _tp.exists():\n"
    "    _t = pd.read_csv(_tp)\n"
    "    _mean_cols = [c for c in _t.columns if c.endswith('_mean')]\n"
    "    _disp = _t[['variant','model'] + _mean_cols].copy()\n"
    "    _disp[_mean_cols] = _disp[_mean_cols].round(3)\n"
    "    display(Markdown('**Tuned within-project 10-fold CV:**'))\n"
    "    display(_disp)\n"
    "else:\n"
    "    print('No tuned_summary file produced.')\n"
))


# ===== Stage 7c - Probability calibration ===================================
cells.append(md(
    "## Stage 7c - Probability calibration sweep\n"
    "\n"
    "Compare uncalibrated probabilities against Platt scaling and isotonic regression for each (variant, model) using stratified 10-fold CV. Calibration quality is reported via Brier score, NLL and Expected Calibration Error (ECE).\n"
    "\n"
    "Outputs: `results/tables/calibration_folds.csv`, `calibration_summary.csv`, `calibration_predictions.parquet`.\n"
))

cells.append(code(
    "from src.models.train import (\n"
    "    calibrated_kfold_cv, fold_results_to_frame, load_variant_matrix,\n"
    "    stratified_kfold_cv, summarize,\n"
    ")\n"
    "from config import LABEL_VARIANTS, CV_FOLDS, CALIBRATION_METHODS\n"
    "\n"
    "_MODELS_CAL = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']\n"
    "_t0 = time.time()\n"
    "_cal_folds = []\n"
    "_cal_long = []\n"
    "for _variant in LABEL_VARIANTS:\n"
    "    _X, _y, _ = load_variant_matrix(_variant)\n"
    "    print(f'[Stage 7c] variant={_variant} rows={len(_X):,} feats={_X.shape[1]}')\n"
    "    for _m in _MODELS_CAL:\n"
    "        _t1 = time.time()\n"
    "        _base = stratified_kfold_cv(_variant, _m, _X, _y, n_splits=CV_FOLDS)\n"
    "        if _base:\n"
    "            _bdf = fold_results_to_frame(_base)\n"
    "            _bdf['method'] = 'uncalibrated'\n"
    "            _bdf['brier'] = float('nan')\n"
    "            _bdf['nll'] = float('nan')\n"
    "            _bdf['ece'] = float('nan')\n"
    "            _cal_folds.append(_bdf)\n"
    "        for _method in CALIBRATION_METHODS:\n"
    "            _t2 = time.time()\n"
    "            _r, _long = calibrated_kfold_cv(_variant, _m, _X, _y,\n"
    "                                              n_splits=CV_FOLDS, method=_method)\n"
    "            if not _r:\n"
    "                continue\n"
    "            _df = fold_results_to_frame(_r)\n"
    "            _df['method'] = _method\n"
    "            _means = _df[['brier','nll','ece','f1']].mean()\n"
    "            print(f'   {_m:<18} {_method:<9} brier={_means[\"brier\"]:.4f} '\n"
    "                  f'ECE={_means[\"ece\"]:.4f} F1={_means[\"f1\"]:.3f} '\n"
    "                  f'({time.time()-_t2:.1f}s)')\n"
    "            _cal_folds.append(_df)\n"
    "            _cal_long.append(_long)\n"
    "\n"
    "if _cal_folds:\n"
    "    _cdf = pd.concat(_cal_folds, ignore_index=True)\n"
    "    _cdf.to_csv(TABLES_DIR / 'calibration_folds.csv', index=False)\n"
    "    summarize(_cdf, group_cols=('variant','model','method')).to_csv(\n"
    "        TABLES_DIR / 'calibration_summary.csv', index=False)\n"
    "if _cal_long:\n"
    "    pd.concat(_cal_long, ignore_index=True).to_parquet(\n"
    "        TABLES_DIR / 'calibration_predictions.parquet', index=False)\n"
    "RUNTIMES['stage_07c_calibrate'] = round(time.time() - _t0, 2)\n"
    "print(f'[Stage 7c] Elapsed: {RUNTIMES[\"stage_07c_calibrate\"]} s')\n"
))

cells.append(code(
    "_cp = TABLES_DIR / 'calibration_summary.csv'\n"
    "if _cp.exists():\n"
    "    _c = pd.read_csv(_cp)\n"
    "    _show = ['variant','model','method','brier_mean','nll_mean','ece_mean','f1_mean']\n"
    "    _show = [c for c in _show if c in _c.columns]\n"
    "    _disp = _c[_show].copy()\n"
    "    _num = [c for c in _disp.columns if _disp[c].dtype.kind == 'f']\n"
    "    _disp[_num] = _disp[_num].round(4)\n"
    "    display(Markdown('**Calibration summary (Brier / NLL / ECE):**'))\n"
    "    display(_disp)\n"
    "else:\n"
    "    print('No calibration summary produced.')\n"
))


# ===== Stage 7d - SMOTE vs class_weight =====================================
cells.append(md(
    "## Stage 7d - Resampling sweep (SMOTE vs class_weight)\n"
    "\n"
    "Compare SMOTE oversampling against the default `class_weight='balanced'` strategy on the within-project 10-fold CV. Deltas in F1, PR-AUC and MCC are emitted to highlight the better imbalance-handling choice per variant.\n"
    "\n"
    "Outputs: `results/tables/within_project_folds_smote.csv`, `within_project_summary_smote.csv`, `resampling_comparison.csv`.\n"
))

cells.append(code(
    "from src.models.train import fold_results_to_frame, load_variant_matrix, stratified_kfold_cv, summarize\n"
    "from config import LABEL_VARIANTS, CV_FOLDS\n"
    "\n"
    "_MODELS_SM = ['logistic_regression', 'decision_tree', 'random_forest',\n"
    "              'xgboost', 'lightgbm', 'svm']\n"
    "_KEY_METRICS = ('f1','pr_auc','mcc','ce_at_20')\n"
    "_t0 = time.time()\n"
    "_sm_folds = []\n"
    "for _variant in LABEL_VARIANTS:\n"
    "    _X, _y, _ = load_variant_matrix(_variant)\n"
    "    for _m in _MODELS_SM:\n"
    "        _t1 = time.time()\n"
    "        _r = stratified_kfold_cv(_variant, _m, _X, _y, n_splits=CV_FOLDS, use_smote=True)\n"
    "        if not _r:\n"
    "            continue\n"
    "        _df = fold_results_to_frame(_r)\n"
    "        _df['resampler'] = 'smote'\n"
    "        _means = _df[list(_KEY_METRICS)].mean()\n"
    "        print(f'[Stage 7d] {_variant} {_m:<18} smote F1={_means[\"f1\"]:.3f} '\n"
    "              f'PR={_means[\"pr_auc\"]:.3f} CE@20={_means[\"ce_at_20\"]:.3f} '\n"
    "              f'({time.time()-_t1:.1f}s)')\n"
    "        _sm_folds.append(_df)\n"
    "\n"
    "if _sm_folds:\n"
    "    _sdf = pd.concat(_sm_folds, ignore_index=True)\n"
    "    _sdf.to_csv(TABLES_DIR / 'within_project_folds_smote.csv', index=False)\n"
    "    _smote_summary = summarize(_sdf)\n"
    "    _smote_summary.to_csv(TABLES_DIR / 'within_project_summary_smote.csv', index=False)\n"
    "    _cw_path = TABLES_DIR / 'within_project_summary.csv'\n"
    "    if _cw_path.exists():\n"
    "        _cw_summary = pd.read_csv(_cw_path)\n"
    "        _cw = _cw_summary.rename(columns={f'{m}_mean': f'{m}_cw' for m in _KEY_METRICS})\n"
    "        _cw = _cw[['variant','model'] + [f'{m}_cw' for m in _KEY_METRICS]]\n"
    "        _sm = _smote_summary.rename(columns={f'{m}_mean': f'{m}_smote' for m in _KEY_METRICS})\n"
    "        _sm = _sm[['variant','model'] + [f'{m}_smote' for m in _KEY_METRICS]]\n"
    "        _comp = _cw.merge(_sm, on=['variant','model'], how='outer')\n"
    "        for _m in _KEY_METRICS:\n"
    "            _comp[f'{_m}_delta'] = (_comp[f'{_m}_smote'] - _comp[f'{_m}_cw']).round(3)\n"
    "            _comp[f'{_m}_cw'] = _comp[f'{_m}_cw'].round(3)\n"
    "            _comp[f'{_m}_smote'] = _comp[f'{_m}_smote'].round(3)\n"
    "        _comp.to_csv(TABLES_DIR / 'resampling_comparison.csv', index=False)\n"
    "RUNTIMES['stage_07d_resample'] = round(time.time() - _t0, 2)\n"
    "print(f'[Stage 7d] Elapsed: {RUNTIMES[\"stage_07d_resample\"]} s')\n"
))

cells.append(code(
    "_rp = TABLES_DIR / 'resampling_comparison.csv'\n"
    "if _rp.exists():\n"
    "    _r = pd.read_csv(_rp)\n"
    "    display(Markdown('**Resampling comparison (SMOTE - class_weight):**'))\n"
    "    display(_r)\n"
    "else:\n"
    "    print('No resampling comparison produced.')\n"
))


# ===== Stage 7e - Temporal within-project T1 -> T2 ==========================
cells.append(md(
    "## Stage 7e - Temporal within-project split (T1 -> T2)\n"
    "\n"
    "For each project, build features and labels at an early percentile snapshot T1 and re-evaluate at a later T2. Training on T1 and testing on T2 quantifies how well the model generalizes to *future* code in the same project - a stricter check than random-fold CV.\n"
    "\n"
    "Outputs: `results/tables/temporal_per_project.csv`, `temporal_summary.csv`.\n"
))

cells.append(code(
    "from src.models.temporal import load_clean_for_temporal, temporal_split_results\n"
    "from config import (OBSERVATION_WINDOW_MONTHS,\n"
    "                    TEMPORAL_T1_PERCENTILE, TEMPORAL_T2_PERCENTILE)\n"
    "\n"
    "_MODELS_TEMP = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']\n"
    "_METRIC_COLS_T = ['precision','recall','f1','roc_auc','pr_auc','mcc','ce_at_20']\n"
    "_t0 = time.time()\n"
    "print(f'[Stage 7e] T1={TEMPORAL_T1_PERCENTILE} T2={TEMPORAL_T2_PERCENTILE} '\n"
    "      f'window={OBSERVATION_WINDOW_MONTHS}')\n"
    "_data = load_clean_for_temporal()\n"
    "_projects_temp = sorted(_data['commits']['PROJECT_ID'].dropna().unique().tolist())\n"
    "print(f'[Stage 7e] Projects in data: {len(_projects_temp)}')\n"
    "_temp_rows = []\n"
    "for _m in _MODELS_TEMP:\n"
    "    _t1 = time.time()\n"
    "    try:\n"
    "        _df = temporal_split_results(\n"
    "            _m, _data, _projects_temp,\n"
    "            p1=TEMPORAL_T1_PERCENTILE, p2=TEMPORAL_T2_PERCENTILE,\n"
    "            window_months=OBSERVATION_WINDOW_MONTHS,\n"
    "        )\n"
    "    except Exception as _exc:\n"
    "        print(f'   {_m:<18} failed: {_exc}')\n"
    "        continue\n"
    "    if _df is None or _df.empty:\n"
    "        print(f'   {_m:<18} no eligible projects')\n"
    "        continue\n"
    "    _means = _df[_METRIC_COLS_T].mean()\n"
    "    print(f'   {_m:<18} projs={len(_df):>2} F1={_means[\"f1\"]:.3f} '\n"
    "          f'PR={_means[\"pr_auc\"]:.3f} CE@20={_means[\"ce_at_20\"]:.3f} '\n"
    "          f'({time.time()-_t1:.1f}s)')\n"
    "    _temp_rows.append(_df)\n"
    "\n"
    "if _temp_rows:\n"
    "    _tdf = pd.concat(_temp_rows, ignore_index=True)\n"
    "    _tdf.to_csv(TABLES_DIR / 'temporal_per_project.csv', index=False)\n"
    "    _summ = (_tdf.groupby('model')[_METRIC_COLS_T]\n"
    "                  .agg(['mean','std','count'])\n"
    "                  .reset_index())\n"
    "    _summ.columns = ['_'.join(c).rstrip('_') if isinstance(c, tuple) else c\n"
    "                      for c in _summ.columns]\n"
    "    _summ.to_csv(TABLES_DIR / 'temporal_summary.csv', index=False)\n"
    "RUNTIMES['stage_07e_temporal'] = round(time.time() - _t0, 2)\n"
    "print(f'[Stage 7e] Elapsed: {RUNTIMES[\"stage_07e_temporal\"]} s')\n"
))

cells.append(code(
    "_tp = TABLES_DIR / 'temporal_summary.csv'\n"
    "if _tp.exists():\n"
    "    _t = pd.read_csv(_tp)\n"
    "    display(Markdown('**Temporal T1 -> T2 within-project summary:**'))\n"
    "    display(_t)\n"
    "else:\n"
    "    print('No temporal summary produced (skipped if any stage 7e step failed).')\n"
))


# ===== 12. Stage 8 ==========================================================
cells.append(md(
    '<a id="stage-8"></a>\n'
    "## Stage 8 - Leave-One-Project-Out cross-project validation\n"
    "\n"
    "For each `(variant, model)` pair, hold out one project at a time and train on the remaining 21. The realistic deployment metric: how well does a model trained on existing projects perform on a brand-new project?\n"
    "\n"
    "After computing LOPO means, a side-by-side `lopo_vs_within.csv` table is built (the **generalization gap**: within-project minus LOPO).\n"
    "\n"
    "Outputs: `results/tables/lopo_folds.csv`, `lopo_summary.csv`, `lopo_vs_within.csv`.\n"
))

cells.append(code(
    "from src.models.cross_project import (\n"
    "    lopo_cv, lopo_results_to_frame, lopo_summary,\n"
    ")\n"
    "from src.analysis.significance import (\n"
    "    attach_ci_to_summary, bootstrap_confidence_intervals, pairwise_wilcoxon,\n"
    ")\n"
    "_METRIC_COLS = ['precision','recall','f1','roc_auc','pr_auc','mcc','ce_at_20']\n"
    "# SVM is intentionally excluded from LOPO due to O(N^2) cost on cross-project folds.\n"
    "_MODELS_LOPO = ['logistic_regression', 'decision_tree', 'random_forest', 'xgboost', 'lightgbm']\n"
    "_lopo_pred_path = TABLES_DIR / 'lopo_predictions.parquet'\n"
    "if _lopo_pred_path.exists():\n"
    "    _lopo_pred_path.unlink()\n"
    "_t0 = time.time()\n"
    "_all_folds = []\n"
    "for _variant in LABEL_VARIANTS:\n"
    "    print(f'[Stage 8] variant={_variant}')\n"
    "    for _m in _MODELS_LOPO:\n"
    "        _t1 = time.time()\n"
    "        _r = lopo_cv(_variant, _m, persist_predictions=True)\n"
    "        if not _r:\n"
    "            print(f'   {_m:<20} SKIPPED'); continue\n"
    "        _df = lopo_results_to_frame(_r)\n"
    "        _means = _df[_METRIC_COLS].mean()\n"
    "        print(f'   {_m:<20} projs={len(_df):>2} F1={_means[\"f1\"]:.3f} '\n"
    "              f'ROC={_means[\"roc_auc\"]:.3f} PR={_means[\"pr_auc\"]:.3f} '\n"
    "              f'MCC={_means[\"mcc\"]:.3f} CE@20={_means[\"ce_at_20\"]:.3f} '\n"
    "              f'({time.time()-_t1:.1f}s)')\n"
    "        _all_folds.append(_df)\n"
    "\n"
    "_fold_df = pd.concat(_all_folds, ignore_index=True)\n"
    "_fold_df.to_csv(TABLES_DIR / 'lopo_folds.csv', index=False)\n"
    "_summary = lopo_summary(_fold_df)\n"
    "_summary.to_csv(TABLES_DIR / 'lopo_summary.csv', index=False)\n"
    "\n"
    "# Generalization gap\n"
    "_within_path = TABLES_DIR / 'within_project_summary.csv'\n"
    "if _within_path.exists():\n"
    "    _wp = pd.read_csv(_within_path)\n"
    "    _keep = ['variant', 'model'] + [f'{m}_mean' for m in _METRIC_COLS]\n"
    "    _wp = _wp[_keep].rename(columns={f'{m}_mean': f'{m}_within' for m in _METRIC_COLS})\n"
    "    _lp = _summary[['variant', 'model'] + [f'{m}_mean' for m in _METRIC_COLS]].rename(\n"
    "        columns={f'{m}_mean': f'{m}_lopo' for m in _METRIC_COLS})\n"
    "    _gap = _wp.merge(_lp, on=['variant', 'model'], how='outer')\n"
    "    for _m in _METRIC_COLS:\n"
    "        _gap[f'{_m}_gap'] = (_gap[f'{_m}_within'] - _gap[f'{_m}_lopo']).round(3)\n"
    "        _gap[f'{_m}_within'] = _gap[f'{_m}_within'].round(3)\n"
    "        _gap[f'{_m}_lopo'] = _gap[f'{_m}_lopo'].round(3)\n"
    "    _gap.to_csv(TABLES_DIR / 'lopo_vs_within.csv', index=False)\n"
    "\n"
    "try:\n"
    "    _ci = bootstrap_confidence_intervals(_fold_df)\n"
    "    _ci.to_csv(TABLES_DIR / 'lopo_ci.csv', index=False)\n"
    "    attach_ci_to_summary(_summary, _ci).to_csv(\n"
    "        TABLES_DIR / 'lopo_summary_with_ci.csv', index=False)\n"
    "    _sig_rows = []\n"
    "    for _metric in ('f1', 'pr_auc'):\n"
    "        _sig_rows.append(pairwise_wilcoxon(_fold_df,\n"
    "            pair_within='variant', contrast_col='model',\n"
    "            metric=_metric, paired_on=('held_out_project',)))\n"
    "    pd.concat(_sig_rows, ignore_index=True).to_csv(\n"
    "        TABLES_DIR / 'lopo_pairwise_significance.csv', index=False)\n"
    "    print('[Stage 8] Significance: lopo_ci.csv + lopo_pairwise_significance.csv')\n"
    "except Exception as _exc:\n"
    "    print(f'[Stage 8] Significance step skipped: {_exc}')\n"
    "\n"
    "RUNTIMES['stage_08_lopo'] = round(time.time() - _t0, 2)\n"
    "print(f'[Stage 8] Elapsed: {RUNTIMES[\"stage_08_lopo\"]} s')\n"
))

cells.append(code(
    "# Inspect LOPO summary and generalization gap.\n"
    "_lopo = pd.read_csv(TABLES_DIR / 'lopo_summary.csv')\n"
    "_mean_cols = [c for c in _lopo.columns if c.endswith('_mean')]\n"
    "_pretty = _lopo[['variant', 'model', 'n_projects'] + _mean_cols].copy()\n"
    "_pretty[_mean_cols] = _pretty[_mean_cols].round(3)\n"
    "display(Markdown('**LOPO summary (mean across held-out projects):**'))\n"
    "display(_pretty)\n"
    "\n"
    "_gap_path = TABLES_DIR / 'lopo_vs_within.csv'\n"
    "if _gap_path.exists():\n"
    "    _gap = pd.read_csv(_gap_path)\n"
    "    _gap_cols = ['variant','model','f1_within','f1_lopo','f1_gap',\n"
    "                 'ce_at_20_within','ce_at_20_lopo','ce_at_20_gap',\n"
    "                 'pr_auc_within','pr_auc_lopo','pr_auc_gap']\n"
    "    _gap_cols = [c for c in _gap_cols if c in _gap.columns]\n"
    "    display(Markdown('**Generalization gap (within - LOPO):**'))\n"
    "    display(_gap[_gap_cols])\n"
))


# ===== 12b. Collision analysis (MOD-Q1) ======================================
cells.append(md(
    '<a id="collision-analysis"></a>\n'
    "## Collision analysis - basename collision vs LOPO (thesis Section 5.5)\n"
    "\n"
    "Post-hoc grouping only: merges `path_overlap_report.csv` (Stage 3) with "
    "`lopo_folds.csv` (Stage 8) for LightGBM on the consequence variant. Writes "
    "`collision_analysis.csv`, `collision_analysis_summary.csv`, and prints "
    "thesis-ready paragraphs.\n"
    "\n"
    "**Skipped** if Stage 8 did not run (for example the demo notebook short-circuits LOPO) "
    "or if `path_overlap_report.csv` is missing.\n"
))

cells.append(writefile_cell("/content/tools/collision_analysis.py", read("tools/collision_analysis.py")))

cells.append(code(
    "import os\n"
    "\n"
    "_lp = TABLES_DIR / 'lopo_folds.csv'\n"
    "_po = TABLES_DIR / 'path_overlap_report.csv'\n"
    "_script = CONTENT / 'tools' / 'collision_analysis.py'\n"
    "if not _lp.exists():\n"
    "    print('[collision] skipped: missing lopo_folds.csv (run Stage 8; full notebook only).')\n"
    "elif not _po.exists():\n"
    "    print('[collision] skipped: missing path_overlap_report.csv (run Stage 3).')\n"
    "elif _script.exists():\n"
    "    import subprocess\n"
    "    import sys\n"
    "\n"
    "    _r = subprocess.run(\n"
    "        [sys.executable, str(_script)], cwd=str(CONTENT), env=os.environ.copy()\n"
    "    )\n"
    "    if _r.returncode != 0:\n"
    "        print(f'[collision] subprocess exited with code {_r.returncode}')\n"
    "else:\n"
    "    # Standalone notebook: ``tools/collision_analysis.py`` was inlined without ``__main__``.\n"
    "    main()\n"
))


# ===== 13. Stage 9 ==========================================================
cells.append(md(
    '<a id="stage-9"></a>\n'
    "## Stage 9 - Sensitivity grid and feature-group ablation\n"
    "\n"
    "Two robustness checks:\n"
    "\n"
    "1. **Sensitivity grid** (consequence variant only): re-derive labels for every (window, percentile) in `{3,6,12} x {10,20,30}`, retrain LightGBM with stratified 10-fold CV, and tabulate the metric battery for each cell. A robust pipeline shows stable or gracefully-degrading numbers across the grid.\n"
    "2. **Feature-group ablation** (all three variants): train with only one of `{static_sonar, historical, project_context}` at a time and with each group removed, against the all-features baseline. This isolates which family carries the predictive signal for each label variant.\n"
    "\n"
    "Outputs: `results/tables/sensitivity_consequence.csv`, `results/tables/feature_ablation.csv`.\n"
))

cells.append(code(
    "from src.analysis.sensitivity import run_sensitivity_grid\n"
    "from src.analysis.ablation import run_ablation\n"
    "\n"
    "_t0 = time.time()\n"
    "print('[Stage 9] Sensitivity grid (consequence, LightGBM)')\n"
    "_grid = run_sensitivity_grid(windows=(3, 6, 12), percentiles=(10.0, 20.0, 30.0))\n"
    "_grid.to_csv(TABLES_DIR / 'sensitivity_consequence.csv', index=False)\n"
    "\n"
    "_ablation_parts = []\n"
    "for _variant in LABEL_VARIANTS:\n"
    "    _t1 = time.time()\n"
    "    print(f'[Stage 9] Feature ablation: variant={_variant}')\n"
    "    _df = run_ablation(_variant, model_name='lightgbm')\n"
    "    _ablation_parts.append(_df)\n"
    "    print(f'   ({time.time() - _t1:.1f}s)')\n"
    "\n"
    "_ablation = pd.concat(_ablation_parts, ignore_index=True)\n"
    "_ablation.to_csv(TABLES_DIR / 'feature_ablation.csv', index=False)\n"
    "RUNTIMES['stage_09_sensitivity'] = round(time.time() - _t0, 2)\n"
    "print(f'[Stage 9] Elapsed: {RUNTIMES[\"stage_09_sensitivity\"]} s')\n"
))

cells.append(code(
    "# Inspect sensitivity grid (3x3) and ablation table (3 d.p.).\n"
    "_grid = pd.read_csv(TABLES_DIR / 'sensitivity_consequence.csv')\n"
    "_keep = ['window_months','percentile','positive_rate_pct',\n"
    "         'f1_mean','roc_auc_mean','pr_auc_mean','mcc_mean','ce_at_20_mean']\n"
    "_keep = [c for c in _keep if c in _grid.columns]\n"
    "_grid_disp = _grid[_keep].copy()\n"
    "_num_cols = [c for c in _grid_disp.columns if _grid_disp[c].dtype.kind == 'f']\n"
    "_grid_disp[_num_cols] = _grid_disp[_num_cols].round(3)\n"
    "display(Markdown('**Sensitivity grid (consequence variant):**'))\n"
    "display(_grid_disp)\n"
    "\n"
    "_abl = pd.read_csv(TABLES_DIR / 'feature_ablation.csv')\n"
    "_keep2 = ['variant','group','mode','n_features',\n"
    "          'f1_mean','roc_auc_mean','pr_auc_mean','ce_at_20_mean']\n"
    "_keep2 = [c for c in _keep2 if c in _abl.columns]\n"
    "_abl_disp = _abl[_keep2].copy()\n"
    "_num_cols2 = [c for c in _abl_disp.columns if _abl_disp[c].dtype.kind == 'f']\n"
    "_abl_disp[_num_cols2] = _abl_disp[_num_cols2].round(3)\n"
    "display(Markdown('**Feature-group ablation:**'))\n"
    "display(_abl_disp)\n"
))


# ===== 14. Stage 10 =========================================================
cells.append(md(
    '<a id="stage-10"></a>\n'
    "## Stage 10 - SHAP, figures and rendered reports\n"
    "\n"
    "Final reporting stage:\n"
    "\n"
    "1. **SHAP + permutation importance** per variant: train one LightGBM on an 80/20 stratified split, compute TreeSHAP and permutation importance, save top-15 tables and full tables.\n"
    "2. **Seven publication-ready figures** (300 DPI PNG + PDF): label-agreement Venn, per-project positive rates, within-vs-LOPO dot-plot, sensitivity heatmap, ablation bar chart, three SHAP summaries (one per variant), and LOPO per-project F1 box-plot.\n"
    "3. **Auto-generated reports**: `docs/06_results.md` (every number derived from the CSV/parquet artefacts) and `docs/07_discussion.md` (scaffold).\n"
))

cells.append(code(
    "from src.analysis.importance import compute_importance\n"
    "from src.reporting import figures as fig\n"
    "from src.reporting.render import render_results, render_discussion_scaffold\n"
    "\n"
    "_t0 = time.time()\n"
    "for _v in ('consequence', 'severity', 'szz'):\n"
    "    print(f'[Stage 10.1] SHAP / permutation importance: {_v}')\n"
    "    _out = compute_importance(_v, model_name='lightgbm')\n"
    "    _out['shap_summary'].head(15).to_csv(TABLES_DIR / f'shap_top15_{_v}.csv', index=False)\n"
    "    _out['permutation'].head(15).to_csv(TABLES_DIR / f'perm_top15_{_v}.csv', index=False)\n"
    "    _out['shap_summary'].to_csv(TABLES_DIR / f'shap_full_{_v}.csv', index=False)\n"
    "    _out['permutation'].to_csv(TABLES_DIR / f'perm_full_{_v}.csv', index=False)\n"
    "    fig.fig_shap_summary(_v, _out['shap_values'], _out['X_sample'], top_n=15)\n"
    "    print(f'   top SHAP: {\", \".join(_out[\"shap_summary\"][\"feature\"].head(5).tolist())}')\n"
    "\n"
    "print('[Stage 10.2] Global figures')\n"
    "fig.fig_label_agreement_venn();        print('   - fig_label_agreement_venn')\n"
    "fig.fig_per_project_positive_rates();  print('   - fig_per_project_positive_rates')\n"
    "fig.fig_within_vs_lopo();              print('   - fig_within_vs_lopo')\n"
    "fig.fig_sensitivity_heatmap();         print('   - fig_sensitivity_heatmap')\n"
    "fig.fig_feature_ablation();            print('   - fig_feature_ablation')\n"
    "fig.fig_lopo_per_project();            print('   - fig_lopo_per_project')\n"
    "\n"
    "_within_pred = TABLES_DIR / 'within_project_predictions.parquet'\n"
    "if _within_pred.exists() and hasattr(fig, 'figure_confusion_matrices'):\n"
    "    try:\n"
    "        _per_proj = fig.figure_confusion_matrices(pd.read_parquet(_within_pred))\n"
    "        print(f'   - figure_confusion_matrices ({len(_per_proj):,} per-project rows)')\n"
    "    except Exception as _exc:\n"
    "        print(f'   confusion matrices skipped: {_exc}')\n"
    "_cal_pred = TABLES_DIR / 'calibration_predictions.parquet'\n"
    "if _cal_pred.exists() and hasattr(fig, 'figure_calibration_diagrams'):\n"
    "    try:\n"
    "        fig.figure_calibration_diagrams(pd.read_parquet(_cal_pred))\n"
    "        print('   - figure_calibration_diagrams')\n"
    "    except Exception as _exc:\n"
    "        print(f'   calibration diagrams skipped: {_exc}')\n"
    "\n"
    "print('[Stage 10.3] Render docs/06_results.md + docs/07_discussion.md')\n"
    "_results_path = render_results()\n"
    "_disc_path = render_discussion_scaffold()\n"
    "print(f'   - {_results_path}')\n"
    "print(f'   - {_disc_path}')\n"
    "\n"
    "RUNTIMES['stage_10_report'] = round(time.time() - _t0, 2)\n"
    "print(f'[Stage 10] Elapsed: {RUNTIMES[\"stage_10_report\"]} s')\n"
))

cells.append(code(
    "# Render every figure inline and preview the auto-generated results report.\n"
    "from IPython.display import Image\n"
    "from config import FIGURES_DIR, DOCS_DIR\n"
    "\n"
    "_figs_in_order = [\n"
    "    'fig_label_agreement_venn',\n"
    "    'fig_per_project_positive_rates',\n"
    "    'fig_within_vs_lopo',\n"
    "    'fig_sensitivity_heatmap',\n"
    "    'fig_feature_ablation',\n"
    "    'fig_shap_consequence',\n"
    "    'fig_shap_severity',\n"
    "    'fig_shap_szz',\n"
    "    'fig_lopo_per_project',\n"
    "]\n"
    "for _name in _figs_in_order:\n"
    "    _png = FIGURES_DIR / f'{_name}.png'\n"
    "    if _png.exists():\n"
    "        display(Markdown(f'**{_name}**'))\n"
    "        display(Image(filename=str(_png)))\n"
    "    else:\n"
    "        print(f'(missing) {_png}')\n"
    "\n"
    "_results_md = (DOCS_DIR / '06_results.md').read_text(encoding='utf-8')\n"
    "_preview = '\\n'.join(_results_md.splitlines()[:80])\n"
    "display(Markdown('---\\n## Preview: docs/06_results.md (first 80 lines)\\n'))\n"
    "display(Markdown(_preview))\n"
))


# ===== 15. Finalisation ====================================================
cells.append(md(
    '<a id="finalisation"></a>\n'
    "## Finalisation - runtime, zip and copy to Drive\n"
    "\n"
    "Persist a per-stage runtime record, build a single zip archive containing only the experimental outputs, and copy it back to Drive at `MyDrive/td_pipeline/outputs/`.\n"
))

cells.append(code(
    "# Persist per-stage runtime record.\n"
    "_total_s = round(sum(RUNTIMES.values()), 2)\n"
    "_runtime_record = {\n"
    "    'per_stage_seconds': RUNTIMES,\n"
    "    'total_seconds': _total_s,\n"
    "    'total_minutes': round(_total_s / 60, 2),\n"
    "    'finished_at_utc': datetime.datetime.utcnow().isoformat() + 'Z',\n"
    "}\n"
    "(CONTENT / 'results' / 'runtime.json').write_text(\n"
    "    json.dumps(_runtime_record, indent=2), encoding='utf-8')\n"
    "print(json.dumps(_runtime_record, indent=2))\n"
))

cells.append(code(
    "# Build the zip of experimental artefacts (no thesis prose included).\n"
    "import zipfile, glob\n"
    "\n"
    "_ts = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')\n"
    "_zip_name = f'td_pipeline_outputs_{_ts}.zip'\n"
    "_zip_path = CONTENT / _zip_name\n"
    "\n"
    "_INCLUDE_GLOBS = [\n"
    "    'data/processed/*.parquet',\n"
    "    'results/tables/*.csv',\n"
    "    'results/tables/db_samples/*.csv',\n"
    "    'results/figures/*.png',\n"
    "    'results/figures/*.pdf',\n"
    "    'results/env_receipt.json',\n"
    "    'results/runtime.json',\n"
    "]\n"
    "_INCLUDE_DOCS = ['docs/06_results.md', 'docs/07_discussion.md']\n"
    "\n"
    "with zipfile.ZipFile(_zip_path, 'w', zipfile.ZIP_DEFLATED) as _zf:\n"
    "    _written = 0\n"
    "    for _g in _INCLUDE_GLOBS:\n"
    "        for _p in sorted(glob.glob(str(CONTENT / _g))):\n"
    "            _arcname = Path(_p).relative_to(CONTENT).as_posix()\n"
    "            _zf.write(_p, _arcname); _written += 1\n"
    "    for _doc_rel in _INCLUDE_DOCS:\n"
    "        _p = CONTENT / _doc_rel\n"
    "        if _p.exists():\n"
    "            _zf.write(_p, _doc_rel); _written += 1\n"
    "\n"
    "print(f'Wrote {_written} files to {_zip_path}')\n"
    "print(f'Zip size: {_zip_path.stat().st_size / (1024**2):.1f} MB')\n"
))

cells.append(code(
    "# Copy the zip to Drive so the user retains it after the Colab VM is recycled.\n"
    "_outputs_dir = DRIVE_ROOT / 'outputs'\n"
    "_outputs_dir.mkdir(parents=True, exist_ok=True)\n"
    "_dest = _outputs_dir / _zip_path.name\n"
    "shutil.copy2(_zip_path, _dest)\n"
    "print(f'Copied to Drive: {_dest}')\n"
))

cells.append(md(
    "## Artefact inventory\n"
    "\n"
    "After successful execution, the zip in `MyDrive/td_pipeline/outputs/` contains exactly:\n"
    "\n"
    "**`data/processed/`** - intermediate parquet tables\n"
    "- `project_snapshots.parquet` - per-project snapshot dates and eligibility.\n"
    "- `clean_git_commits.parquet`, `clean_git_commits_changes.parquet`, `clean_sonar_issues.parquet`, `clean_sonar_measures.parquet`, `clean_szz.parquet`, `clean_jira_issues.parquet` - cleaned source tables.\n"
    "- `labels_consequence.parquet`, `labels_severity.parquet`, `labels_szz.parquet` - the three label variants.\n"
    "- `features_static.parquet`, `features_historical.parquet` - feature matrices.\n"
    "- `dataset_consequence.parquet`, `dataset_severity.parquet`, `dataset_szz.parquet` - per-variant training matrices (after leakage audit).\n"
    "\n"
    "**`results/tables/`** - all CSV result tables\n"
    "- `db_schema.csv`, `db_table_counts.csv`, `db_samples/<table>.csv` (Stage 1).\n"
    "- `project_stats.csv` (Stage 2).\n"
    "- `path_overlap_report.csv` (Stage 3).\n"
    "- `label_summary.csv`, `label_agreement.csv` (Stage 4).\n"
    "- `feature_summary.csv` (Stage 5).\n"
    "- `dataset_summary.csv` (Stage 6).\n"
    "- `within_project_folds.csv`, `within_project_summary.csv` (Stage 7).\n"
    "- `lopo_folds.csv`, `lopo_summary.csv`, `lopo_vs_within.csv` (Stage 8).\n"
    "- `sensitivity_consequence.csv`, `feature_ablation.csv` (Stage 9).\n"
    "- `shap_top15_<variant>.csv`, `shap_full_<variant>.csv`, `perm_top15_<variant>.csv`, `perm_full_<variant>.csv` (Stage 10).\n"
    "\n"
    "**`results/figures/`** - 9 figures, each as PNG + PDF (300 DPI)\n"
    "- `fig_label_agreement_venn`, `fig_per_project_positive_rates`, `fig_within_vs_lopo`, `fig_sensitivity_heatmap`, `fig_feature_ablation`, `fig_shap_consequence`, `fig_shap_severity`, `fig_shap_szz`, `fig_lopo_per_project`.\n"
    "\n"
    "**`docs/`** - exactly two auto-generated markdown files\n"
    "- `06_results.md`, `07_discussion.md`.\n"
    "\n"
    "**Top-level**\n"
    "- `results/env_receipt.json` - Python / package versions, DB fingerprint, random seed.\n"
    "- `results/runtime.json` - per-stage wall-clock breakdown.\n"
))


# ---------------------------------------------------------------------------
# Assemble notebook
# ---------------------------------------------------------------------------
notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# ---------------------------------------------------------------------------
# Demo variant: a 5-project subset that runs stages 1-7 in 5-10 minutes on a
# free Colab T4 runtime. Stages 8 (LOPO) and 9 (sensitivity grid + ablation)
# are short-circuited with banner cells - they need many projects to be
# meaningful, and the demo notebook is intended as an "instant replication"
# artefact for examiners and reviewers, not as a substitute for the full
# overnight run.
# ---------------------------------------------------------------------------
_DEMO_PROJECT_FILTER_BANNER = (
    "## DEMO MODE - 5-project subset\n"
    "\n"
    "This notebook is the *demo* variant of the TD pipeline. It runs stages 1-7 "
    "on the following projects only:\n"
    "\n"
    + "".join(f"- `{p}`\n" for p in DEMO_PROJECTS)
    + "\n"
    "Stages 8 (Leave-One-Project-Out) and 9 (sensitivity grid + ablation) are "
    "short-circuited because they require many projects to be statistically "
    "meaningful. Stage 10 still emits a smaller-scope SHAP analysis and a "
    "render of `docs/06_results.md`.\n"
    "\n"
    "Expected wall-clock on a free Colab T4: 5-10 minutes. Use the *full* "
    "notebook (`td_pipeline_colab.ipynb`) for the 22-project, stages-1-10 "
    "thesis-grade reproduction."
)

_DEMO_PROJECT_FILTER_CODE = (
    "# Demo-mode project filter - applied to project_snapshots.parquet so every\n"
    "# downstream stage sees only the 5 demo projects. Also sets TD_PROJECTS\n"
    "# so ``src.data.load_data.list_projects`` returns the same subset when used.\n"
    "import os\n"
    "from config import PROCESSED_DATA_DIR as _PD\n"
    "DEMO_PROJECTS = " + repr(DEMO_PROJECTS) + "\n"
    "os.environ['TD_PROJECTS'] = ','.join(DEMO_PROJECTS)\n"
    "_snap_path = _PD / 'project_snapshots.parquet'\n"
    "if _snap_path.exists():\n"
    "    _df = pd.read_parquet(_snap_path)\n"
    "    _df['eligible'] = _df['eligible'] & _df['project_id'].isin(DEMO_PROJECTS)\n"
    "    _df.to_parquet(_snap_path, index=False)\n"
    "    print(f'[demo] eligible projects after filter: '\n"
    "          f\"{_df.loc[_df['eligible'], 'project_id'].tolist()}\")\n"
    "else:\n"
    "    print('[demo] project_snapshots.parquet not yet built; filter will apply on next run.')\n"
)

_DEMO_STAGE8_BANNER = (
    "## Stage 8 - SKIPPED in demo mode\n"
    "\n"
    "Leave-One-Project-Out cross-project validation needs 10+ projects to "
    "produce meaningful averages. The demo subset of 5 projects would yield "
    "high-variance LOPO numbers that are not informative.\n"
    "\n"
    "Run the *full* notebook (`td_pipeline_colab.ipynb`) for the 22-project "
    "LOPO results reported in the thesis."
)

_DEMO_STAGE9_BANNER = (
    "## Stage 9 - SKIPPED in demo mode\n"
    "\n"
    "The 3 x 3 sensitivity grid retrains 9 LightGBM models on relabelled "
    "datasets, and feature-group ablation retrains 33 models. Both are "
    "expensive and only meaningful at full scale. Run the *full* notebook "
    "for both."
)


def _build_demo_notebook(orig_cells: list[dict]) -> list[dict]:
    """Take a deep copy of the full-notebook cells and patch them for demo mode.

    Edits in order:
    1. Title and overview markdown -> point at the 5-project subset.
    2. Right after Stage 2 (snapshot selection) inject a project-filter cell
       so Stages 3-7 only see the 5 demo projects.
    3. Replace Stage 8 + Stage 9 cells with banner-only "skipped" markdown.
    4. Reduce Optuna TUNING_TRIALS to 5 in any %%writefile cell.
    """
    new_cells = copy.deepcopy(orig_cells)

    # 1. Title and scope updates
    for cell in new_cells:
        if cell["cell_type"] != "markdown":
            continue
        text = "".join(cell["source"])
        if "TD Prediction Pipeline - Replication Notebook" in text and "demo" not in text.lower():
            text = text.replace(
                "# TD Prediction Pipeline - Replication Notebook\n",
                "# TD Prediction Pipeline - Replication Notebook (demo, 5 projects)\n",
            )
            text = text.replace(
                "Reproduces Stages 1-10",
                "Reproduces Stages 1-7 on a 5-project subset",
            )
            cell["source"] = _split_keep_lines(text)
            break

    # 1b. TOC — collision cells ship only with Stage 8; demo replaces Stage 8 entirely.
    for cell in new_cells:
        if cell["cell_type"] != "markdown":
            continue
        text = "".join(cell["source"])
        if "## Table of contents" not in text:
            continue
        text = text.replace(
            "12. [Collision analysis (construct validity)](#collision-analysis)\n",
            "",
        )
        text = text.replace("13. [Stage 9", "12. [Stage 9")
        text = text.replace("14. [Stage 10", "13. [Stage 10")
        text = text.replace("15. [Finalisation", "14. [Finalisation")
        cell["source"] = _split_keep_lines(text)
        break

    # 2. Insert a project-filter markdown + code cell after Stage 2's inspection
    #    cell (the cell that displays the eligible-snapshots dataframe).
    insert_after = None
    for i, cell in enumerate(new_cells):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "_eligible[_disp_cols]" in src and "Eligibility:" in src:
            insert_after = i
            break
    if insert_after is None:
        raise RuntimeError("could not locate Stage 2 inspection cell to insert demo filter after")
    demo_md = md(_DEMO_PROJECT_FILTER_BANNER)
    demo_code = code(_DEMO_PROJECT_FILTER_CODE)
    new_cells = new_cells[: insert_after + 1] + [demo_md, demo_code] + new_cells[insert_after + 1 :]

    # 3. Stage 8 + Stage 9 short-circuit. Find each section's leading markdown,
    #    replace its content with the banner, and turn the immediately following
    #    code cell(s) (until the next "## Stage" or "<a id=\"stage-...\">") into
    #    a single no-op print.
    def _short_circuit(stage_id: str, banner: str) -> None:
        nonlocal new_cells
        start = end = None
        for i, c in enumerate(new_cells):
            text = "".join(c["source"])
            if c["cell_type"] == "markdown" and f'id="{stage_id}"' in text:
                start = i; continue
            if start is not None and c["cell_type"] == "markdown" and (
                'id="stage-' in text or 'id="finalisation"' in text
            ) and i > start:
                end = i; break
        if start is None:
            print(f"[demo] warn: could not find {stage_id} block; left unchanged")
            return
        if end is None:
            end = len(new_cells)
        replacement = [md(f'<a id="{stage_id}"></a>\n' + banner),
                       code(f'print(\"[demo] {stage_id} short-circuited; see full notebook.\")\n')]
        new_cells = new_cells[:start] + replacement + new_cells[end:]

    _short_circuit("stage-8", _DEMO_STAGE8_BANNER)
    _short_circuit("stage-9", _DEMO_STAGE9_BANNER)

    # 4. Reduce TUNING_TRIALS to 5 in any %%writefile config cell.
    for cell in new_cells:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "%%writefile" in src and "TUNING_TRIALS" in src:
            src = src.replace("TUNING_TRIALS = 30", "TUNING_TRIALS = 5  # demo mode\n")
            cell["source"] = _split_keep_lines(src)

    return new_cells


# ---------------------------------------------------------------------------
# Standalone variant: inline every %%writefile cell as a regular code cell so
# the notebook can be edited and re-run in a demo workflow without writing any
# files to disk. The 10 pipeline stages and their artefacts are unchanged.
# ---------------------------------------------------------------------------
_INTERNAL_IMPORT = re.compile(r"^(\s*)from\s+([\w.]+)\s+import\b")


def _is_internal_module(mod: str) -> bool:
    return (
        mod == "config"
        or mod == "src"
        or mod.startswith("src.")
        or mod.startswith(".")
    )


def _strip_internal_imports(text: str) -> str:
    """Drop ``from config|src...|. import ...`` (single-line and parenthesised)."""
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        m = _INTERNAL_IMPORT.match(line)
        if m and _is_internal_module(m.group(2)):
            if "(" in line and ")" not in line:
                while i < len(lines) and ")" not in lines[i]:
                    i += 1
                i += 1
                continue
            while i < len(lines) and lines[i].rstrip().endswith("\\"):
                i += 1
            i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _rewrite_file_dunder(text: str) -> str:
    """Inlined modules cannot rely on ``__file__``; map all bootstrap forms to
    ``CONTENT`` (the notebook-defined project root)."""
    text = text.replace("Path(__file__).resolve().parents[2]", "CONTENT")
    text = text.replace("Path(__file__).parent.parent.parent", "CONTENT")
    text = text.replace("Path(__file__).parent", "CONTENT")
    text = text.replace("Path(__file__)", "CONTENT")
    return text


def _strip_if_main_guard_from_notebook_tool(body: str) -> str:
    """Remove trailing ``if __name__ == ...: main()`` so standalone can call ``main()`` later."""
    lines = body.rstrip().split("\n")
    while lines and lines[-1].strip() == "":
        lines.pop()
    while len(lines) >= 2:
        prev_norm = lines[-2].strip().replace("'", '"')
        if prev_norm == 'if __name__ == "__main__":' and lines[-1].strip() == "main()":
            lines = lines[:-2]
            while lines and lines[-1].strip() == "":
                lines.pop()
            continue
        break
    return "\n".join(lines).rstrip() + "\n"


def _convert_for_standalone(orig_cells: list[dict]) -> list[dict]:
    new_cells: list[dict] = []
    for cell in copy.deepcopy(orig_cells):
        if cell["cell_type"] != "code":
            new_cells.append(cell)
            continue
        src = "".join(cell["source"])
        if src.startswith("%%writefile"):
            first_line = src.split("\n", 1)[0]
            body = src.split("\n", 1)[1] if "\n" in src else ""
            if "/tools/collision_analysis.py" not in first_line:
                body = _rewrite_file_dunder(body)
            body = _strip_internal_imports(body)
            if "/tools/collision_analysis.py" in first_line:
                body = _strip_if_main_guard_from_notebook_tool(body)
            cell["source"] = _split_keep_lines(body)
            new_cells.append(cell)
            continue
        if src.lstrip().startswith("# Final modules cell"):
            cell["source"] = _split_keep_lines(
                "# Sanity-check: config and src/ module names are now in the notebook namespace.\n"
                "print('Inlined modules ready.')\n"
                "print('  PROCESSED_DATA_DIR =', PROCESSED_DATA_DIR)\n"
                "print('  TD_DATASET_PATH    =', TD_DATASET_PATH)\n"
            )
            new_cells.append(cell)
            continue
        cell["source"] = _split_keep_lines(_strip_internal_imports(src))
        new_cells.append(cell)
    return new_cells


def _patch_standalone_markdown(orig_cells: list[dict]) -> list[dict]:
    title_old = "# TD Prediction Pipeline - Replication Notebook\n"
    title_new = "# TD Prediction Pipeline - Replication Notebook (standalone / demo variant)\n"
    scope_old = (
        "**Scope.** Experimentation only. Hand-written thesis prose chapters are *not* produced or copied by this notebook.\n"
    )
    scope_new = (
        "**Scope.** Experimentation only. Hand-written thesis prose chapters are *not* produced or copied by this notebook.\n"
        "\n"
        "**This is the standalone / demo variant.** All `src/` modules and `config.py` are inlined as regular code cells (no `%%writefile`). Edit any function cell and re-run it; downstream cells will pick up the new definition immediately, with no kernel restart. Use `td_pipeline_colab.ipynb` (the writefile variant) when you need byte-identical thesis-replication runs that mirror the on-disk repository layout.\n"
    )
    modules_old_1 = (
        "The next 14 cells materialise the production code modules under `/content/src/` using `%%writefile`. Module contents are *verbatim* copies of the repository sources - they are the same code that the local `scripts/01..10` use, ensuring byte-identical experimental results.\n"
    )
    modules_new_1 = (
        "The next 14 cells inline the production code modules verbatim into the notebook namespace - no `%%writefile`, no on-disk `src/` package. Cells appear in topological order, so each module only references names defined in cells above it.\n"
    )
    modules_old_2 = (
        "Each cell is preceded by a one-line caption stating the file path and purpose. After all 14 modules are written, a final cell imports them so subsequent stage cells can call them directly.\n"
    )
    modules_new_2 = (
        "Each cell is preceded by a one-line caption stating the file path and purpose. Editing a function in any of these cells and re-running it propagates immediately to subsequent stages - ideal for demos, debugging or ad-hoc experimentation.\n"
    )
    config_old = (
        "The next cell writes `config.py` to `/content/`. It is the single source of truth for paths, snapshot policy, labelling parameters, feature catalogues and model definitions.\n"
    )
    config_new = (
        "The next cell defines configuration globals - paths, snapshot policy, labelling parameters, feature catalogues, model definitions - directly in the notebook namespace. Every downstream cell reads these names from globals, so editing this cell and re-running it propagates immediately.\n"
    )
    mermaid_old = "Mods[src/ modules writefile]"
    mermaid_new = "Mods[src/ modules inlined]"
    for cell in orig_cells:
        if cell["cell_type"] != "markdown":
            continue
        src = "".join(cell["source"])
        src = src.replace(title_old, title_new)
        src = src.replace(scope_old, scope_new)
        src = src.replace(modules_old_1, modules_new_1)
        src = src.replace(modules_old_2, modules_new_2)
        src = src.replace(config_old, config_new)
        src = src.replace(mermaid_old, mermaid_new)
        cell["source"] = _split_keep_lines(src)
    return orig_cells


def _emit_notebooks(mode: str) -> None:
    """Write notebook JSON files according to ``mode`` (full / demo / standalone / all)."""
    modes = {"full", "demo", "standalone"} if mode == "all" else {mode}
    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)

    if "full" in modes:
        OUT_FULL.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
        sz = OUT_FULL.stat().st_size / 1024
        print(f"Wrote {OUT_FULL} ({sz:.1f} KiB, {len(cells)} cells)")

    if "demo" in modes:
        demo_cells_local = _build_demo_notebook(cells)
        demo_nb = {
            "cells": demo_cells_local,
            "metadata": notebook["metadata"],
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        OUT_DEMO.write_text(json.dumps(demo_nb, indent=1, ensure_ascii=False), encoding="utf-8")
        sz_d = OUT_DEMO.stat().st_size / 1024
        print(f"Wrote {OUT_DEMO} ({sz_d:.1f} KiB, {len(demo_cells_local)} cells)")

    if "standalone" in modes:
        standalone_cells_local = _patch_standalone_markdown(_convert_for_standalone(cells))
        standalone_nb = {
            "cells": standalone_cells_local,
            "metadata": notebook["metadata"],
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        OUT_STANDALONE.write_text(
            json.dumps(standalone_nb, indent=1, ensure_ascii=False),
            encoding="utf-8",
        )
        sz_s = OUT_STANDALONE.stat().st_size / 1024
        print(f"Wrote {OUT_STANDALONE} ({sz_s:.1f} KiB, {len(standalone_cells_local)} cells)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Colab replication notebooks from src/.")
    parser.add_argument(
        "--mode",
        choices=("full", "demo", "standalone", "all"),
        default="all",
        help="Which notebook(s) to emit (default: all three).",
    )
    args = parser.parse_args()
    _emit_notebooks(args.mode)


if __name__ == "__main__":
    main()
