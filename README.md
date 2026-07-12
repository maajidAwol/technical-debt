# Predicting High-Risk Technical Debt in Open-Source Software Projects Using Machine Learning and Code Metrics

Replication package for the paper *"Predicting High-Risk Technical Debt in
Open-Source Software Projects Using Machine Learning and Code Metrics"*
(Abdulmajid Awol Seid, Tesfay Gidey Hailu — Department of Software
Engineering, Addis Ababa Science and Technology University), submitted to
*Information and Software Technology*.

The framework predicts, at a chosen snapshot date, which Java files of a
project will become **maintenance-intensive within the following six
months**. Labels combine static analysis evidence (SonarQube) with
historical maintenance signals (bug-fix activity, churn, contributors), and
a 27-feature LightGBM model is evaluated under both within-project and
leave-one-project-out (LOPO) cross-project validation on 22 Apache Java
projects (12,449 files) drawn from the Technical Debt Dataset v2.0.

**Headline result:** on previously unseen projects, inspecting only the top
20% of files ranked by the model recovers **82.4%** of the files that
become maintenance-intensive (CE@20 = 0.8242), roughly **4.1×** the
recovery of random inspection at the same budget.

## Repository layout

| Path | Contents |
|---|---|
| `run_pipeline.py` | Orchestrator — runs the numbered pipeline stages with logging and caching (`python run_pipeline.py --list`) |
| `config.py` | All paths, snapshot policy, labeling weights, feature catalogue, model grid |
| `scripts/` | Pipeline stages `01`–`13`: inspect DB → profile projects → clean → label → features → dataset → train → tune → LOPO → ablation → report → persist → verify → score GitHub repo |
| `src/` | Library code: `data/` (cleaning, labeling, SZZ), `features/` (5 families), `models/` (training, LOPO), `analysis/` (SHAP, ablation), `inference/` (scoring new repos) |
| `models/` | Trained LightGBM model (`best_model.pkl`), scaler, feature names, `model_card.json`, `score_project.py` |
| `notebooks/td_pipeline_colab.ipynb` | End-to-end scoring notebook (Colab-ready) |
| `data/processed/` | Final artifacts: `dataset_final.parquet` (12,449 × 27+), `labels.parquet`, per-family feature tables, derived signal weights |
| `results/tables/` | Every number in the paper: LOPO results, ablation, feature importance, per-project statistics |
| `results/figures/` | All paper figures |
| `.github/workflows/risk_scoring.yml` | CI workflow that scores a repository with the persisted model |

## Quick start

```bash
git clone https://github.com/maajidAwol/technical-debt.git
cd technical-debt
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Score any Java repository with the trained model

No raw data needed — the persisted model in `models/` is self-contained:

```bash
python src/inference/github_scorer.py --repo-url https://github.com/apache/commons-cli --output risk_report.csv
```

or open `notebooks/td_pipeline_colab.ipynb` in Google Colab.

### Reproduce the full experiment

1. Download the **Technical Debt Dataset v2.0** SQLite database
   (<https://github.com/clowee/The-Technical-Debt-Dataset>) and place it at
   `data/raw/td_V2.db` (~3 GB; not distributed here, per its license).
2. Run the pipeline:

```bash
python run_pipeline.py            # all stages
python run_pipeline.py --list     # see the stage catalogue
python run_pipeline.py --from 7   # e.g. retrain from the modeling stage
```

Stages 1–4 rebuild the cleaned intermediate tables from the raw database;
stages 5–6 engineer features and assemble `dataset_final.parquet`; stages
7–12 train, tune, evaluate (within-project + LOPO), ablate, and persist.
Every stage writes its outputs to `data/processed/` or `results/` and its
log to `results/run_logs/`. Random seeds are fixed (`RANDOM_STATE = 42` in
`config.py`).

The small final artifacts (`dataset_final.parquet`, `labels.parquet`,
feature tables) are committed, so stages 7 onward can be run **without**
downloading the raw database.

## Model card (summary)

| Property | Value |
|---|---|
| Model | LightGBM (tuned; see `models/model_card.json`) |
| Features | 27, in 5 families (static debt, size/complexity, historical change, prior defects, co-change graph) |
| Label | Dual-signal combined-weight (6 signals, empirically weighted, 0.50 threshold) |
| LOPO F1 / ROC-AUC / PR-AUC | 0.7261 / 0.9679 / 0.8871 |
| LOPO CE@20 | 0.8242 |
| Within-project F1 | 0.8686 |
| Training corpus | 22 Apache Java projects, 12,449 files |

## Citation

If you use this package, please cite the paper (bibliographic details will
be added upon publication).

## License

MIT — see [LICENSE](LICENSE). The Technical Debt Dataset v2.0 is distributed
separately by its authors under its own terms.
