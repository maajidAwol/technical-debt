# 01. Research Methodology

*Paper-ready source for the Methodology chapter.*

## Mapping to Proposal Section 3.1

This study follows an experimental research design on empirical OSS data.
The pipeline comprises seven stages (Stages 0-6 data, Stages 7-10 modeling),
each producing an inspectable artifact that gates the next stage.

## Pipeline Overview

```
Dataset (TD v2.0)
    -> Snapshot (Stage 2)
    -> Cleaning (Stage 3)
    -> Labels (Stage 4)    -+
    -> Features (Stage 5)  -+-> Merged Dataset (Stage 6)
                               -> Models (Stage 7-8)
                                  -> Sensitivity + Ablation (Stage 9)
                                     -> Reporting (Stage 10)
```

## Key Methodological Choices

- **Module granularity**: file-level (one row per Java source file per project per snapshot)
- **Temporal split**: features `AUTHOR_DATE <= t`, labels `t < AUTHOR_DATE <= t + W`
- **Snapshot policy**: median master-branch commit date per project
- **Observation window**: 6 months (primary), {3, 6, 12} sensitivity
- **Labeling**: three variants - consequence-oriented (primary), severity baseline, SZZ baseline
- **Features**: static + historical (Proposal Table 1)
- **Models**: Decision Tree, Random Forest, SVM, XGBoost, LightGBM
- **Evaluation**: 10-fold stratified within-project CV + Leave-One-Project-Out cross-project

## Reproducibility

- Random seed: `RANDOM_STATE = 42` in every randomized procedure
- Deterministic SQL queries (ordered)
- Every intermediate artifact saved as Parquet
- Pipeline script `run_pipeline.py` reproduces the entire run end-to-end
- `RESEARCH_LOG.md` timestamps every methodological decision

*This file is auto-referenced from the thesis; its content is directly
usable as the basis for Chapter 3.1.*
