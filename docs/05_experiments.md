# 05. Experiments

*Paper-ready source for Methodology Chapter 3.5.*

## Experimental Grid

| Dimension | Values | Count |
|-----------|--------|-------|
| Label variants | consequence, severity, szz | 3 |
| Models | DT, RF, SVM, XGBoost, LightGBM | 5 |
| Observation windows (sensitivity) | 3, 6, 12 months | 3 |
| Percentile thresholds (sensitivity) | 10%, 20%, 30% | 3 |
| Feature groups (ablation) | static, historical, combined | 3 |

## Primary Experiments

1. **Within-project evaluation** (Stage 7): per-project stratified 10-fold CV
   across all 5 models x 3 variants = 15 configurations.
2. **Cross-project (LOPO) evaluation** (Stage 8): leave-one-project-out over
   33 projects; all 5 models x 3 variants = 4950 trained models total.

## Secondary Experiments (Stage 9)

3. **Sensitivity analysis** (consequence variant only): 3 windows x 3 percentiles
   = 9 configurations, LOPO only.
4. **Feature-group ablation**: best model x 3 variants x 3 feature groups = 9
   configurations, LOPO only.
5. **Label agreement**: Cohen's kappa + Jaccard across variant pairs.
6. **Cost-Effectiveness @ top-20%**: prioritization-oriented metric.

## Class Imbalance Handling

- Primary approach: `class_weight='balanced'` (RF, SVM, LGBM, DT)
- XGBoost: `scale_pos_weight` tuned over grid
- SMOTE applied only as a fallback if primary approaches under-perform

## Hyperparameter Tuning

`GridSearchCV` with inner 5-fold CV on training folds. Outer evaluation is
unaffected. Only applied to the two top-performing models from Stage 7
baseline run to keep computation bounded.

## Statistical Significance

Wilcoxon signed-rank test on per-project F1 scores across LOPO folds to
compare model pairs and variant pairs. Effect sizes reported via
Cliff's delta.
