# Thesis Output Mapping

Every pipeline output maps to a specific thesis section or artifact.

| Thesis Chapter | Artifact | Type | Stage |
|----------------|----------|------|-------|
| 1 Introduction | - | Manual | - |
| 2 Literature Review | - | Manual | - |
| 3.1 Research Design | `docs/01_methodology.md` | Text | 0 |
| 3.2 Data Sources | `docs/02_dataset.md` + `results/tables/project_stats.csv` | Text + Table | 2 |
| 3.3 Features | `docs/04_features.md` + `results/tables/feature_catalog.csv` | Text + Table | 5, 6 |
| 3.4 Labeling | `docs/03_labeling.md` + `results/figures/label_overlap_venn.png` | Text + Figure | 4, 9 |
| 3.5 Model Development | `docs/05_experiments.md` | Text | 0 |
| 4.1 Descriptive Stats | `results/tables/project_stats.csv` | Table | 2 |
| 4.2 Label Comparison | `results/tables/label_agreement.csv`, `results/figures/label_overlap_venn.png` | Table + Figure | 9 |
| 4.3 Within-Project | `results/tables/within_project_results.csv` | Table | 7 |
| 4.4 Cross-Project | `results/tables/lopo_results.csv` | Table | 8 |
| 4.5 Sensitivity | `results/tables/sensitivity.csv` | Table | 9 |
| 4.6 Ablation | `results/tables/ablation.csv` | Table | 9 |
| 4.7 Feature Importance | `results/tables/feature_importance_*.csv` + `results/figures/shap_*.png` | Table + Figure | 10 |
| 4.8 CE@20% | `results/tables/ce_at_20.csv` | Table | 9 |
| 5 Discussion | `docs/07_discussion.md` | Text (manual) | - |
| 6 Threats to Validity | `docs/threats_to_validity.md` | Text | 0 |
| 7 Conclusion | - | Manual | - |
| Appendix A Schema | `results/tables/db_schema.csv` | Table | 1 |
| Appendix B Research Log | `RESEARCH_LOG.md` | Text | - |
