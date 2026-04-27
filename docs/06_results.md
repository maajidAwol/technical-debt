# Chapter 6 - Results

This chapter reports the empirical results of the High-Risk Technical Debt prediction pipeline. All numbers, figures and tables are generated directly from the CSV and parquet artefacts produced by the pipeline (see ``results/tables`` and ``results/figures``) and are fully reproducible by re-running ``scripts/01_inspect_db.py`` through ``scripts/10_report.py``.

## 6.1 Corpus and temporal snapshot

The study covers **22 eligible Apache Java projects** from the Technical Debt Dataset v2.0. For every project we compute a per-project snapshot ``t`` equal to the median commit date; features are restricted to events on or before ``t`` and labels are derived from an observation window of ``OBSERVATION_WINDOW_MONTHS`` (primary: 6 months).

Project snapshot details (all 22 eligible projects):

| project                       | snapshot_t   |   pre_commits |   post_commits |   files_pre |   authors_pre |
|:------------------------------|:-------------|--------------:|---------------:|------------:|--------------:|
| org.apache:archiva            | 2012-01-05   |          4348 |           1095 |        3543 |            24 |
| org.apache:batik              | 2001-12-14   |          1763 |            374 |        2870 |            11 |
| org.apache:bcel               | 2015-08-16   |           877 |            217 |         584 |            20 |
| org.apache:cayenne            | 2012-12-04   |          3367 |            135 |        4258 |            14 |
| org.apache:cocoon             | 2005-03-19   |          6581 |            580 |        6760 |            50 |
| org.apache:codec              | 2012-08-27   |          1073 |            133 |         286 |            23 |
| org.apache:commons-cli        | 2008-11-10   |           511 |             66 |         229 |            23 |
| org.apache:commons-fileupload | 2013-03-11   |           591 |            167 |          98 |            28 |
| org.apache:commons-jelly      | 2003-02-07   |           973 |             85 |         622 |            15 |
| org.apache:commons-jexl       | 2009-12-11   |           964 |            116 |         470 |            21 |
| org.apache:configuration      | 2013-06-01   |          1707 |            184 |         648 |            21 |
| org.apache:daemon             | 2010-10-02   |           626 |            101 |         228 |            18 |
| org.apache:dbcp               | 2014-02-07   |          1233 |            165 |         186 |            31 |
| org.apache:digester           | 2011-02-19   |          1097 |            777 |         434 |            28 |
| org.apache:felix              | 2011-06-29   |          7779 |            606 |        4564 |            38 |
| org.apache:hive               | 2015-12-20   |          7867 |            865 |       13771 |           118 |
| org.apache:httpclient         | 2012-06-07   |          1645 |            127 |         814 |             9 |
| org.apache:httpcore           | 2012-01-17   |          1803 |             70 |         826 |             8 |
| org.apache:net                | 2011-03-25   |          1249 |            162 |         538 |            22 |
| org.apache:thrift             | 2013-06-25   |          3193 |            178 |        1525 |            55 |
| org.apache:vfs                | 2012-11-15   |          1537 |             76 |         485 |            21 |
| org.apache:zookeeper          | 2014-04-04   |          1181 |             91 |        1293 |            13 |

## 6.2 Three-variant labeling

Each (project, basename) pair is labeled three ways: a **consequence** label (top 20% by weighted risk score over bug-fix commits, future churn, SZZ events in the 6-month window), a **severity** label (any open SonarQube BLOCKER or CRITICAL issue at ``t``), and an **SZZ** label (touched by a fault-fixing commit inside the window).

### 6.2.1 Per-project positive rates

| project_id                    |   n_basenames |   consequence_positives |   consequence_rate_pct |   severity_positives |   severity_rate_pct |   szz_positives |   szz_rate_pct |
|:------------------------------|--------------:|------------------------:|-----------------------:|---------------------:|--------------------:|----------------:|---------------:|
| org.apache:archiva            |          1585 |                     309 |                  19.50 |                   88 |                5.55 |              15 |           0.95 |
| org.apache:batik              |          1712 |                     341 |                  19.92 |                  225 |               13.14 |               0 |           0.00 |
| org.apache:bcel               |           471 |                      93 |                  19.75 |                   60 |               12.74 |              18 |           3.82 |
| org.apache:cayenne            |          3570 |                     330 |                   9.24 |                  189 |                5.29 |               5 |           0.14 |
| org.apache:cocoon             |          2702 |                     221 |                   8.18 |                  117 |                4.33 |               0 |           0.00 |
| org.apache:codec              |           117 |                      24 |                  20.51 |                    2 |                1.71 |               3 |           2.56 |
| org.apache:commons-cli        |           159 |                      32 |                  20.13 |                   11 |                6.92 |               0 |           0.00 |
| org.apache:commons-fileupload |            50 |                       9 |                  18.00 |                    4 |                8.00 |               6 |          12.00 |
| org.apache:commons-jelly      |           419 |                      61 |                  14.56 |                   13 |                3.10 |               3 |           0.72 |
| org.apache:commons-jexl       |           158 |                      32 |                  20.25 |                   16 |               10.13 |              10 |           6.33 |
| org.apache:configuration      |           408 |                      81 |                  19.85 |                   25 |                6.13 |               4 |           0.98 |
| org.apache:daemon             |            12 |                       3 |                  25.00 |                    6 |               50.00 |               0 |           0.00 |
| org.apache:dbcp               |           113 |                      23 |                  20.35 |                   22 |               19.47 |               9 |           7.96 |
| org.apache:digester           |           333 |                      67 |                  20.12 |                   21 |                6.31 |               0 |           0.00 |
| org.apache:felix              |          3737 |                     378 |                  10.12 |                  574 |               15.36 |             111 |           2.97 |
| org.apache:hive               |          5307 |                    1040 |                  19.60 |                  669 |               12.61 |               0 |           0.00 |
| org.apache:httpclient         |           697 |                     140 |                  20.09 |                   59 |                8.46 |              24 |           3.44 |
| org.apache:httpcore           |           782 |                     111 |                  14.19 |                   41 |                5.24 |              13 |           1.66 |
| org.apache:net                |           250 |                      50 |                  20.00 |                   42 |               16.80 |              10 |           4.00 |
| org.apache:thrift             |           179 |                      11 |                   6.15 |                   25 |               13.97 |               3 |           1.68 |
| org.apache:vfs                |           405 |                      79 |                  19.51 |                   38 |                9.38 |               1 |           0.25 |
| org.apache:zookeeper          |           745 |                      98 |                  13.15 |                   58 |                7.79 |              62 |           8.32 |

See Figure ``fig_per_project_positive_rates``.

### 6.2.2 Label agreement

Agreement between the three variants is low (Cohen's kappa in [0.05, 0.21]), confirming that they identify largely disjoint file sets:

| variant_a   | variant_b   |   positives_a |   positives_b |   agreement_pct |   cohen_kappa |   jaccard |   intersection |   union |
|:------------|:------------|--------------:|--------------:|----------------:|--------------:|----------:|---------------:|--------:|
| consequence | severity    |          3533 |          2305 |         82.9500 |        0.2092 |    0.1775 |            880 |    4958 |
| consequence | szz         |          3533 |           297 |         86.4400 |        0.1337 |    0.0831 |            294 |    3536 |
| severity    | szz         |          2305 |           297 |         89.8900 |        0.0498 |    0.0367 |             92 |    2510 |

See Figure ``fig_label_agreement_venn``.

## 6.3 Dataset-build leakage audit

After merging features with labels, each variant's dataset is checked for label-leakage. ``SEVERITY_LEAKY_FEATURES`` (the six per-severity counts plus ``max_severity_rank``) are dropped from the severity dataset; no leaky features remain in any variant.

| variant     |   rows |   columns_total |   n_features |   positive_rate_pct |   positives |   retained_leaky_cols |   cols_with_any_na |
|:------------|-------:|----------------:|-------------:|--------------------:|------------:|----------------------:|-------------------:|
| consequence |  23911 |              66 |           62 |               14.78 |        3533 |                   nan |                  0 |
| severity    |  23911 |              59 |           56 |                9.64 |        2305 |                   nan |                  0 |
| szz         |  23911 |              65 |           62 |                1.24 |         297 |                   nan |                  0 |

## 6.4 Within-project 10-fold cross-validation

Stratified 10-fold cross-validation on the combined (22-project, 23,911-row) dataset. Mean metrics across folds:

| variant     | model               |   precision_mean |   recall_mean |   f1_mean |   roc_auc_mean |   pr_auc_mean |   mcc_mean |   ce_at_20_mean |
|:------------|:--------------------|-----------------:|--------------:|----------:|---------------:|--------------:|-----------:|----------------:|
| consequence | decision_tree       |            0.487 |         0.470 |     0.478 |          0.692 |         0.308 |      0.390 |           0.500 |
| consequence | lightgbm            |            0.481 |         0.734 |     0.581 |          0.897 |         0.652 |      0.506 |           0.702 |
| consequence | logistic_regression |            0.356 |         0.737 |     0.480 |          0.827 |         0.505 |      0.390 |           0.586 |
| consequence | random_forest       |            0.742 |         0.407 |     0.525 |          0.900 |         0.663 |      0.497 |           0.716 |
| consequence | xgboost             |            0.723 |         0.440 |     0.547 |          0.899 |         0.662 |      0.510 |           0.703 |
| severity    | decision_tree       |            0.645 |         0.623 |     0.633 |          0.793 |         0.439 |      0.595 |           0.668 |
| severity    | lightgbm            |            0.601 |         0.849 |     0.704 |          0.975 |         0.816 |      0.679 |           0.960 |
| severity    | logistic_regression |            0.523 |         0.865 |     0.652 |          0.962 |         0.757 |      0.630 |           0.924 |
| severity    | random_forest       |            0.804 |         0.591 |     0.680 |          0.974 |         0.811 |      0.662 |           0.964 |
| severity    | xgboost             |            0.780 |         0.641 |     0.704 |          0.976 |         0.817 |      0.679 |           0.970 |
| szz         | decision_tree       |            0.183 |         0.185 |     0.182 |          0.587 |         0.046 |      0.173 |           0.309 |
| szz         | lightgbm            |            0.286 |         0.266 |     0.273 |          0.926 |         0.230 |      0.266 |           0.889 |
| szz         | logistic_regression |            0.052 |         0.832 |     0.098 |          0.922 |         0.191 |      0.178 |           0.835 |
| szz         | random_forest       |            0.317 |         0.040 |     0.068 |          0.927 |         0.251 |      0.105 |           0.922 |
| szz         | xgboost             |            0.395 |         0.100 |     0.155 |          0.937 |         0.241 |      0.190 |           0.906 |

### Within-project highlights
- **Consequence**: best model = **lightgbm**, F1=0.581, CE@20=0.702.
- **Severity**: best model = **lightgbm**, F1=0.704, CE@20=0.960.
- **SZZ**: best model = **lightgbm**, F1=0.273, CE@20=0.889.

## 6.5 Leave-One-Project-Out cross-project validation

For each variant and model we train on 21 projects and test on the held-out project, repeating for every project. The SZZ variant covers 16/22 projects because 6 projects have zero SZZ positives in their observation window.

| variant     | model               |   n_projects |   precision_mean |   recall_mean |   f1_mean |   roc_auc_mean |   pr_auc_mean |   mcc_mean |   ce_at_20_mean |
|:------------|:--------------------|-------------:|-----------------:|--------------:|----------:|---------------:|--------------:|-----------:|----------------:|
| consequence | decision_tree       |           22 |            0.285 |         0.285 |     0.275 |          0.581 |         0.221 |      0.159 |           0.331 |
| consequence | lightgbm            |           22 |            0.373 |         0.533 |     0.418 |          0.776 |         0.466 |      0.297 |           0.484 |
| consequence | logistic_regression |           22 |            0.328 |         0.652 |     0.375 |          0.780 |         0.483 |      0.248 |           0.497 |
| consequence | random_forest       |           22 |            0.604 |         0.149 |     0.222 |          0.794 |         0.493 |      0.237 |           0.526 |
| consequence | xgboost             |           22 |            0.518 |         0.277 |     0.329 |          0.780 |         0.466 |      0.277 |           0.509 |
| severity    | decision_tree       |           22 |            0.564 |         0.438 |     0.484 |          0.701 |         0.324 |      0.446 |           0.505 |
| severity    | lightgbm            |           22 |            0.606 |         0.700 |     0.613 |          0.958 |         0.708 |      0.591 |           0.872 |
| severity    | logistic_regression |           22 |            0.529 |         0.774 |     0.608 |          0.950 |         0.705 |      0.587 |           0.874 |
| severity    | random_forest       |           22 |            0.756 |         0.337 |     0.450 |          0.960 |         0.716 |      0.465 |           0.881 |
| severity    | xgboost             |           22 |            0.727 |         0.501 |     0.569 |          0.959 |         0.717 |      0.555 |           0.898 |
| szz         | decision_tree       |           16 |            0.108 |         0.047 |     0.051 |          0.516 |         0.059 |      0.047 |           0.198 |
| szz         | lightgbm            |           16 |            0.032 |         0.017 |     0.014 |          0.706 |         0.130 |      0.011 |           0.496 |
| szz         | logistic_regression |           16 |            0.164 |         0.465 |     0.113 |          0.786 |         0.164 |      0.104 |           0.562 |
| szz         | random_forest       |           16 |            0.000 |         0.000 |     0.000 |          0.781 |         0.132 |      0.000 |           0.605 |
| szz         | xgboost             |           16 |            0.008 |         0.004 |     0.005 |          0.728 |         0.147 |      0.005 |           0.488 |

### 6.5.1 Generalization gap

Difference between within-project and LOPO mean performance:

| variant     | model               |   precision_within |   precision_lopo |   precision_gap |   recall_within |   recall_lopo |   recall_gap |   f1_within |   f1_lopo |   f1_gap |   roc_auc_within |   roc_auc_lopo |   roc_auc_gap |   pr_auc_within |   pr_auc_lopo |   pr_auc_gap |   mcc_within |   mcc_lopo |   mcc_gap |   ce_at_20_within |   ce_at_20_lopo |   ce_at_20_gap |
|:------------|:--------------------|-------------------:|-----------------:|----------------:|----------------:|--------------:|-------------:|------------:|----------:|---------:|-----------------:|---------------:|--------------:|----------------:|--------------:|-------------:|-------------:|-----------:|----------:|------------------:|----------------:|---------------:|
| consequence | decision_tree       |              0.487 |            0.285 |           0.202 |           0.470 |         0.285 |        0.185 |       0.478 |     0.275 |    0.203 |            0.692 |          0.581 |         0.111 |           0.308 |         0.221 |        0.088 |        0.390 |      0.159 |     0.231 |             0.500 |           0.331 |          0.170 |
| consequence | lightgbm            |              0.481 |            0.373 |           0.108 |           0.734 |         0.533 |        0.201 |       0.581 |     0.418 |    0.163 |            0.897 |          0.776 |         0.121 |           0.652 |         0.466 |        0.186 |        0.506 |      0.297 |     0.209 |             0.702 |           0.484 |          0.218 |
| consequence | logistic_regression |              0.356 |            0.328 |           0.028 |           0.737 |         0.652 |        0.085 |       0.480 |     0.375 |    0.105 |            0.827 |          0.780 |         0.047 |           0.505 |         0.483 |        0.022 |        0.390 |      0.248 |     0.142 |             0.586 |           0.497 |          0.089 |
| consequence | random_forest       |              0.742 |            0.604 |           0.138 |           0.407 |         0.149 |        0.258 |       0.525 |     0.222 |    0.303 |            0.900 |          0.794 |         0.106 |           0.663 |         0.493 |        0.170 |        0.497 |      0.237 |     0.260 |             0.716 |           0.526 |          0.189 |
| consequence | xgboost             |              0.723 |            0.518 |           0.205 |           0.440 |         0.277 |        0.164 |       0.547 |     0.329 |    0.218 |            0.899 |          0.780 |         0.118 |           0.662 |         0.466 |        0.197 |        0.510 |      0.277 |     0.232 |             0.703 |           0.509 |          0.194 |
| severity    | decision_tree       |              0.645 |            0.564 |           0.081 |           0.623 |         0.438 |        0.185 |       0.633 |     0.484 |    0.150 |            0.793 |          0.701 |         0.092 |           0.439 |         0.324 |        0.115 |        0.595 |      0.446 |     0.149 |             0.668 |           0.505 |          0.163 |
| severity    | lightgbm            |              0.601 |            0.606 |          -0.004 |           0.849 |         0.700 |        0.149 |       0.704 |     0.613 |    0.091 |            0.975 |          0.958 |         0.016 |           0.816 |         0.708 |        0.107 |        0.679 |      0.591 |     0.088 |             0.960 |           0.872 |          0.087 |
| severity    | logistic_regression |              0.523 |            0.529 |          -0.006 |           0.865 |         0.774 |        0.091 |       0.652 |     0.608 |    0.044 |            0.962 |          0.950 |         0.012 |           0.757 |         0.705 |        0.052 |        0.630 |      0.587 |     0.043 |             0.924 |           0.874 |          0.049 |
| severity    | random_forest       |              0.804 |            0.756 |           0.048 |           0.591 |         0.337 |        0.254 |       0.680 |     0.450 |    0.230 |            0.974 |          0.960 |         0.014 |           0.811 |         0.716 |        0.095 |        0.662 |      0.465 |     0.197 |             0.964 |           0.881 |          0.083 |
| severity    | xgboost             |              0.780 |            0.727 |           0.053 |           0.641 |         0.501 |        0.140 |       0.704 |     0.569 |    0.135 |            0.976 |          0.959 |         0.016 |           0.817 |         0.717 |        0.101 |        0.679 |      0.555 |     0.124 |             0.970 |           0.898 |          0.072 |
| szz         | decision_tree       |              0.183 |            0.108 |           0.075 |           0.185 |         0.047 |        0.137 |       0.182 |     0.051 |    0.131 |            0.587 |          0.516 |         0.071 |           0.046 |         0.059 |       -0.013 |        0.173 |      0.047 |     0.126 |             0.309 |           0.198 |          0.111 |
| szz         | lightgbm            |              0.286 |            0.032 |           0.254 |           0.266 |         0.017 |        0.249 |       0.273 |     0.014 |    0.259 |            0.926 |          0.706 |         0.219 |           0.230 |         0.130 |        0.100 |        0.266 |      0.011 |     0.255 |             0.889 |           0.496 |          0.393 |
| szz         | logistic_regression |              0.052 |            0.164 |          -0.112 |           0.832 |         0.465 |        0.367 |       0.098 |     0.113 |   -0.015 |            0.922 |          0.786 |         0.136 |           0.191 |         0.164 |        0.026 |        0.178 |      0.104 |     0.073 |             0.835 |           0.562 |          0.273 |
| szz         | random_forest       |              0.317 |            0.000 |           0.317 |           0.040 |         0.000 |        0.040 |       0.068 |     0.000 |    0.068 |            0.927 |          0.781 |         0.147 |           0.251 |         0.132 |        0.118 |        0.105 |      0.000 |     0.105 |             0.922 |           0.605 |          0.317 |
| szz         | xgboost             |              0.395 |            0.008 |           0.388 |           0.100 |         0.004 |        0.096 |       0.155 |     0.005 |    0.150 |            0.937 |          0.728 |         0.209 |           0.241 |         0.147 |        0.094 |        0.190 |      0.005 |     0.185 |             0.906 |           0.488 |          0.418 |

- **Consequence**: best LOPO model = **lightgbm**, F1=0.418, CE@20=0.484.
- **Severity** (best LOPO): **lightgbm**, F1=0.613, CE@20=0.872.
- **SZZ** (best LOPO): **logistic_regression**, F1=0.113, CE@20=0.562.

See Figures ``fig_within_vs_lopo`` and ``fig_lopo_per_project``.

## 6.6 Sensitivity to labeling parameters

The consequence-variant default is (window=6 months, percentile=top 20%). The grid below shows LightGBM 10-fold CV performance across a 3x3 parameter sweep:

|   window_months |   percentile |   positive_rate_pct |   f1_mean |   roc_auc_mean |   pr_auc_mean |   ce_at_20_mean |
|----------------:|-------------:|--------------------:|----------:|---------------:|--------------:|----------------:|
|           3.000 |       10.000 |               7.710 |     0.478 |          0.894 |         0.514 |           0.765 |
|           3.000 |       20.000 |              10.760 |     0.565 |          0.911 |         0.648 |           0.767 |
|           3.000 |       30.000 |              11.760 |     0.589 |          0.917 |         0.680 |           0.773 |
|           6.000 |       10.000 |               9.650 |     0.489 |          0.887 |         0.519 |           0.730 |
|           6.000 |       20.000 |              14.780 |     0.581 |          0.897 |         0.652 |           0.702 |
|           6.000 |       30.000 |              19.390 |     0.662 |          0.913 |         0.754 |           0.686 |
|          12.000 |       10.000 |              10.000 |     0.497 |          0.883 |         0.548 |           0.726 |
|          12.000 |       20.000 |              18.970 |     0.630 |          0.894 |         0.705 |           0.659 |
|          12.000 |       30.000 |              25.150 |     0.709 |          0.907 |         0.792 |           0.620 |

See Figure ``fig_sensitivity_heatmap``.

## 6.7 Feature-group ablation

Three feature groups are defined: ``static_sonar`` (per-basename SonarQube aggregates at ``t``), ``historical`` (pre-``t`` Git commit process metrics), and ``project_context`` (project-level SonarQube measures at the most recent analysis <= ``t``).

| variant     | group           | mode                 |   n_features |   precision_mean |   recall_mean |   f1_mean |   roc_auc_mean |   pr_auc_mean |   mcc_mean |   ce_at_20_mean |
|:------------|:----------------|:---------------------|-------------:|-----------------:|--------------:|----------:|---------------:|--------------:|-----------:|----------------:|
| consequence | all             | all_features         |           61 |            0.481 |         0.734 |     0.581 |          0.897 |         0.652 |      0.506 |           0.702 |
| consequence | static_sonar    | only_this_group      |           16 |            0.329 |         0.661 |     0.439 |          0.792 |         0.450 |      0.332 |           0.527 |
| consequence | static_sonar    | leave_out_this_group |           45 |            0.468 |         0.751 |     0.576 |          0.891 |         0.631 |      0.502 |           0.694 |
| consequence | historical      | only_this_group      |           15 |            0.443 |         0.744 |     0.555 |          0.881 |         0.610 |      0.478 |           0.670 |
| consequence | historical      | leave_out_this_group |           46 |            0.382 |         0.681 |     0.489 |          0.830 |         0.524 |      0.395 |           0.602 |
| consequence | project_context | only_this_group      |           30 |            0.195 |         0.665 |     0.301 |          0.590 |         0.174 |      0.133 |           0.229 |
| consequence | project_context | leave_out_this_group |           31 |            0.463 |         0.739 |     0.569 |          0.888 |         0.630 |      0.493 |           0.685 |
| severity    | all             | all_features         |           55 |            0.601 |         0.849 |     0.704 |          0.975 |         0.816 |      0.679 |           0.960 |
| severity    | static_sonar    | only_this_group      |           10 |            0.510 |         0.879 |     0.646 |          0.969 |         0.781 |      0.626 |           0.941 |
| severity    | static_sonar    | leave_out_this_group |           45 |            0.357 |         0.712 |     0.476 |          0.891 |         0.527 |      0.431 |           0.725 |
| severity    | historical      | only_this_group      |           15 |            0.345 |         0.693 |     0.461 |          0.876 |         0.503 |      0.413 |           0.701 |
| severity    | historical      | leave_out_this_group |           40 |            0.544 |         0.881 |     0.672 |          0.972 |         0.802 |      0.652 |           0.951 |
| severity    | project_context | only_this_group      |           30 |            0.137 |         0.708 |     0.229 |          0.634 |         0.141 |      0.136 |           0.313 |
| severity    | project_context | leave_out_this_group |           25 |            0.597 |         0.846 |     0.700 |          0.974 |         0.809 |      0.674 |           0.957 |
| szz         | all             | all_features         |           61 |            0.286 |         0.266 |     0.273 |          0.926 |         0.230 |      0.266 |           0.889 |
| szz         | static_sonar    | only_this_group      |           16 |            0.027 |         0.225 |     0.048 |          0.634 |         0.052 |      0.045 |           0.495 |
| szz         | static_sonar    | leave_out_this_group |           45 |            0.279 |         0.300 |     0.287 |          0.928 |         0.221 |      0.279 |           0.882 |
| szz         | historical      | only_this_group      |           15 |            0.223 |         0.232 |     0.226 |          0.875 |         0.162 |      0.217 |           0.788 |
| szz         | historical      | leave_out_this_group |           46 |            0.089 |         0.312 |     0.138 |          0.871 |         0.119 |      0.147 |           0.761 |
| szz         | project_context | only_this_group      |           30 |            0.037 |         0.899 |     0.070 |          0.859 |         0.058 |      0.145 |           0.707 |
| szz         | project_context | leave_out_this_group |           31 |            0.265 |         0.225 |     0.241 |          0.879 |         0.174 |      0.235 |           0.778 |

See Figure ``fig_feature_ablation``.

## 6.8 Feature importance (SHAP + permutation)

For each variant we train a single LightGBM classifier on an 80/20 stratified split and compute TreeSHAP values on the test set. The permutation importance is a secondary check (ROC-AUC drop when the column is shuffled). Tables per variant: ``shap_top15_{variant}.csv`` and ``perm_top15_{variant}.csv`` in ``results/tables``. Figures: ``fig_shap_{variant}``.

### 6.8.1 Top-15 SHAP features - consequence

| feature                            |   mean_abs_shap |   mean_shap | sign   |
|:-----------------------------------|----------------:|------------:|:-------|
| pseudo_ncloc_at_t                  |          0.9453 |     -0.1929 | -      |
| days_since_last_change_at_snapshot |          0.9395 |      0.0991 | +      |
| file_age_days_at_snapshot          |          0.3454 |      0.0093 | +      |
| total_commits_pre                  |          0.2002 |      0.0213 | +      |
| avg_change_size_pre                |          0.1686 |     -0.0229 | -      |
| project_duplicated_lines_density   |          0.1517 |      0.0008 | +      |
| n_distinct_rules                   |          0.1398 |      0.0174 | +      |
| project_comment_lines_density      |          0.1318 |     -0.0364 | -      |
| ownership_ratio_pre                |          0.1240 |     -0.0290 | -      |
| code_added_pre                     |          0.1198 |     -0.0258 | -      |
| code_churn_pre                     |          0.1092 |     -0.0088 | -      |
| max_single_commit_churn_pre        |          0.1086 |     -0.0373 | -      |
| std_change_size_pre                |          0.0971 |      0.0180 | +      |
| code_removed_pre                   |          0.0931 |     -0.0159 | -      |
| total_contributors_pre             |          0.0894 |     -0.0008 | -      |

### 6.8.2 Top-15 SHAP features - severity

| feature                            |   mean_abs_shap |   mean_shap | sign   |
|:-----------------------------------|----------------:|------------:|:-------|
| total_debt_minutes                 |          3.2045 |      0.7607 | +      |
| n_distinct_rules                   |          0.4410 |      0.1326 | +      |
| max_single_commit_churn_pre        |          0.3063 |     -0.0109 | -      |
| n_bug                              |          0.2380 |      0.1485 | +      |
| file_age_days_at_snapshot          |          0.1741 |     -0.0189 | -      |
| days_since_last_change_at_snapshot |          0.1651 |     -0.0184 | -      |
| avg_change_size_pre                |          0.1499 |     -0.0038 | -      |
| code_added_pre                     |          0.1469 |      0.0070 | +      |
| n_issues_open                      |          0.1391 |     -0.0258 | -      |
| ownership_ratio_pre                |          0.1157 |      0.0129 | +      |
| code_churn_pre                     |          0.1110 |     -0.0144 | -      |
| code_removed_pre                   |          0.1095 |      0.0222 | +      |
| n_code_smell                       |          0.1047 |      0.0376 | +      |
| std_change_size_pre                |          0.1001 |     -0.0034 | -      |
| project_function_complexity        |          0.0895 |     -0.0055 | -      |

### 6.8.3 Top-15 SHAP features - szz

| feature                            |   mean_abs_shap |   mean_shap | sign   |
|:-----------------------------------|----------------:|------------:|:-------|
| project_function_complexity        |          1.5084 |      0.1891 | +      |
| days_since_last_change_at_snapshot |          1.0455 |     -0.0957 | -      |
| ownership_ratio_pre                |          0.4118 |     -0.0224 | -      |
| pseudo_ncloc_at_t                  |          0.4047 |     -0.0512 | -      |
| max_single_commit_churn_pre        |          0.3623 |     -0.0142 | -      |
| file_age_days_at_snapshot          |          0.3593 |     -0.0139 | -      |
| project_file_complexity            |          0.3352 |      0.0524 | +      |
| code_added_pre                     |          0.3276 |     -0.0265 | -      |
| avg_change_size_pre                |          0.3207 |     -0.0175 | -      |
| project_sqale_debt_ratio           |          0.2573 |      0.0119 | +      |
| code_removed_pre                   |          0.2525 |      0.0026 | +      |
| recent_churn_90d_pre               |          0.2412 |     -0.0222 | -      |
| recent_churn_30d_pre               |          0.2400 |     -0.0087 | -      |
| code_churn_pre                     |          0.2073 |     -0.0175 | -      |
| total_contributors_pre             |          0.1740 |      0.0298 | +      |
