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
| consequence |  23911 |              86 |           82 |               14.78 |        3533 |                   nan |                  0 |
| severity    |  23911 |              79 |           76 |                9.64 |        2305 |                   nan |                  0 |
| szz         |  23911 |              83 |           80 |                1.24 |         297 |                   nan |                  0 |

## 6.4 Within-project 10-fold cross-validation

Stratified 10-fold cross-validation on the combined (22-project, 23,911-row) dataset. Mean metrics across folds:

| variant     | model               |   precision_mean |   recall_mean |   f1_mean |   roc_auc_mean |   pr_auc_mean |   mcc_mean |   ce_at_20_mean |
|:------------|:--------------------|-----------------:|--------------:|----------:|---------------:|--------------:|-----------:|----------------:|
| consequence | decision_tree       |            0.508 |         0.495 |     0.501 |          0.706 |         0.327 |      0.416 |           0.521 |
| consequence | lightgbm            |            0.509 |         0.749 |     0.606 |          0.909 |         0.686 |      0.536 |           0.724 |
| consequence | logistic_regression |            0.363 |         0.739 |     0.487 |          0.835 |         0.517 |      0.398 |           0.594 |
| consequence | random_forest       |            0.784 |         0.429 |     0.554 |          0.915 |         0.703 |      0.531 |           0.746 |
| consequence | svm                 |            0.633 |         0.309 |     0.415 |          0.841 |         0.514 |      0.381 |           0.602 |
| consequence | xgboost             |            0.742 |         0.467 |     0.573 |          0.911 |         0.695 |      0.536 |           0.736 |
| severity    | decision_tree       |            0.656 |         0.627 |     0.641 |          0.796 |         0.448 |      0.604 |           0.674 |
| severity    | lightgbm            |            0.620 |         0.849 |     0.716 |          0.976 |         0.820 |      0.691 |           0.963 |
| severity    | logistic_regression |            0.520 |         0.869 |     0.651 |          0.962 |         0.758 |      0.629 |           0.924 |
| severity    | random_forest       |            0.806 |         0.580 |     0.674 |          0.974 |         0.807 |      0.656 |           0.968 |
| severity    | svm                 |            0.725 |         0.515 |     0.602 |          0.959 |         0.722 |      0.577 |           0.918 |
| severity    | xgboost             |            0.785 |         0.650 |     0.711 |          0.976 |         0.820 |      0.687 |           0.968 |
| szz         | decision_tree       |            0.243 |         0.232 |     0.235 |          0.611 |         0.068 |      0.227 |           0.360 |
| szz         | lightgbm            |            0.352 |         0.306 |     0.324 |          0.936 |         0.264 |      0.319 |           0.916 |
| szz         | logistic_regression |            0.058 |         0.825 |     0.108 |          0.923 |         0.213 |      0.190 |           0.852 |
| szz         | random_forest       |            0.504 |         0.057 |     0.098 |          0.933 |         0.280 |      0.157 |           0.943 |
| szz         | svm                 |            0.350 |         0.027 |     0.050 |          0.910 |         0.184 |      0.093 |           0.859 |
| szz         | xgboost             |            0.467 |         0.120 |     0.186 |          0.946 |         0.276 |      0.228 |           0.919 |

### Within-project highlights
- **Consequence**: best model = **lightgbm**, F1=0.606, CE@20=0.724.
- **Severity**: best model = **lightgbm**, F1=0.716, CE@20=0.963.
- **SZZ**: best model = **lightgbm**, F1=0.324, CE@20=0.916.

## 6.5 Leave-One-Project-Out cross-project validation

For each variant and model we train on 21 projects and test on the held-out project, repeating for every project. The SZZ variant covers 16/22 projects because 6 projects have zero SZZ positives in their observation window.

| variant     | model               |   n_projects |   precision_mean |   recall_mean |   f1_mean |   roc_auc_mean |   pr_auc_mean |   mcc_mean |   ce_at_20_mean |
|:------------|:--------------------|-------------:|-----------------:|--------------:|----------:|---------------:|--------------:|-----------:|----------------:|
| consequence | decision_tree       |           22 |            0.266 |         0.285 |     0.266 |          0.574 |         0.215 |      0.135 |           0.287 |
| consequence | lightgbm            |           22 |            0.366 |         0.491 |     0.401 |          0.765 |         0.459 |      0.296 |           0.485 |
| consequence | logistic_regression |           22 |            0.298 |         0.665 |     0.363 |          0.759 |         0.455 |      0.224 |           0.476 |
| consequence | random_forest       |           22 |            0.558 |         0.120 |     0.190 |          0.788 |         0.486 |      0.214 |           0.502 |
| consequence | xgboost             |           22 |            0.536 |         0.266 |     0.333 |          0.782 |         0.475 |      0.283 |           0.492 |
| severity    | decision_tree       |           22 |            0.570 |         0.434 |     0.479 |          0.700 |         0.322 |      0.447 |           0.482 |
| severity    | lightgbm            |           22 |            0.584 |         0.683 |     0.602 |          0.956 |         0.702 |      0.571 |           0.871 |
| severity    | logistic_regression |           22 |            0.521 |         0.773 |     0.600 |          0.946 |         0.702 |      0.568 |           0.884 |
| severity    | random_forest       |           22 |            0.761 |         0.264 |     0.376 |          0.960 |         0.703 |      0.408 |           0.873 |
| severity    | xgboost             |           22 |            0.707 |         0.503 |     0.570 |          0.959 |         0.711 |      0.553 |           0.870 |
| szz         | decision_tree       |           16 |            0.024 |         0.042 |     0.021 |          0.500 |         0.040 |      0.007 |           0.177 |
| szz         | lightgbm            |           16 |            0.130 |         0.055 |     0.038 |          0.713 |         0.151 |      0.052 |           0.489 |
| szz         | logistic_regression |           16 |            0.126 |         0.412 |     0.107 |          0.775 |         0.184 |      0.100 |           0.554 |
| szz         | random_forest       |           16 |            0.000 |         0.000 |     0.000 |          0.760 |         0.168 |      0.000 |           0.542 |
| szz         | xgboost             |           16 |            0.062 |         0.007 |     0.013 |          0.773 |         0.137 |      0.018 |           0.528 |

### 6.5.1 Generalization gap

Difference between within-project and LOPO mean performance:

| variant     | model               |   precision_within |   precision_lopo |   precision_gap |   recall_within |   recall_lopo |   recall_gap |   f1_within |   f1_lopo |   f1_gap |   roc_auc_within |   roc_auc_lopo |   roc_auc_gap |   pr_auc_within |   pr_auc_lopo |   pr_auc_gap |   mcc_within |   mcc_lopo |   mcc_gap |   ce_at_20_within |   ce_at_20_lopo |   ce_at_20_gap |
|:------------|:--------------------|-------------------:|-----------------:|----------------:|----------------:|--------------:|-------------:|------------:|----------:|---------:|-----------------:|---------------:|--------------:|----------------:|--------------:|-------------:|-------------:|-----------:|----------:|------------------:|----------------:|---------------:|
| consequence | decision_tree       |              0.508 |            0.266 |           0.242 |           0.495 |         0.285 |        0.210 |       0.501 |     0.266 |    0.235 |            0.706 |          0.574 |         0.132 |           0.327 |         0.215 |        0.112 |        0.416 |      0.135 |     0.281 |             0.521 |           0.287 |          0.235 |
| consequence | lightgbm            |              0.509 |            0.366 |           0.143 |           0.749 |         0.491 |        0.258 |       0.606 |     0.401 |    0.204 |            0.909 |          0.765 |         0.144 |           0.686 |         0.459 |        0.227 |        0.536 |      0.296 |     0.240 |             0.724 |           0.485 |          0.239 |
| consequence | logistic_regression |              0.363 |            0.298 |           0.065 |           0.739 |         0.665 |        0.075 |       0.487 |     0.363 |    0.124 |            0.835 |          0.759 |         0.076 |           0.517 |         0.455 |        0.062 |        0.398 |      0.224 |     0.174 |             0.594 |           0.476 |          0.118 |
| consequence | random_forest       |              0.784 |            0.558 |           0.226 |           0.429 |         0.120 |        0.308 |       0.554 |     0.190 |    0.363 |            0.915 |          0.788 |         0.127 |           0.703 |         0.486 |        0.217 |        0.531 |      0.214 |     0.318 |             0.746 |           0.502 |          0.243 |
| consequence | svm                 |              0.633 |          nan     |         nan     |           0.309 |       nan     |      nan     |       0.415 |   nan     |  nan     |            0.841 |        nan     |       nan     |           0.514 |       nan     |      nan     |        0.381 |    nan     |   nan     |             0.602 |         nan     |        nan     |
| consequence | xgboost             |              0.742 |            0.536 |           0.207 |           0.467 |         0.266 |        0.201 |       0.573 |     0.333 |    0.240 |            0.911 |          0.782 |         0.129 |           0.695 |         0.475 |        0.220 |        0.536 |      0.283 |     0.254 |             0.736 |           0.492 |          0.243 |
| severity    | decision_tree       |              0.656 |            0.570 |           0.086 |           0.627 |         0.434 |        0.194 |       0.641 |     0.479 |    0.162 |            0.796 |          0.700 |         0.096 |           0.448 |         0.322 |        0.126 |        0.604 |      0.447 |     0.157 |             0.674 |           0.482 |          0.192 |
| severity    | lightgbm            |              0.620 |            0.584 |           0.036 |           0.849 |         0.683 |        0.166 |       0.716 |     0.602 |    0.114 |            0.976 |          0.956 |         0.020 |           0.820 |         0.702 |        0.117 |        0.691 |      0.571 |     0.120 |             0.963 |           0.871 |          0.092 |
| severity    | logistic_regression |              0.520 |            0.521 |          -0.000 |           0.869 |         0.773 |        0.096 |       0.651 |     0.600 |    0.051 |            0.962 |          0.946 |         0.016 |           0.758 |         0.702 |        0.056 |        0.629 |      0.568 |     0.061 |             0.924 |           0.884 |          0.040 |
| severity    | random_forest       |              0.806 |            0.761 |           0.046 |           0.580 |         0.264 |        0.316 |       0.674 |     0.376 |    0.298 |            0.974 |          0.960 |         0.014 |           0.807 |         0.703 |        0.104 |        0.656 |      0.408 |     0.249 |             0.968 |           0.873 |          0.095 |
| severity    | svm                 |              0.725 |          nan     |         nan     |           0.515 |       nan     |      nan     |       0.602 |   nan     |  nan     |            0.959 |        nan     |       nan     |           0.722 |       nan     |      nan     |        0.577 |    nan     |   nan     |             0.918 |         nan     |        nan     |
| severity    | xgboost             |              0.785 |            0.707 |           0.078 |           0.650 |         0.503 |        0.147 |       0.711 |     0.570 |    0.141 |            0.976 |          0.959 |         0.018 |           0.820 |         0.711 |        0.109 |        0.687 |      0.553 |     0.134 |             0.968 |           0.870 |          0.098 |
| szz         | decision_tree       |              0.243 |            0.024 |           0.218 |           0.232 |         0.042 |        0.189 |       0.235 |     0.021 |    0.214 |            0.611 |          0.500 |         0.111 |           0.068 |         0.040 |        0.028 |        0.227 |      0.007 |     0.220 |             0.360 |           0.177 |          0.183 |
| szz         | lightgbm            |              0.352 |            0.130 |           0.223 |           0.306 |         0.055 |        0.251 |       0.324 |     0.038 |    0.285 |            0.936 |          0.713 |         0.222 |           0.264 |         0.151 |        0.113 |        0.319 |      0.052 |     0.266 |             0.916 |           0.489 |          0.427 |
| szz         | logistic_regression |              0.058 |            0.126 |          -0.068 |           0.825 |         0.412 |        0.414 |       0.108 |     0.107 |    0.001 |            0.923 |          0.775 |         0.149 |           0.213 |         0.184 |        0.029 |        0.190 |      0.100 |     0.090 |             0.852 |           0.554 |          0.298 |
| szz         | random_forest       |              0.504 |            0.000 |           0.504 |           0.057 |         0.000 |        0.057 |       0.098 |     0.000 |    0.098 |            0.933 |          0.760 |         0.173 |           0.280 |         0.168 |        0.112 |        0.157 |      0.000 |     0.157 |             0.943 |           0.542 |          0.401 |
| szz         | svm                 |              0.350 |          nan     |         nan     |           0.027 |       nan     |      nan     |       0.050 |   nan     |  nan     |            0.910 |        nan     |       nan     |           0.184 |       nan     |      nan     |        0.093 |    nan     |   nan     |             0.859 |         nan     |        nan     |
| szz         | xgboost             |              0.467 |            0.062 |           0.405 |           0.120 |         0.007 |        0.114 |       0.186 |     0.012 |    0.174 |            0.946 |          0.773 |         0.173 |           0.276 |         0.137 |        0.139 |        0.228 |      0.018 |     0.209 |             0.919 |           0.528 |          0.391 |

- **Consequence**: best LOPO model = **lightgbm**, F1=0.401, CE@20=0.485.
- **Severity** (best LOPO): **lightgbm**, F1=0.602, CE@20=0.871.
- **SZZ** (best LOPO): **logistic_regression**, F1=0.107, CE@20=0.554.

See Figures ``fig_within_vs_lopo`` and ``fig_lopo_per_project``.

## 6.6 Sensitivity to labeling parameters

The consequence-variant default is (window=6 months, percentile=top 20%). The grid below shows LightGBM 10-fold CV performance across a 3x3 parameter sweep:

|   window_months |   percentile |   positive_rate_pct |   f1_mean |   roc_auc_mean |   pr_auc_mean |   ce_at_20_mean |
|----------------:|-------------:|--------------------:|----------:|---------------:|--------------:|----------------:|
|           3.000 |       10.000 |               7.710 |     0.506 |          0.906 |         0.555 |           0.785 |
|           3.000 |       20.000 |              10.760 |     0.589 |          0.922 |         0.677 |           0.794 |
|           3.000 |       30.000 |              11.760 |     0.612 |          0.925 |         0.703 |           0.794 |
|           6.000 |       10.000 |               9.650 |     0.512 |          0.900 |         0.562 |           0.758 |
|           6.000 |       20.000 |              14.780 |     0.606 |          0.909 |         0.686 |           0.724 |
|           6.000 |       30.000 |              19.390 |     0.680 |          0.922 |         0.777 |           0.705 |
|          12.000 |       10.000 |              10.000 |     0.529 |          0.895 |         0.580 |           0.766 |
|          12.000 |       20.000 |              18.970 |     0.662 |          0.907 |         0.735 |           0.685 |
|          12.000 |       30.000 |              25.150 |     0.727 |          0.917 |         0.813 |           0.632 |

See Figure ``fig_sensitivity_heatmap``.

## 6.7 Feature-group ablation

Three feature groups are defined: ``static_sonar`` (per-basename SonarQube aggregates at ``t``), ``historical`` (pre-``t`` Git commit process metrics), and ``project_context`` (project-level SonarQube measures at the most recent analysis <= ``t``).

| variant     | group           | mode                 |   n_features |   precision_mean |   recall_mean |   f1_mean |   roc_auc_mean |   pr_auc_mean |   mcc_mean |   ce_at_20_mean |
|:------------|:----------------|:---------------------|-------------:|-----------------:|--------------:|----------:|---------------:|--------------:|-----------:|----------------:|
| consequence | all             | all_features         |           81 |            0.509 |         0.749 |     0.606 |          0.909 |         0.686 |      0.536 |           0.724 |
| consequence | static_sonar    | only_this_group      |           16 |            0.329 |         0.661 |     0.439 |          0.792 |         0.450 |      0.332 |           0.527 |
| consequence | static_sonar    | leave_out_this_group |           65 |            0.500 |         0.750 |     0.599 |          0.904 |         0.672 |      0.529 |           0.720 |
| consequence | historical      | only_this_group      |           15 |            0.443 |         0.744 |     0.555 |          0.881 |         0.610 |      0.478 |           0.670 |
| consequence | historical      | leave_out_this_group |           66 |            0.496 |         0.754 |     0.598 |          0.899 |         0.670 |      0.528 |           0.718 |
| consequence | project_context | only_this_group      |           30 |            0.195 |         0.665 |     0.301 |          0.590 |         0.174 |      0.133 |           0.229 |
| consequence | project_context | leave_out_this_group |           51 |            0.508 |         0.754 |     0.607 |          0.907 |         0.675 |      0.538 |           0.720 |
| consequence | cocg            | only_this_group      |           10 |            0.414 |         0.716 |     0.525 |          0.859 |         0.581 |      0.440 |           0.629 |
| consequence | cocg            | leave_out_this_group |           71 |            0.493 |         0.744 |     0.593 |          0.899 |         0.658 |      0.521 |           0.712 |
| consequence | prior_defect    | only_this_group      |           10 |            0.396 |         0.590 |     0.474 |          0.776 |         0.477 |      0.372 |           0.562 |
| consequence | prior_defect    | leave_out_this_group |           71 |            0.506 |         0.754 |     0.606 |          0.907 |         0.676 |      0.537 |           0.722 |
| severity    | all             | all_features         |           75 |            0.620 |         0.849 |     0.716 |          0.976 |         0.820 |      0.691 |           0.963 |
| severity    | static_sonar    | only_this_group      |           10 |            0.510 |         0.879 |     0.646 |          0.969 |         0.781 |      0.626 |           0.941 |
| severity    | static_sonar    | leave_out_this_group |           65 |            0.388 |         0.687 |     0.496 |          0.896 |         0.547 |      0.448 |           0.736 |
| severity    | historical      | only_this_group      |           15 |            0.345 |         0.693 |     0.461 |          0.876 |         0.503 |      0.413 |           0.701 |
| severity    | historical      | leave_out_this_group |           60 |            0.605 |         0.849 |     0.706 |          0.975 |         0.818 |      0.681 |           0.959 |
| severity    | project_context | only_this_group      |           30 |            0.137 |         0.708 |     0.229 |          0.634 |         0.141 |      0.136 |           0.313 |
| severity    | project_context | leave_out_this_group |           45 |            0.617 |         0.838 |     0.710 |          0.974 |         0.814 |      0.684 |           0.955 |
| severity    | cocg            | only_this_group      |           10 |            0.285 |         0.652 |     0.396 |          0.827 |         0.435 |      0.339 |           0.614 |
| severity    | cocg            | leave_out_this_group |           65 |            0.608 |         0.849 |     0.708 |          0.975 |         0.815 |      0.683 |           0.963 |
| severity    | prior_defect    | only_this_group      |           10 |            0.254 |         0.520 |     0.341 |          0.730 |         0.338 |      0.264 |           0.525 |
| severity    | prior_defect    | leave_out_this_group |           65 |            0.621 |         0.843 |     0.715 |          0.975 |         0.819 |      0.689 |           0.962 |
| szz         | all             | all_features         |           79 |            0.352 |         0.306 |     0.324 |          0.935 |         0.264 |      0.319 |           0.916 |
| szz         | static_sonar    | only_this_group      |           16 |            0.027 |         0.225 |     0.048 |          0.634 |         0.052 |      0.045 |           0.495 |
| szz         | static_sonar    | leave_out_this_group |           63 |            0.279 |         0.266 |     0.269 |          0.934 |         0.241 |      0.262 |           0.912 |
| szz         | historical      | only_this_group      |           15 |            0.223 |         0.232 |     0.226 |          0.875 |         0.162 |      0.217 |           0.788 |
| szz         | historical      | leave_out_this_group |           64 |            0.272 |         0.276 |     0.272 |          0.933 |         0.248 |      0.264 |           0.912 |
| szz         | project_context | only_this_group      |           30 |            0.037 |         0.899 |     0.070 |          0.859 |         0.058 |      0.145 |           0.707 |
| szz         | project_context | leave_out_this_group |           49 |            0.332 |         0.265 |     0.292 |          0.921 |         0.234 |      0.288 |           0.882 |
| szz         | cocg            | only_this_group      |           10 |            0.207 |         0.340 |     0.256 |          0.887 |         0.186 |      0.253 |           0.811 |
| szz         | cocg            | leave_out_this_group |           69 |            0.292 |         0.279 |     0.283 |          0.930 |         0.234 |      0.276 |           0.896 |
| szz         | prior_defect    | only_this_group      |            8 |            0.040 |         0.367 |     0.071 |          0.622 |         0.075 |      0.087 |           0.455 |
| szz         | prior_defect    | leave_out_this_group |           71 |            0.329 |         0.279 |     0.300 |          0.931 |         0.259 |      0.294 |           0.899 |

See Figure ``fig_feature_ablation``.

## 6.8 Feature importance (SHAP + permutation)

For each variant we train a single LightGBM classifier on an 80/20 stratified split and compute TreeSHAP values on the test set. The permutation importance is a secondary check (ROC-AUC drop when the column is shuffled). Tables per variant: ``shap_top15_{variant}.csv`` and ``perm_top15_{variant}.csv`` in ``results/tables``. Figures: ``fig_shap_{variant}``.

### 6.8.1 Top-15 SHAP features - consequence

| feature                            |   mean_abs_shap |   mean_shap | sign   |
|:-----------------------------------|----------------:|------------:|:-------|
| pseudo_ncloc_at_t                  |          0.8563 |     -0.1689 | -      |
| days_since_last_change_at_snapshot |          0.8142 |      0.0858 | +      |
| cocg_closeness                     |          0.3041 |      0.0089 | +      |
| file_age_days_at_snapshot          |          0.2536 |      0.0057 | +      |
| cocg_strength_mean                 |          0.1898 |     -0.0170 | -      |
| cocg_strength_sum                  |          0.1524 |     -0.0195 | -      |
| n_distinct_rules                   |          0.1393 |      0.0094 | +      |
| cocg_degree                        |          0.1187 |      0.0136 | +      |
| time_since_last_bugfix_days        |          0.1173 |      0.0116 | +      |
| avg_change_size_pre                |          0.1080 |      0.0003 | +      |
| szz_inducing_pre                   |          0.1007 |     -0.0118 | -      |
| project_comment_lines_density      |          0.1003 |     -0.0246 | -      |
| total_commits_pre                  |          0.0949 |      0.0089 | +      |
| code_added_pre                     |          0.0887 |     -0.0176 | -      |
| cocg_pagerank                      |          0.0830 |     -0.0003 | -      |

### 6.8.2 Top-15 SHAP features - severity

| feature                            |   mean_abs_shap |   mean_shap | sign   |
|:-----------------------------------|----------------:|------------:|:-------|
| total_debt_minutes                 |          3.2441 |      0.7421 | +      |
| n_distinct_rules                   |          0.3636 |      0.1226 | +      |
| max_single_commit_churn_pre        |          0.2747 |      0.0081 | +      |
| n_bug                              |          0.2572 |      0.1499 | +      |
| avg_change_size_pre                |          0.1226 |     -0.0138 | -      |
| time_since_last_bugfix_days        |          0.1152 |      0.0068 | +      |
| total_contributors_pre             |          0.1125 |      0.0138 | +      |
| file_age_days_at_snapshot          |          0.1039 |     -0.0156 | -      |
| project_function_complexity        |          0.1034 |      0.0033 | +      |
| days_since_last_change_at_snapshot |          0.0988 |     -0.0134 | -      |
| code_added_pre                     |          0.0972 |     -0.0042 | -      |
| ownership_ratio_pre                |          0.0950 |      0.0054 | +      |
| std_change_size_pre                |          0.0863 |     -0.0054 | -      |
| cocg_betweenness                   |          0.0849 |     -0.0034 | -      |
| cocg_pagerank                      |          0.0826 |      0.0042 | +      |

### 6.8.3 Top-15 SHAP features - szz

| feature                            |   mean_abs_shap |   mean_shap | sign   |
|:-----------------------------------|----------------:|------------:|:-------|
| project_function_complexity        |          1.2632 |      0.1845 | +      |
| days_since_last_change_at_snapshot |          0.8482 |     -0.0219 | -      |
| pseudo_ncloc_at_t                  |          0.4569 |     -0.0721 | -      |
| cocg_strength_sum                  |          0.3765 |      0.0047 | +      |
| file_age_days_at_snapshot          |          0.3427 |     -0.0434 | -      |
| cocg_pagerank                      |          0.3263 |     -0.0478 | -      |
| project_file_complexity            |          0.3261 |      0.0608 | +      |
| cocg_strength_mean                 |          0.3248 |     -0.0876 | -      |
| code_added_pre                     |          0.3172 |     -0.0637 | -      |
| ownership_ratio_pre                |          0.2819 |      0.0225 | +      |
| max_single_commit_churn_pre        |          0.2733 |      0.0107 | +      |
| cocg_degree                        |          0.2724 |      0.0335 | +      |
| avg_change_size_pre                |          0.2384 |     -0.0045 | -      |
| code_removed_pre                   |          0.2133 |      0.0160 | +      |
| code_churn_pre                     |          0.2109 |     -0.0557 | -      |

## 6.9 Hyperparameter tuning (Stage 7b)

Per-(variant, model) Optuna random-TPE search optimising mean PR-AUC on an inner stratified K-fold split (`TUNING_TRIALS` trials). The selected configuration is then re-evaluated on the canonical outer 10-fold CV. SVM is excluded from tuning (see Research Log 2026-04-27 / refinement 3).

| variant     | model               |   precision_mean |   recall_mean |   f1_mean |   roc_auc_mean |   pr_auc_mean |   mcc_mean |   ce_at_20_mean |
|:------------|:--------------------|-----------------:|--------------:|----------:|---------------:|--------------:|-----------:|----------------:|
| consequence | decision_tree       |            0.390 |         0.761 |     0.515 |          0.849 |         0.556 |      0.434 |           0.636 |
| consequence | lightgbm            |            0.628 |         0.664 |     0.645 |          0.916 |         0.712 |      0.582 |           0.746 |
| consequence | logistic_regression |            0.365 |         0.739 |     0.488 |          0.835 |         0.517 |      0.399 |           0.595 |
| consequence | random_forest       |            0.742 |         0.476 |     0.580 |          0.916 |         0.700 |      0.542 |           0.745 |
| consequence | xgboost             |            0.752 |         0.506 |     0.604 |          0.918 |         0.716 |      0.566 |           0.746 |
| severity    | decision_tree       |            0.447 |         0.955 |     0.609 |          0.964 |         0.765 |      0.605 |           0.946 |
| severity    | lightgbm            |            0.619 |         0.862 |     0.721 |          0.977 |         0.831 |      0.698 |           0.967 |
| severity    | logistic_regression |            0.520 |         0.866 |     0.650 |          0.962 |         0.759 |      0.628 |           0.924 |
| severity    | random_forest       |            0.785 |         0.629 |     0.697 |          0.975 |         0.811 |      0.674 |           0.968 |
| severity    | xgboost             |            0.796 |         0.645 |     0.712 |          0.977 |         0.828 |      0.690 |           0.973 |
| szz         | decision_tree       |            0.081 |         0.636 |     0.144 |          0.784 |         0.146 |      0.203 |           0.683 |
| szz         | lightgbm            |            0.255 |         0.411 |     0.313 |          0.947 |         0.298 |      0.312 |           0.933 |
| szz         | logistic_regression |            0.060 |         0.822 |     0.111 |          0.926 |         0.229 |      0.193 |           0.866 |
| szz         | random_forest       |            0.506 |         0.121 |     0.188 |          0.944 |         0.273 |      0.236 |           0.956 |
| szz         | xgboost             |            0.385 |         0.070 |     0.118 |          0.951 |         0.272 |      0.159 |           0.953 |

Selected hyperparameters per (variant, model) are persisted in ``tuned_params.json``. The tuner respects the proposal's PR-AUC objective for imbalanced classes (Saito and Rehmsmeier 2015).

## 6.10 Confidence intervals and pairwise significance

Bootstrap 95% CIs (10000 resamples, percentile method) are computed on per-fold metrics for the within-project summary. Pairwise Wilcoxon signed-rank tests are run on per-fold F1 and PR-AUC; p-values are Bonferroni-corrected per variant family (15 pairs for 6 within-project models, 10 for the 5-model LOPO scope).

| variant     | model               |   f1_mean |   f1_ci_low |   f1_ci_high |   pr_auc_mean |   pr_auc_ci_low |   pr_auc_ci_high |   ce_at_20_mean |   ce_at_20_ci_low |   ce_at_20_ci_high |
|:------------|:--------------------|----------:|------------:|-------------:|--------------:|----------------:|-----------------:|----------------:|------------------:|-------------------:|
| consequence | decision_tree       |     0.501 |       0.489 |        0.515 |         0.327 |           0.317 |            0.338 |           0.521 |             0.505 |              0.540 |
| consequence | lightgbm            |     0.606 |       0.597 |        0.614 |         0.686 |           0.674 |            0.697 |           0.724 |             0.712 |              0.734 |
| consequence | logistic_regression |     0.487 |       0.481 |        0.493 |         0.517 |           0.501 |            0.532 |           0.594 |             0.580 |              0.609 |
| consequence | random_forest       |     0.554 |       0.537 |        0.568 |         0.703 |           0.692 |            0.714 |           0.746 |             0.733 |              0.757 |
| consequence | svm                 |     0.415 |       0.399 |        0.431 |         0.514 |           0.502 |            0.525 |           0.602 |             0.588 |              0.617 |
| consequence | xgboost             |     0.573 |       0.562 |        0.584 |         0.695 |           0.685 |            0.705 |           0.736 |             0.725 |              0.744 |
| severity    | decision_tree       |     0.641 |       0.623 |        0.659 |         0.448 |           0.427 |            0.471 |           0.674 |             0.653 |              0.693 |
| severity    | lightgbm            |     0.716 |       0.703 |        0.726 |         0.820 |           0.808 |            0.828 |           0.963 |             0.954 |              0.972 |
| severity    | logistic_regression |     0.651 |       0.638 |        0.663 |         0.758 |           0.747 |            0.767 |           0.924 |             0.911 |              0.936 |
| severity    | random_forest       |     0.674 |       0.666 |        0.682 |         0.807 |           0.793 |            0.817 |           0.968 |             0.955 |              0.978 |
| severity    | svm                 |     0.602 |       0.586 |        0.618 |         0.722 |           0.710 |            0.733 |           0.918 |             0.905 |              0.929 |
| severity    | xgboost             |     0.711 |       0.697 |        0.725 |         0.820 |           0.807 |            0.830 |           0.968 |             0.958 |              0.977 |
| szz         | decision_tree       |     0.235 |       0.207 |        0.265 |         0.068 |           0.055 |            0.081 |           0.360 |             0.326 |              0.400 |
| szz         | lightgbm            |     0.324 |       0.269 |        0.376 |         0.264 |           0.221 |            0.303 |           0.916 |             0.879 |              0.952 |
| szz         | logistic_regression |     0.108 |       0.103 |        0.112 |         0.213 |           0.188 |            0.240 |           0.852 |             0.809 |              0.893 |
| szz         | random_forest       |     0.098 |       0.056 |        0.148 |         0.280 |           0.234 |            0.324 |           0.943 |             0.916 |              0.970 |
| szz         | svm                 |     0.050 |       0.019 |        0.081 |         0.184 |           0.154 |            0.216 |           0.859 |             0.829 |              0.886 |
| szz         | xgboost             |     0.186 |       0.120 |        0.251 |         0.276 |           0.226 |            0.323 |           0.919 |             0.887 |              0.950 |

### 6.10.1 Pairwise Wilcoxon significance (within-project)

| variant     | metric   | model_a             | model_b             |   n_pairs |   mean_a |   mean_b |   mean_diff |   p_value |   p_value_bonferroni |
|:------------|:---------|:--------------------|:--------------------|----------:|---------:|---------:|------------:|----------:|---------------------:|
| consequence | f1       | decision_tree       | lightgbm            |        10 |   0.5011 |   0.6056 |     -0.1046 |    0.0020 |               0.0293 |
| consequence | f1       | decision_tree       | logistic_regression |        10 |   0.5011 |   0.4872 |      0.0139 |    0.1602 |               1.0000 |
| consequence | f1       | decision_tree       | random_forest       |        10 |   0.5011 |   0.5539 |     -0.0528 |    0.0020 |               0.0293 |
| consequence | f1       | decision_tree       | svm                 |        10 |   0.5011 |   0.4152 |      0.0858 |    0.0020 |               0.0293 |
| consequence | f1       | decision_tree       | xgboost             |        10 |   0.5011 |   0.5731 |     -0.0720 |    0.0020 |               0.0293 |
| consequence | f1       | lightgbm            | logistic_regression |        10 |   0.6056 |   0.4872 |      0.1185 |    0.0020 |               0.0293 |
| consequence | f1       | lightgbm            | random_forest       |        10 |   0.6056 |   0.5539 |      0.0517 |    0.0020 |               0.0293 |
| consequence | f1       | lightgbm            | svm                 |        10 |   0.6056 |   0.4152 |      0.1904 |    0.0020 |               0.0293 |
| consequence | f1       | lightgbm            | xgboost             |        10 |   0.6056 |   0.5731 |      0.0325 |    0.0020 |               0.0293 |
| consequence | f1       | logistic_regression | random_forest       |        10 |   0.4872 |   0.5539 |     -0.0667 |    0.0020 |               0.0293 |
| consequence | f1       | logistic_regression | svm                 |        10 |   0.4872 |   0.4152 |      0.0719 |    0.0020 |               0.0293 |
| consequence | f1       | logistic_regression | xgboost             |        10 |   0.4872 |   0.5731 |     -0.0859 |    0.0020 |               0.0293 |
| consequence | f1       | random_forest       | svm                 |        10 |   0.5539 |   0.4152 |      0.1387 |    0.0020 |               0.0293 |
| consequence | f1       | random_forest       | xgboost             |        10 |   0.5539 |   0.5731 |     -0.0192 |    0.0137 |               0.2051 |
| consequence | f1       | svm                 | xgboost             |        10 |   0.4152 |   0.5731 |     -0.1579 |    0.0020 |               0.0293 |
| severity    | f1       | decision_tree       | lightgbm            |        10 |   0.6411 |   0.7162 |     -0.0751 |    0.0020 |               0.0293 |
| severity    | f1       | decision_tree       | logistic_regression |        10 |   0.6411 |   0.6508 |     -0.0097 |    0.1934 |               1.0000 |
| severity    | f1       | decision_tree       | random_forest       |        10 |   0.6411 |   0.6743 |     -0.0331 |    0.0020 |               0.0293 |
| severity    | f1       | decision_tree       | svm                 |        10 |   0.6411 |   0.6021 |      0.0390 |    0.0020 |               0.0293 |
| severity    | f1       | decision_tree       | xgboost             |        10 |   0.6411 |   0.7107 |     -0.0696 |    0.0020 |               0.0293 |
| severity    | f1       | lightgbm            | logistic_regression |        10 |   0.7162 |   0.6508 |      0.0654 |    0.0020 |               0.0293 |
| severity    | f1       | lightgbm            | random_forest       |        10 |   0.7162 |   0.6743 |      0.0420 |    0.0020 |               0.0293 |
| severity    | f1       | lightgbm            | svm                 |        10 |   0.7162 |   0.6021 |      0.1141 |    0.0020 |               0.0293 |
| severity    | f1       | lightgbm            | xgboost             |        10 |   0.7162 |   0.7107 |      0.0055 |    0.1934 |               1.0000 |
| severity    | f1       | logistic_regression | random_forest       |        10 |   0.6508 |   0.6743 |     -0.0234 |    0.0020 |               0.0293 |
| severity    | f1       | logistic_regression | svm                 |        10 |   0.6508 |   0.6021 |      0.0488 |    0.0059 |               0.0879 |
| severity    | f1       | logistic_regression | xgboost             |        10 |   0.6508 |   0.7107 |     -0.0599 |    0.0020 |               0.0293 |
| severity    | f1       | random_forest       | svm                 |        10 |   0.6743 |   0.6021 |      0.0722 |    0.0020 |               0.0293 |
| severity    | f1       | random_forest       | xgboost             |        10 |   0.6743 |   0.7107 |     -0.0365 |    0.0020 |               0.0293 |
| severity    | f1       | svm                 | xgboost             |        10 |   0.6021 |   0.7107 |     -0.1087 |    0.0020 |               0.0293 |
| szz         | f1       | decision_tree       | lightgbm            |        10 |   0.2352 |   0.3238 |     -0.0886 |    0.0273 |               0.4102 |
| szz         | f1       | decision_tree       | logistic_regression |        10 |   0.2352 |   0.1081 |      0.1271 |    0.0020 |               0.0293 |
| szz         | f1       | decision_tree       | random_forest       |        10 |   0.2352 |   0.0982 |      0.1370 |    0.0039 |               0.0586 |
| szz         | f1       | decision_tree       | svm                 |        10 |   0.2352 |   0.0496 |      0.1856 |    0.0020 |               0.0293 |
| szz         | f1       | decision_tree       | xgboost             |        10 |   0.2352 |   0.1864 |      0.0488 |    0.1309 |               1.0000 |
| szz         | f1       | lightgbm            | logistic_regression |        10 |   0.3238 |   0.1081 |      0.2157 |    0.0020 |               0.0293 |
| szz         | f1       | lightgbm            | random_forest       |        10 |   0.3238 |   0.0982 |      0.2256 |    0.0020 |               0.0293 |
| szz         | f1       | lightgbm            | svm                 |        10 |   0.3238 |   0.0496 |      0.2741 |    0.0020 |               0.0293 |
| szz         | f1       | lightgbm            | xgboost             |        10 |   0.3238 |   0.1864 |      0.1374 |    0.0039 |               0.0586 |
| szz         | f1       | logistic_regression | random_forest       |        10 |   0.1081 |   0.0982 |      0.0099 |    0.4922 |               1.0000 |
| szz         | f1       | logistic_regression | svm                 |        10 |   0.1081 |   0.0496 |      0.0584 |    0.0273 |               0.4102 |
| szz         | f1       | logistic_regression | xgboost             |        10 |   0.1081 |   0.1864 |     -0.0783 |    0.0645 |               0.9668 |
| szz         | f1       | random_forest       | svm                 |        10 |   0.0982 |   0.0496 |      0.0486 |    0.0938 |               1.0000 |
| szz         | f1       | random_forest       | xgboost             |        10 |   0.0982 |   0.1864 |     -0.0882 |    0.0078 |               0.1172 |
| szz         | f1       | svm                 | xgboost             |        10 |   0.0496 |   0.1864 |     -0.1367 |    0.0156 |               0.2344 |
| consequence | pr_auc   | decision_tree       | lightgbm            |        10 |   0.3269 |   0.6857 |     -0.3588 |    0.0020 |               0.0293 |
| consequence | pr_auc   | decision_tree       | logistic_regression |        10 |   0.3269 |   0.5166 |     -0.1897 |    0.0020 |               0.0293 |
| consequence | pr_auc   | decision_tree       | random_forest       |        10 |   0.3269 |   0.7032 |     -0.3763 |    0.0020 |               0.0293 |
| consequence | pr_auc   | decision_tree       | svm                 |        10 |   0.3269 |   0.5135 |     -0.1866 |    0.0020 |               0.0293 |
| consequence | pr_auc   | decision_tree       | xgboost             |        10 |   0.3269 |   0.6952 |     -0.3683 |    0.0020 |               0.0293 |
| consequence | pr_auc   | lightgbm            | logistic_regression |        10 |   0.6857 |   0.5166 |      0.1691 |    0.0020 |               0.0293 |
| consequence | pr_auc   | lightgbm            | random_forest       |        10 |   0.6857 |   0.7032 |     -0.0175 |    0.0020 |               0.0293 |
| consequence | pr_auc   | lightgbm            | svm                 |        10 |   0.6857 |   0.5135 |      0.1722 |    0.0020 |               0.0293 |
| consequence | pr_auc   | lightgbm            | xgboost             |        10 |   0.6857 |   0.6952 |     -0.0095 |    0.0195 |               0.2930 |
| consequence | pr_auc   | logistic_regression | random_forest       |        10 |   0.5166 |   0.7032 |     -0.1866 |    0.0020 |               0.0293 |
| consequence | pr_auc   | logistic_regression | svm                 |        10 |   0.5166 |   0.5135 |      0.0031 |    0.6250 |               1.0000 |
| consequence | pr_auc   | logistic_regression | xgboost             |        10 |   0.5166 |   0.6952 |     -0.1786 |    0.0020 |               0.0293 |
| consequence | pr_auc   | random_forest       | svm                 |        10 |   0.7032 |   0.5135 |      0.1897 |    0.0020 |               0.0293 |
| consequence | pr_auc   | random_forest       | xgboost             |        10 |   0.7032 |   0.6952 |      0.0080 |    0.1055 |               1.0000 |
| consequence | pr_auc   | svm                 | xgboost             |        10 |   0.5135 |   0.6952 |     -0.1817 |    0.0020 |               0.0293 |
| severity    | pr_auc   | decision_tree       | lightgbm            |        10 |   0.4482 |   0.8196 |     -0.3714 |    0.0020 |               0.0293 |
| severity    | pr_auc   | decision_tree       | logistic_regression |        10 |   0.4482 |   0.7575 |     -0.3093 |    0.0020 |               0.0293 |
| severity    | pr_auc   | decision_tree       | random_forest       |        10 |   0.4482 |   0.8067 |     -0.3584 |    0.0020 |               0.0293 |
| severity    | pr_auc   | decision_tree       | svm                 |        10 |   0.4482 |   0.7217 |     -0.2734 |    0.0020 |               0.0293 |
| severity    | pr_auc   | decision_tree       | xgboost             |        10 |   0.4482 |   0.8202 |     -0.3720 |    0.0020 |               0.0293 |
| severity    | pr_auc   | lightgbm            | logistic_regression |        10 |   0.8196 |   0.7575 |      0.0621 |    0.0020 |               0.0293 |
| severity    | pr_auc   | lightgbm            | random_forest       |        10 |   0.8196 |   0.8067 |      0.0130 |    0.0020 |               0.0293 |
| severity    | pr_auc   | lightgbm            | svm                 |        10 |   0.8196 |   0.7217 |      0.0980 |    0.0020 |               0.0293 |
| severity    | pr_auc   | lightgbm            | xgboost             |        10 |   0.8196 |   0.8202 |     -0.0006 |    0.7695 |               1.0000 |
| severity    | pr_auc   | logistic_regression | random_forest       |        10 |   0.7575 |   0.8067 |     -0.0491 |    0.0020 |               0.0293 |
| severity    | pr_auc   | logistic_regression | svm                 |        10 |   0.7575 |   0.7217 |      0.0359 |    0.0020 |               0.0293 |
| severity    | pr_auc   | logistic_regression | xgboost             |        10 |   0.7575 |   0.8202 |     -0.0627 |    0.0020 |               0.0293 |
| severity    | pr_auc   | random_forest       | svm                 |        10 |   0.8067 |   0.7217 |      0.0850 |    0.0020 |               0.0293 |
| severity    | pr_auc   | random_forest       | xgboost             |        10 |   0.8067 |   0.8202 |     -0.0135 |    0.0020 |               0.0293 |
| severity    | pr_auc   | svm                 | xgboost             |        10 |   0.7217 |   0.8202 |     -0.0985 |    0.0020 |               0.0293 |
| szz         | pr_auc   | decision_tree       | lightgbm            |        10 |   0.0675 |   0.2640 |     -0.1964 |    0.0020 |               0.0293 |
| szz         | pr_auc   | decision_tree       | logistic_regression |        10 |   0.0675 |   0.2134 |     -0.1458 |    0.0020 |               0.0293 |
| szz         | pr_auc   | decision_tree       | random_forest       |        10 |   0.0675 |   0.2801 |     -0.2126 |    0.0020 |               0.0293 |
| szz         | pr_auc   | decision_tree       | svm                 |        10 |   0.0675 |   0.1844 |     -0.1168 |    0.0020 |               0.0293 |
| szz         | pr_auc   | decision_tree       | xgboost             |        10 |   0.0675 |   0.2759 |     -0.2083 |    0.0020 |               0.0293 |
| szz         | pr_auc   | lightgbm            | logistic_regression |        10 |   0.2640 |   0.2134 |      0.0506 |    0.0371 |               0.5566 |
| szz         | pr_auc   | lightgbm            | random_forest       |        10 |   0.2640 |   0.2801 |     -0.0162 |    0.3223 |               1.0000 |
| szz         | pr_auc   | lightgbm            | svm                 |        10 |   0.2640 |   0.1844 |      0.0796 |    0.0273 |               0.4102 |
| szz         | pr_auc   | lightgbm            | xgboost             |        10 |   0.2640 |   0.2759 |     -0.0119 |    0.4922 |               1.0000 |
| szz         | pr_auc   | logistic_regression | random_forest       |        10 |   0.2134 |   0.2801 |     -0.0667 |    0.0059 |               0.0879 |
| szz         | pr_auc   | logistic_regression | svm                 |        10 |   0.2134 |   0.1844 |      0.0290 |    0.0840 |               1.0000 |
| szz         | pr_auc   | logistic_regression | xgboost             |        10 |   0.2134 |   0.2759 |     -0.0625 |    0.0098 |               0.1465 |
| szz         | pr_auc   | random_forest       | svm                 |        10 |   0.2801 |   0.1844 |      0.0957 |    0.0039 |               0.0586 |
| szz         | pr_auc   | random_forest       | xgboost             |        10 |   0.2801 |   0.2759 |      0.0043 |    0.4922 |               1.0000 |
| szz         | pr_auc   | svm                 | xgboost             |        10 |   0.1844 |   0.2759 |     -0.0915 |    0.0137 |               0.2051 |

## 6.11 Probability calibration (Stage 7c)

Platt scaling and isotonic regression are wrapped via `CalibratedClassifierCV` with an inner stratified K-fold split, then evaluated on the canonical outer 10-fold CV. Lower Brier score, lower negative log-likelihood and lower Expected Calibration Error (ECE) all indicate better calibration. The uncalibrated baseline is included for reference.

| variant     | model               | method       |   pr_auc_mean |   f1_mean |   brier_mean |   nll_mean |   ece_mean |
|:------------|:--------------------|:-------------|--------------:|----------:|-------------:|-----------:|-----------:|
| consequence | lightgbm            | isotonic     |        0.6674 |    0.4595 |       0.0872 |     0.3004 |     0.0871 |
| consequence | lightgbm            | platt        |        0.6630 |    0.4988 |       0.0863 |     0.2975 |     0.0804 |
| consequence | lightgbm            | uncalibrated |        0.6857 |    0.6056 |     nan      |   nan      |   nan      |
| consequence | logistic_regression | isotonic     |        0.4990 |    0.2641 |       0.1091 |     0.3653 |     0.0938 |
| consequence | logistic_regression | platt        |        0.5102 |    0.1737 |       0.1029 |     0.3467 |     0.0676 |
| consequence | logistic_regression | uncalibrated |        0.5166 |    0.4872 |     nan      |   nan      |   nan      |
| consequence | random_forest       | isotonic     |        0.6893 |    0.4964 |       0.0805 |     0.2709 |     0.0601 |
| consequence | random_forest       | platt        |        0.6850 |    0.5150 |       0.0806 |     0.2825 |     0.0668 |
| consequence | random_forest       | uncalibrated |        0.7032 |    0.5539 |     nan      |   nan      |   nan      |
| consequence | xgboost             | isotonic     |        0.6716 |    0.4530 |       0.0863 |     0.2966 |     0.0840 |
| consequence | xgboost             | platt        |        0.6707 |    0.4832 |       0.0847 |     0.2983 |     0.0815 |
| consequence | xgboost             | uncalibrated |        0.6952 |    0.5731 |     nan      |   nan      |   nan      |
| severity    | lightgbm            | isotonic     |        0.8245 |    0.6810 |       0.0372 |     0.1205 |     0.0234 |
| severity    | lightgbm            | platt        |        0.8217 |    0.7102 |       0.0378 |     0.1336 |     0.0263 |
| severity    | lightgbm            | uncalibrated |        0.8196 |    0.7162 |     nan      |   nan      |   nan      |
| severity    | logistic_regression | isotonic     |        0.7552 |    0.6304 |       0.0426 |     0.1430 |     0.0164 |
| severity    | logistic_regression | platt        |        0.7564 |    0.5895 |       0.0448 |     0.1572 |     0.0309 |
| severity    | logistic_regression | uncalibrated |        0.7575 |    0.6508 |     nan      |   nan      |   nan      |
| severity    | random_forest       | isotonic     |        0.8012 |    0.6859 |       0.0383 |     0.1217 |     0.0163 |
| severity    | random_forest       | platt        |        0.8006 |    0.6868 |       0.0386 |     0.1335 |     0.0240 |
| severity    | random_forest       | uncalibrated |        0.8067 |    0.6743 |     nan      |   nan      |   nan      |
| severity    | xgboost             | isotonic     |        0.8244 |    0.6837 |       0.0370 |     0.1190 |     0.0225 |
| severity    | xgboost             | platt        |        0.8211 |    0.6868 |       0.0385 |     0.1413 |     0.0288 |
| severity    | xgboost             | uncalibrated |        0.8202 |    0.7107 |     nan      |   nan      |   nan      |
| szz         | lightgbm            | isotonic     |        0.2481 |    0.0000 |       0.0110 |     0.0481 |     0.0045 |
| szz         | lightgbm            | platt        |        0.2419 |    0.0000 |       0.0113 |     0.0569 |     0.0049 |
| szz         | lightgbm            | uncalibrated |        0.2640 |    0.3238 |     nan      |   nan      |   nan      |
| szz         | logistic_regression | isotonic     |        0.1931 |    0.0000 |       0.0114 |     0.0528 |     0.0103 |
| szz         | logistic_regression | platt        |        0.1971 |    0.0000 |       0.0114 |     0.0525 |     0.0044 |
| szz         | logistic_regression | uncalibrated |        0.2134 |    0.1081 |     nan      |   nan      |   nan      |
| szz         | random_forest       | isotonic     |        0.2306 |    0.0000 |       0.0110 |     0.0470 |     0.0023 |
| szz         | random_forest       | platt        |        0.2458 |    0.0125 |       0.0111 |     0.0530 |     0.0057 |
| szz         | random_forest       | uncalibrated |        0.2801 |    0.0982 |     nan      |   nan      |   nan      |
| szz         | xgboost             | isotonic     |        0.2413 |    0.0000 |       0.0109 |     0.0462 |     0.0037 |
| szz         | xgboost             | platt        |        0.2452 |    0.0000 |       0.0112 |     0.0556 |     0.0052 |
| szz         | xgboost             | uncalibrated |        0.2759 |    0.1864 |     nan      |   nan      |   nan      |

See Figure ``fig_calibration_reliability``.

## 6.12 Resampling comparison (Stage 7d)

SMOTE oversampling on the *training fold only* vs the established `class_weight='balanced'` strategy. The delta column reports (SMOTE - class-weight) on each metric; positive = SMOTE wins.

| variant     | model               |   f1_cw |   pr_auc_cw |   mcc_cw |   ce_at_20_cw |   f1_smote |   pr_auc_smote |   mcc_smote |   ce_at_20_smote |   f1_delta |   pr_auc_delta |   mcc_delta |   ce_at_20_delta |
|:------------|:--------------------|--------:|------------:|---------:|--------------:|-----------:|---------------:|------------:|-----------------:|-----------:|---------------:|------------:|-----------------:|
| consequence | decision_tree       |   0.501 |       0.327 |    0.416 |         0.521 |      0.509 |          0.326 |       0.417 |            0.582 |      0.008 |         -0.001 |       0.000 |            0.060 |
| consequence | lightgbm            |   0.606 |       0.686 |    0.536 |         0.724 |      0.610 |          0.679 |       0.546 |            0.714 |      0.005 |         -0.007 |       0.010 |           -0.010 |
| consequence | logistic_regression |   0.487 |       0.517 |    0.398 |         0.594 |      0.481 |          0.486 |       0.382 |            0.573 |     -0.006 |         -0.031 |      -0.016 |           -0.021 |
| consequence | random_forest       |   0.554 |       0.703 |    0.531 |         0.746 |      0.635 |          0.694 |       0.570 |            0.735 |      0.081 |         -0.009 |       0.039 |           -0.010 |
| consequence | svm                 |   0.415 |       0.514 |    0.381 |         0.602 |      0.485 |          0.509 |       0.397 |            0.593 |      0.070 |         -0.004 |       0.016 |           -0.009 |
| consequence | xgboost             |   0.573 |       0.695 |    0.536 |         0.736 |      0.605 |          0.675 |       0.537 |            0.710 |      0.032 |         -0.020 |       0.001 |           -0.026 |
| severity    | decision_tree       |   0.641 |       0.448 |    0.604 |         0.674 |      0.613 |          0.412 |       0.574 |            0.750 |     -0.028 |         -0.036 |      -0.030 |            0.076 |
| severity    | lightgbm            |   0.716 |       0.820 |    0.691 |         0.963 |      0.704 |          0.806 |       0.674 |            0.958 |     -0.012 |         -0.013 |      -0.017 |           -0.005 |
| severity    | logistic_regression |   0.651 |       0.758 |    0.629 |         0.924 |      0.647 |          0.746 |       0.621 |            0.908 |     -0.004 |         -0.012 |      -0.008 |           -0.016 |
| severity    | random_forest       |   0.674 |       0.807 |    0.656 |         0.968 |      0.674 |          0.758 |       0.650 |            0.956 |      0.000 |         -0.049 |      -0.006 |           -0.012 |
| severity    | svm                 |   0.602 |       0.722 |    0.577 |         0.918 |      0.616 |          0.715 |       0.595 |            0.899 |      0.014 |         -0.007 |       0.018 |           -0.018 |
| severity    | xgboost             |   0.711 |       0.820 |    0.687 |         0.968 |      0.698 |          0.802 |       0.670 |            0.957 |     -0.013 |         -0.019 |      -0.017 |           -0.011 |
| szz         | decision_tree       |   0.235 |       0.068 |    0.227 |         0.360 |      0.175 |          0.045 |       0.171 |            0.377 |     -0.061 |         -0.023 |      -0.056 |            0.017 |
| szz         | lightgbm            |   0.324 |       0.264 |    0.319 |         0.916 |      0.291 |          0.238 |       0.285 |            0.919 |     -0.033 |         -0.026 |      -0.034 |            0.003 |
| szz         | logistic_regression |   0.108 |       0.213 |    0.190 |         0.852 |      0.127 |          0.182 |       0.195 |            0.835 |      0.019 |         -0.031 |       0.005 |           -0.017 |
| szz         | random_forest       |   0.098 |       0.280 |    0.157 |         0.943 |      0.272 |          0.237 |       0.265 |            0.926 |      0.174 |         -0.043 |       0.108 |           -0.017 |
| szz         | svm                 |   0.050 |       0.184 |    0.093 |         0.859 |      0.128 |          0.143 |       0.195 |            0.805 |      0.078 |         -0.042 |       0.102 |           -0.054 |
| szz         | xgboost             |   0.186 |       0.276 |    0.228 |         0.919 |      0.294 |          0.238 |       0.287 |            0.906 |      0.108 |         -0.038 |       0.060 |           -0.013 |

## 6.13 Temporal within-project validation (Stage 7e)

For each project we pick `T1` (40th percentile of commit dates) and `T2` (70th percentile), rebuild features and consequence labels at each, train at T1 and test at T2. This is the future-data sanity check Falessi et al. (2020) recommend for defect prediction benchmarks. Mean across eligible projects:

| model               |   precision_mean |   recall_mean |   f1_mean |   roc_auc_mean |   pr_auc_mean |   mcc_mean |   ce_at_20_mean |
|:--------------------|-----------------:|--------------:|----------:|---------------:|--------------:|-----------:|----------------:|
| lightgbm            |            0.419 |         0.294 |     0.301 |          0.734 |         0.374 |      0.238 |           0.475 |
| logistic_regression |            0.354 |         0.352 |     0.247 |          0.660 |         0.324 |      0.169 |           0.336 |
| random_forest       |            0.438 |         0.123 |     0.155 |          0.740 |         0.388 |      0.140 |           0.451 |
| xgboost             |            0.457 |         0.232 |     0.258 |          0.719 |         0.367 |      0.217 |           0.443 |

## 6.14 Confusion matrices and per-project errors

From the persisted within-project predictions (`within_project_predictions.parquet`), confusion matrices are computed per (variant, model) over all 10 outer folds combined. Per-project error rates surface heterogeneity that the global F1 hides.

Top-10 worst (variant, model, project) cells by error rate (consequence variant):

| variant     | model               | project_id                    |    n |   tp |   fp |   fn |   tn |   error_rate |
|:------------|:--------------------|:------------------------------|-----:|-----:|-----:|-----:|-----:|-------------:|
| consequence | logistic_regression | org.apache:daemon             |   12 |    3 |    9 |    0 |    0 |        0.750 |
| consequence | logistic_regression | org.apache:commons-fileupload |   50 |    6 |   24 |    3 |   17 |        0.540 |
| consequence | xgboost             | org.apache:daemon             |   12 |    0 |    2 |    3 |    7 |        0.417 |
| consequence | decision_tree       | org.apache:daemon             |   12 |    0 |    2 |    3 |    7 |        0.417 |
| consequence | logistic_regression | org.apache:commons-jelly      |  419 |   43 |  145 |   18 |  213 |        0.389 |
| consequence | logistic_regression | org.apache:httpclient         |  697 |  115 |  235 |   25 |  322 |        0.373 |
| consequence | logistic_regression | org.apache:batik              | 1712 |  255 |  549 |   86 |  822 |        0.371 |
| consequence | logistic_regression | org.apache:commons-jexl       |  158 |   24 |   48 |    8 |   78 |        0.354 |
| consequence | lightgbm            | org.apache:daemon             |   12 |    2 |    3 |    1 |    6 |        0.333 |
| consequence | logistic_regression | org.apache:vfs                |  405 |   75 |  129 |    4 |  197 |        0.328 |

See Figure ``fig_confusion_matrices``.
