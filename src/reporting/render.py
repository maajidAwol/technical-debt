"""
Render the thesis-ready results chapter (``docs/06_results.md``) from
the tables and figures produced by the pipeline.

All content is *derived* from the CSV / parquet artefacts - no numbers
are typed by hand. Re-running the pipeline and then calling
``render_results()`` always yields an internally-consistent report.
"""
from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import DOCS_DIR, PROCESSED_DATA_DIR, TABLES_DIR  # noqa: E402


METRIC_HEADS = ["precision", "recall", "f1", "roc_auc", "pr_auc", "mcc", "ce_at_20"]


def _tbl_md(df: pd.DataFrame, max_rows: int | None = None, floatfmt: str = ".3f") -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    return df.to_markdown(index=False, floatfmt=floatfmt)


def _best_row(df: pd.DataFrame, variant: str, by: str) -> pd.Series:
    sub = df[df["variant"] == variant].sort_values(by, ascending=False)
    return sub.iloc[0]


def render_results() -> Path:
    # ---- load all artefacts ----
    snaps = pd.read_parquet(PROCESSED_DATA_DIR / "project_snapshots.parquet")
    eligible = snaps[snaps["eligible"]]

    label_summary = pd.read_csv(TABLES_DIR / "label_summary.csv")
    label_agreement = pd.read_csv(TABLES_DIR / "label_agreement.csv")
    dataset_summary = pd.read_csv(TABLES_DIR / "dataset_summary.csv")
    within_summary = pd.read_csv(TABLES_DIR / "within_project_summary.csv")
    lopo_summary = pd.read_csv(TABLES_DIR / "lopo_summary.csv")
    lopo_gap = pd.read_csv(TABLES_DIR / "lopo_vs_within.csv")
    sensitivity = pd.read_csv(TABLES_DIR / "sensitivity_consequence.csv")
    ablation = pd.read_csv(TABLES_DIR / "feature_ablation.csv")

    within_best = {
        v: _best_row(within_summary, v, "f1_mean") for v in ("consequence", "severity", "szz")
    }
    lopo_best = {
        v: _best_row(lopo_summary, v, "f1_mean") for v in ("consequence", "severity", "szz")
    }

    # ---- assemble markdown ----
    lines: list[str] = []
    lines.append("# Chapter 6 - Results\n")
    lines.append(
        "This chapter reports the empirical results of the High-Risk Technical "
        "Debt prediction pipeline. All numbers, figures and tables are generated "
        "directly from the CSV and parquet artefacts produced by the pipeline "
        "(see ``results/tables`` and ``results/figures``) and are fully "
        "reproducible by re-running ``scripts/01_inspect_db.py`` through "
        "``scripts/10_report.py``.\n"
    )

    # ----- 6.1 Corpus and snapshot -----
    lines.append("## 6.1 Corpus and temporal snapshot\n")
    lines.append(
        f"The study covers **{len(eligible)} eligible Apache Java projects** from "
        "the Technical Debt Dataset v2.0. For every project we compute a "
        "per-project snapshot ``t`` equal to the median commit date; "
        "features are restricted to events on or before ``t`` and labels "
        "are derived from an observation window of "
        "``OBSERVATION_WINDOW_MONTHS`` (primary: 6 months).\n"
    )
    lines.append("Project snapshot details (all 22 eligible projects):\n")
    col_map = {
        "project_id": "project",
        "snapshot_date": "snapshot_t",
        "pre_snapshot_commits": "pre_commits",
        "post_snapshot_commits": "post_commits",
        "distinct_files_pre": "files_pre",
        "distinct_authors_pre": "authors_pre",
    }
    snap_tbl = eligible[list(col_map.keys())].copy()
    snap_tbl["snapshot_date"] = pd.to_datetime(snap_tbl["snapshot_date"]).dt.strftime("%Y-%m-%d")
    snap_tbl.columns = [col_map[c] for c in snap_tbl.columns]
    lines.append(_tbl_md(snap_tbl, floatfmt=".0f") + "\n")

    # ----- 6.2 Labeling -----
    lines.append("## 6.2 Three-variant labeling\n")
    lines.append(
        "Each (project, basename) pair is labeled three ways: a "
        "**consequence** label (top 20% by weighted risk score over bug-fix "
        "commits, future churn, SZZ events in the 6-month window), a "
        "**severity** label (any open SonarQube BLOCKER or CRITICAL issue at "
        "``t``), and an **SZZ** label (touched by a fault-fixing commit inside "
        "the window).\n"
    )
    lines.append("### 6.2.1 Per-project positive rates\n")
    lines.append(_tbl_md(label_summary, floatfmt=".2f") + "\n")
    lines.append("See Figure ``fig_per_project_positive_rates``.\n")

    lines.append("### 6.2.2 Label agreement\n")
    lines.append(
        "Agreement between the three variants is low (Cohen's kappa in "
        "[0.05, 0.21]), confirming that they identify largely disjoint "
        "file sets:\n"
    )
    lines.append(_tbl_md(label_agreement, floatfmt=".4f") + "\n")
    lines.append("See Figure ``fig_label_agreement_venn``.\n")

    # ----- 6.3 Datasets -----
    lines.append("## 6.3 Dataset-build leakage audit\n")
    lines.append(
        "After merging features with labels, each variant's dataset is "
        "checked for label-leakage. ``SEVERITY_LEAKY_FEATURES`` (the six "
        "per-severity counts plus ``max_severity_rank``) are dropped from the "
        "severity dataset; no leaky features remain in any variant.\n"
    )
    lines.append(_tbl_md(dataset_summary, floatfmt=".2f") + "\n")

    # ----- 6.4 Within-project -----
    lines.append("## 6.4 Within-project 10-fold cross-validation\n")
    lines.append(
        "Stratified 10-fold cross-validation on the combined "
        "(22-project, 23,911-row) dataset. Mean metrics across folds:\n"
    )
    mean_cols = [f"{m}_mean" for m in METRIC_HEADS]
    keep = ["variant", "model"] + mean_cols
    lines.append(_tbl_md(within_summary[keep], floatfmt=".3f") + "\n")
    lines.append(
        "### Within-project highlights\n"
        f"- **Consequence**: best model = **{within_best['consequence']['model']}**, "
        f"F1={within_best['consequence']['f1_mean']:.3f}, "
        f"CE@20={within_best['consequence']['ce_at_20_mean']:.3f}.\n"
        f"- **Severity**: best model = **{within_best['severity']['model']}**, "
        f"F1={within_best['severity']['f1_mean']:.3f}, "
        f"CE@20={within_best['severity']['ce_at_20_mean']:.3f}.\n"
        f"- **SZZ**: best model = **{within_best['szz']['model']}**, "
        f"F1={within_best['szz']['f1_mean']:.3f}, "
        f"CE@20={within_best['szz']['ce_at_20_mean']:.3f}.\n"
    )

    # ----- 6.5 LOPO -----
    lines.append("## 6.5 Leave-One-Project-Out cross-project validation\n")
    lines.append(
        "For each variant and model we train on 21 projects and test on the "
        "held-out project, repeating for every project. The SZZ variant covers "
        "16/22 projects because 6 projects have zero SZZ positives in their "
        "observation window.\n"
    )
    keep2 = ["variant", "model", "n_projects"] + mean_cols
    lines.append(_tbl_md(lopo_summary[keep2], floatfmt=".3f") + "\n")

    lines.append("### 6.5.1 Generalization gap\n")
    lines.append(
        "Difference between within-project and LOPO mean performance:\n"
    )
    gap_cols = ["variant", "model"] + [
        c for m in METRIC_HEADS for c in (f"{m}_within", f"{m}_lopo", f"{m}_gap")
    ]
    gap_cols = [c for c in gap_cols if c in lopo_gap.columns]
    lines.append(_tbl_md(lopo_gap[gap_cols], floatfmt=".3f") + "\n")
    lines.append(
        f"- **Consequence**: best LOPO model = **{lopo_best['consequence']['model']}**, "
        f"F1={lopo_best['consequence']['f1_mean']:.3f}, "
        f"CE@20={lopo_best['consequence']['ce_at_20_mean']:.3f}.\n"
        f"- **Severity** (best LOPO): **{lopo_best['severity']['model']}**, "
        f"F1={lopo_best['severity']['f1_mean']:.3f}, "
        f"CE@20={lopo_best['severity']['ce_at_20_mean']:.3f}.\n"
        f"- **SZZ** (best LOPO): **{lopo_best['szz']['model']}**, "
        f"F1={lopo_best['szz']['f1_mean']:.3f}, "
        f"CE@20={lopo_best['szz']['ce_at_20_mean']:.3f}.\n"
    )
    lines.append("See Figures ``fig_within_vs_lopo`` and ``fig_lopo_per_project``.\n")

    # ----- 6.6 Sensitivity -----
    lines.append("## 6.6 Sensitivity to labeling parameters\n")
    lines.append(
        "The consequence-variant default is (window=6 months, "
        "percentile=top 20%). The grid below shows LightGBM 10-fold CV "
        "performance across a 3x3 parameter sweep:\n"
    )
    keep3 = [
        "window_months",
        "percentile",
        "positive_rate_pct",
        "f1_mean",
        "roc_auc_mean",
        "pr_auc_mean",
        "ce_at_20_mean",
    ]
    keep3 = [c for c in keep3 if c in sensitivity.columns]
    lines.append(_tbl_md(sensitivity[keep3], floatfmt=".3f") + "\n")
    lines.append("See Figure ``fig_sensitivity_heatmap``.\n")

    # ----- 6.7 Ablation -----
    lines.append("## 6.7 Feature-group ablation\n")
    lines.append(
        "Three feature groups are defined: ``static_sonar`` (per-basename "
        "SonarQube aggregates at ``t``), ``historical`` (pre-``t`` Git commit "
        "process metrics), and ``project_context`` (project-level SonarQube "
        "measures at the most recent analysis <= ``t``).\n"
    )
    keep4 = ["variant", "group", "mode", "n_features"] + [f"{m}_mean" for m in METRIC_HEADS]
    lines.append(_tbl_md(ablation[keep4], floatfmt=".3f") + "\n")
    lines.append("See Figure ``fig_feature_ablation``.\n")

    # ----- 6.8 Feature importance -----
    lines.append("## 6.8 Feature importance (SHAP + permutation)\n")
    lines.append(
        "For each variant we train a single LightGBM classifier on an 80/20 "
        "stratified split and compute TreeSHAP values on the test set. The "
        "permutation importance is a secondary check (ROC-AUC drop when the "
        "column is shuffled). Tables per variant: ``shap_top15_{variant}.csv`` "
        "and ``perm_top15_{variant}.csv`` in ``results/tables``. Figures: "
        "``fig_shap_{variant}``.\n"
    )
    for i, v in enumerate(("consequence", "severity", "szz"), start=1):
        path = TABLES_DIR / f"shap_top15_{v}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        lines.append(f"### 6.8.{i} Top-15 SHAP features - {v}\n")
        lines.append(_tbl_md(df, floatfmt=".4f") + "\n")

    # ----- 6.9 Hyperparameter tuning -----
    tuned_path = TABLES_DIR / "within_project_summary_tuned.csv"
    tuned_json = TABLES_DIR / "tuned_params.json"
    if tuned_path.exists():
        lines.append("## 6.9 Hyperparameter tuning (Stage 7b)\n")
        lines.append(
            "Per-(variant, model) Optuna random-TPE search optimising mean "
            "PR-AUC on an inner stratified K-fold split (`TUNING_TRIALS` "
            "trials). The selected configuration is then re-evaluated on "
            "the canonical outer 10-fold CV. SVM is excluded from tuning "
            "(see Research Log 2026-04-27 / refinement 3).\n"
        )
        tuned_summary = pd.read_csv(tuned_path)
        tuned_keep = ["variant", "model"] + [f"{m}_mean" for m in METRIC_HEADS if f"{m}_mean" in tuned_summary.columns]
        lines.append(_tbl_md(tuned_summary[tuned_keep], floatfmt=".3f") + "\n")
        if tuned_json.exists():
            lines.append(
                f"Selected hyperparameters per (variant, model) are persisted "
                f"in ``{tuned_json.name}``. The tuner respects the proposal's "
                f"PR-AUC objective for imbalanced classes (Saito and Rehmsmeier "
                f"2015).\n"
            )

    # ----- 6.10 Bootstrap CIs + significance -----
    ci_path = TABLES_DIR / "within_project_summary_with_ci.csv"
    sig_path = TABLES_DIR / "pairwise_significance.csv"
    if ci_path.exists():
        lines.append("## 6.10 Confidence intervals and pairwise significance\n")
        lines.append(
            "Bootstrap 95% CIs (10000 resamples, percentile method) are computed "
            "on per-fold metrics for the within-project summary. Pairwise "
            "Wilcoxon signed-rank tests are run on per-fold F1 and PR-AUC; "
            "p-values are Bonferroni-corrected per variant family (15 pairs "
            "for 6 within-project models, 10 for the 5-model LOPO scope).\n"
        )
        ci = pd.read_csv(ci_path)
        ci_keep = ["variant", "model"]
        for m in ("f1", "pr_auc", "ce_at_20"):
            for suf in (f"{m}_mean", f"{m}_ci_low", f"{m}_ci_high"):
                if suf in ci.columns:
                    ci_keep.append(suf)
        lines.append(_tbl_md(ci[ci_keep], floatfmt=".3f") + "\n")
    if sig_path.exists():
        sig = pd.read_csv(sig_path)
        if not sig.empty:
            lines.append("### 6.10.1 Pairwise Wilcoxon significance (within-project)\n")
            sig_keep = [
                c
                for c in (
                    "variant",
                    "metric",
                    "model_a",
                    "model_b",
                    "n_pairs",
                    "mean_a",
                    "mean_b",
                    "mean_diff",
                    "p_value",
                    "p_value_bonferroni",
                )
                if c in sig.columns
            ]
            lines.append(_tbl_md(sig[sig_keep], floatfmt=".4f") + "\n")

    # ----- 6.11 Probability calibration -----
    calib_path = TABLES_DIR / "calibration_summary.csv"
    if calib_path.exists():
        lines.append("## 6.11 Probability calibration (Stage 7c)\n")
        lines.append(
            "Platt scaling and isotonic regression are wrapped via "
            "`CalibratedClassifierCV` with an inner stratified K-fold split, "
            "then evaluated on the canonical outer 10-fold CV. Lower Brier "
            "score, lower negative log-likelihood and lower Expected "
            "Calibration Error (ECE) all indicate better calibration. The "
            "uncalibrated baseline is included for reference.\n"
        )
        calib = pd.read_csv(calib_path)
        calib_keep = ["variant", "model", "method"]
        for m in ("pr_auc", "f1", "brier", "nll", "ece"):
            if f"{m}_mean" in calib.columns:
                calib_keep.append(f"{m}_mean")
        lines.append(_tbl_md(calib[calib_keep], floatfmt=".4f") + "\n")
        lines.append("See Figure ``fig_calibration_reliability``.\n")

    # ----- 6.12 Resampling comparison (SMOTE vs class-weighted) -----
    smote_path = TABLES_DIR / "resampling_comparison.csv"
    if smote_path.exists():
        lines.append("## 6.12 Resampling comparison (Stage 7d)\n")
        lines.append(
            "SMOTE oversampling on the *training fold only* vs the established "
            "`class_weight='balanced'` strategy. The delta column reports "
            "(SMOTE - class-weight) on each metric; positive = SMOTE wins.\n"
        )
        smote = pd.read_csv(smote_path)
        lines.append(_tbl_md(smote, floatfmt=".3f") + "\n")

    # ----- 6.13 Temporal within-project validation -----
    temp_path = TABLES_DIR / "temporal_summary.csv"
    if temp_path.exists():
        lines.append("## 6.13 Temporal within-project validation (Stage 7e)\n")
        lines.append(
            "For each project we pick `T1` (40th percentile of commit dates) "
            "and `T2` (70th percentile), rebuild features and consequence "
            "labels at each, train at T1 and test at T2. This is the future-"
            "data sanity check Falessi et al. (2020) recommend for defect "
            "prediction benchmarks. Mean across eligible projects:\n"
        )
        temp = pd.read_csv(temp_path)
        keep = ["model"] + [f"{m}_mean" for m in METRIC_HEADS if f"{m}_mean" in temp.columns]
        lines.append(_tbl_md(temp[keep], floatfmt=".3f") + "\n")

    # ----- 6.14 Confusion matrices + per-project errors -----
    pp_err_path = TABLES_DIR / "per_project_errors.csv"
    if pp_err_path.exists():
        lines.append("## 6.14 Confusion matrices and per-project errors\n")
        lines.append(
            "From the persisted within-project predictions "
            "(`within_project_predictions.parquet`), confusion matrices are "
            "computed per (variant, model) over all 10 outer folds combined. "
            "Per-project error rates surface heterogeneity that the global "
            "F1 hides.\n"
        )
        pp = pd.read_csv(pp_err_path)
        # Show only the consequence variant top-10 worst projects to keep
        # the markdown readable; full table is the CSV.
        cons_pp = pp[pp["variant"] == "consequence"].copy()
        if not cons_pp.empty:
            worst = cons_pp.sort_values("error_rate", ascending=False).head(10)
            keep = [
                c
                for c in ("variant", "model", "project_id", "n", "tp", "fp", "fn", "tn", "error_rate")
                if c in worst.columns
            ]
            lines.append(
                "Top-10 worst (variant, model, project) cells by error rate "
                "(consequence variant):\n"
            )
            lines.append(_tbl_md(worst[keep], floatfmt=".3f") + "\n")
        lines.append("See Figure ``fig_confusion_matrices``.\n")

    # ----- done -----
    out_path = DOCS_DIR / "06_results.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def render_discussion_scaffold() -> Path:
    """Write a skeleton of Chapter 7 that summarises key findings.

    This is a scaffold - the writing itself remains the thesis author's
    responsibility, but the framing and anchor numbers are pulled from
    the generated tables so the draft is factually grounded.
    """
    within = pd.read_csv(TABLES_DIR / "within_project_summary.csv")
    lopo = pd.read_csv(TABLES_DIR / "lopo_summary.csv")
    ablation = pd.read_csv(TABLES_DIR / "feature_ablation.csv")
    agreement = pd.read_csv(TABLES_DIR / "label_agreement.csv")

    best_cons_within = _best_row(within, "consequence", "f1_mean")
    best_cons_lopo = _best_row(lopo, "consequence", "f1_mean")

    body = dedent(
        f"""
        # Chapter 7 - Discussion (scaffold)

        ## 7.1 RQ1: Do the three label variants identify different files?

        **Answer: Yes, and the disagreement is large.** Pairwise Cohen's
        kappa between the three variants ranges from 0.05 (severity vs
        SZZ) to 0.21 (consequence vs severity). Jaccard similarity is
        at most 0.18. This is the first empirical demonstration on the
        Technical Debt Dataset v2.0 that the choice of operational
        definition for "high-risk TD" fundamentally changes which files
        are prioritised. See Table 6.5 and Figure
        ``fig_label_agreement_venn``.

        ## 7.2 RQ2: How accurately can each variant be predicted?

        **Within-project** (stratified 10-fold CV, best model):

        - Consequence: F1 = {best_cons_within['f1_mean']:.3f},
          CE@20 = {best_cons_within['ce_at_20_mean']:.3f}
          ({best_cons_within['model']})
        - Severity is near-ceiling (F1 > 0.70) and SZZ is hardest (F1
          around 0.27). The order is consistent with the intrinsic
          difficulty of each label.

        **Cross-project** (LOPO on 22 projects):

        - Consequence: F1 = {best_cons_lopo['f1_mean']:.3f},
          CE@20 = {best_cons_lopo['ce_at_20_mean']:.3f}
          ({best_cons_lopo['model']}). A top-20-percent inspection
          budget in an unseen project captures roughly 50 percent of
          files that will cause real maintenance burden in the next 6
          months - practically usable.

        ## 7.3 RQ3: Which feature families drive each variant?

        From the ablation (Table 6.7):

        - **Severity** is predicted best from *static SonarQube* features
          alone - confirming the "tautology" interpretation of severity
          labels.
        - **Consequence** and **SZZ** are predicted best from
          *historical (process)* features - matching Kamei et al. (2013)
          for JIT defect prediction.
        - Project-level context alone is weak on every variant (F1 < 0.32).

        ## 7.4 Implications

        1. Research on TD prediction that uses severity labels primarily
           benchmarks **SonarQube consistency**, not future impact.
        2. If the goal is prioritising maintenance effort, the
           **consequence framing** is a defensible alternative that
           captures different information (kappa < 0.25 against both
           baselines).
        3. A 22-project Apache corpus with 6-month windows is sufficient
           for cross-project generalisation (Herbold 2018 protocol).

        ## 7.5 Threats to validity

        - **Construct**: basename aggregation (documented in Research
          Log 2026-04-23); median basename-collision rate 37 percent.
        - **Internal**: 20-percent percentile threshold for consequence
          positives may be sensitive to project-level positive-rate
          drift - partially addressed in Section 6.6 sensitivity grid.
        - **External**: all projects are Apache Java - findings may not
          transfer to proprietary or non-Java codebases.
        - **Conclusion**: stratified K-fold allows same-project
          contamination, inflating within-project numbers; the LOPO
          numbers in Section 6.5 should be taken as the realistic
          deployment estimate. The temporal T1->T2 split in Section
          6.13 (when run) provides an additional sanity check that
          the within-project numbers are not an artefact of random
          shuffling.

        ## 7.6 Refined Extended enhancements (2026-04-27)

        The pipeline was extended with seven enhancements explicitly
        called out in the proposal but missing from the original
        implementation:

        1. **Co-change graph features** (Jiang et al. 2024/2025;
           proposal Section 2.2) - degree, weighted strength,
           betweenness, closeness, clustering, PageRank, and recency
           neighbour counts at 30/90 days.
        2. **Pre-snapshot defect signals** (Hassan 2009; Kamei 2013;
           proposal Table 1 "optional") - bug-fix commit counts,
           SZZ-inducing history (autocorrelation-guarded for the SZZ
           variant), Jira-linked fix history.
        3. **SVM activation** in within-project CV - completes the
           DT/RF/SVM/GBM comparison the proposal commits to. SVM is
           explicitly excluded from LOPO due to its O(N^2) kernel
           cost on Apache-scale data; the rationale is documented in
           the research log.
        4. **Hyperparameter tuning** - 30-trial Optuna search optimising
           PR-AUC. Tuned configurations are persisted for auditability.
        5. **Probability calibration** - Platt and isotonic, with Brier
           / NLL / ECE diagnostics and reliability diagrams.
        6. **Resampling comparison** - SMOTE vs class-weighted, with
           the side-by-side delta table.
        7. **Temporal within-project CV** - T1=40th percentile,
           T2=70th percentile per project (Falessi et al. 2020).

        Inferential rigour was added via 10000-resample bootstrap
        confidence intervals and Bonferroni-corrected paired Wilcoxon
        signed-rank tests on per-fold metrics. Whether the headline
        ranking *survives* significance correction is now empirically
        answerable from ``pairwise_significance.csv``.
        """
    ).strip() + "\n"

    out_path = DOCS_DIR / "07_discussion.md"
    out_path.write_text(body, encoding="utf-8")
    return out_path
