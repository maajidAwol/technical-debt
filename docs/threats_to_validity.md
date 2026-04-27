# Threats to Validity

*Paper-ready source for Chapter 6.*

## Construct Validity

**Threat**: "High-risk TD" is not directly observable; we approximate it
with maintenance-outcome proxies (bug-fix commits, churn, SZZ defects).

**Mitigation**: We report three label variants and quantify their
agreement (Cohen's kappa, Jaccard). We also manually inspect 20 randomly
sampled positive files per project and variant to confirm they reflect
expert intuition of high-risk.

## Internal Validity

**Threat**: Label leakage if features inadvertently encode the label
definition.

**Mitigation**: Stage 6 performs an automated leakage audit (correlation
>0.95 flagged). Severity-count features are explicitly dropped for the
severity variant.

**Threat**: Temporal leakage if features use post-snapshot data.

**Mitigation**: All queries filter on `AUTHOR_DATE <= t` for historical
features and `SNAPSHOT_DATE <= t` for static features. Stage 6 verifies
no feature column values differ between (`t`) and (`t + 1 day`) runs.

## External Validity

**Threat**: Generalization beyond Apache Java projects.

**Mitigation**: LOPO evaluation simulates unseen-project performance for
33 diverse Apache projects. Optional Stage 10 extension adds 1-2
non-Apache Java projects. We explicitly caveat non-Java / non-Apache
generalization.

## Conclusion Validity

**Threat**: Small project counts may produce high-variance LOPO results.

**Mitigation**: We report mean +/- std across folds and use Wilcoxon
signed-rank with Holm correction for multiple comparisons.

## Reliability

**Threat**: Reproducibility of results.

**Mitigation**: Single SQLite input file (citable), deterministic SQL
queries, `RANDOM_STATE = 42` throughout, Parquet intermediate artifacts,
complete `RESEARCH_LOG.md` decision log.
