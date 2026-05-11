"""
Bootstrap confidence intervals + Wilcoxon paired significance tests.

Adds the inferential rigour required by the proposal Section 3.5
("results will be reported with 95% confidence intervals and pairwise
significance tests") and recommended by the SE benchmarking literature
(Demsar 2006; Herbold et al. 2018). Two complementary procedures are
exposed:

1. :func:`bootstrap_confidence_intervals` - per-(variant, model) /
   per-(variant, model, scope) 95% CIs computed by resampling fold-
   level metrics 10,000 times. Uses the percentile method.

2. :func:`pairwise_wilcoxon` - Wilcoxon signed-rank tests on paired
   per-fold (or per-project) metric differences across all model
   pairs within a variant, with Bonferroni correction applied per
   scope (15 pairs for the 6-model within-project family, 10 pairs
   for the 5-model LOPO family).

Both procedures emit *long-form* tidy DataFrames so the reporting
module can pivot them however it wants.

References
----------
- Efron, B., Tibshirani, R. J. (1993). Introduction to the bootstrap.
- Wilcoxon, F. (1945). Individual comparisons by ranking methods.
- Demsar, J. (2006). Statistical comparisons of classifiers over
  multiple data sets. JMLR.
- Herbold, S., et al. (2018). A comparative study to benchmark
  cross-project defect prediction. IEEE TSE.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import BOOTSTRAP_RESAMPLES, RANDOM_STATE  # noqa: E402


DEFAULT_METRIC_COLS = ("precision", "recall", "f1", "roc_auc", "pr_auc", "mcc", "ce_at_20")


def bootstrap_confidence_intervals(
    fold_df: pd.DataFrame,
    group_cols: Iterable[str] = ("variant", "model"),
    metric_cols: Iterable[str] = DEFAULT_METRIC_COLS,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence_level: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Per-group 95% bootstrap CIs for each metric.

    Parameters
    ----------
    fold_df :
        Long-form per-fold (or per-project) metrics. Must contain at
        least ``group_cols + metric_cols`` columns. Each row is one
        observation - typically one fold of the within-project CV or
        one held-out project of LOPO.
    group_cols :
        Columns identifying a unique cell (e.g.
        ``("variant", "model")`` or
        ``("variant", "model", "method")`` for calibration).
    metric_cols :
        Metric columns to bootstrap. Missing columns are ignored
        gracefully.
    n_resamples :
        Number of resamples (default ``BOOTSTRAP_RESAMPLES`` = 10,000).
    confidence_level :
        Two-sided coverage; 0.95 yields the 2.5/97.5 percentiles.
    random_state :
        Seed for reproducibility.

    Returns
    -------
    DataFrame with columns
    ``(group_cols..., metric, mean, ci_low, ci_high, n)``.
    """
    rng = np.random.default_rng(random_state)
    metric_cols = [c for c in metric_cols if c in fold_df.columns]
    if not metric_cols:
        return pd.DataFrame()

    alpha = (1.0 - confidence_level) / 2.0
    rows: list[dict] = []
    group_cols = list(group_cols)
    for keys, sub in fold_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(sub)
        if n < 2:
            for m in metric_cols:
                rows.append(
                    {
                        **dict(zip(group_cols, keys)),
                        "metric": m,
                        "mean": float(sub[m].mean()) if m in sub else float("nan"),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "n": n,
                    }
                )
            continue
        for m in metric_cols:
            vals = sub[m].dropna().values
            if len(vals) < 2:
                rows.append(
                    {
                        **dict(zip(group_cols, keys)),
                        "metric": m,
                        "mean": float(np.mean(vals)) if len(vals) else float("nan"),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "n": int(len(vals)),
                    }
                )
                continue
            # Vectorised bootstrap: resample indices, compute means.
            idx = rng.integers(0, len(vals), size=(n_resamples, len(vals)))
            boot_means = vals[idx].mean(axis=1)
            ci_low, ci_high = np.percentile(boot_means, [100 * alpha, 100 * (1 - alpha)])
            rows.append(
                {
                    **dict(zip(group_cols, keys)),
                    "metric": m,
                    "mean": float(np.mean(vals)),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "n": int(len(vals)),
                }
            )
    return pd.DataFrame(rows)


def attach_ci_to_summary(
    summary_df: pd.DataFrame,
    ci_df: pd.DataFrame,
    metric_cols: Iterable[str] = DEFAULT_METRIC_COLS,
    group_cols: Iterable[str] = ("variant", "model"),
) -> pd.DataFrame:
    """Pivot ``ci_df`` (long form from :func:`bootstrap_confidence_intervals`)
    onto a summary frame with ``_mean`` columns, attaching ``_ci_low`` and
    ``_ci_high`` columns alongside.
    """
    group_cols = list(group_cols)
    out = summary_df.copy()
    for m in metric_cols:
        sub = ci_df[ci_df["metric"] == m]
        if sub.empty:
            continue
        keep = group_cols + ["ci_low", "ci_high"]
        sub = sub[keep].rename(
            columns={"ci_low": f"{m}_ci_low", "ci_high": f"{m}_ci_high"}
        )
        out = out.merge(sub, on=group_cols, how="left")
    return out


def pairwise_wilcoxon(
    fold_df: pd.DataFrame,
    pair_within: str = "variant",
    contrast_col: str = "model",
    metric: str = "f1",
    paired_on: Iterable[str] = ("fold",),
    bonferroni: bool = True,
) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests across ``contrast_col`` levels.

    For each ``pair_within`` group (typically each variant), enumerate
    all model-vs-model pairs; pair their per-fold (or per-project)
    metric vectors on ``paired_on`` keys, then run
    :func:`scipy.stats.wilcoxon`. Bonferroni-correct p-values within
    each ``pair_within`` group (so 15 pairs at 6 models, 10 at 5).

    Returns one row per pair with ``statistic``, ``p_value``,
    ``p_value_bonferroni``, ``mean_a``, ``mean_b``, ``mean_diff``,
    ``effect_size_r`` (rank-biserial), and ``n_pairs``.
    """
    paired_on = list(paired_on)
    rows: list[dict] = []
    for var, sub_var in fold_df.groupby(pair_within, dropna=False):
        levels = sorted(sub_var[contrast_col].dropna().unique().tolist())
        n_pairs = max(1, len(levels) * (len(levels) - 1) // 2)
        for a, b in combinations(levels, 2):
            sa = sub_var[sub_var[contrast_col] == a].set_index(paired_on)[metric]
            sb = sub_var[sub_var[contrast_col] == b].set_index(paired_on)[metric]
            common = sa.index.intersection(sb.index)
            if len(common) < 3:
                rows.append(
                    {
                        pair_within: var,
                        "model_a": a,
                        "model_b": b,
                        "metric": metric,
                        "n_pairs": int(len(common)),
                        "mean_a": float(sa.reindex(common).mean()) if len(common) else float("nan"),
                        "mean_b": float(sb.reindex(common).mean()) if len(common) else float("nan"),
                        "mean_diff": float("nan"),
                        "statistic": float("nan"),
                        "p_value": float("nan"),
                        "p_value_bonferroni": float("nan"),
                        "effect_size_r": float("nan"),
                    }
                )
                continue
            va = sa.reindex(common).values
            vb = sb.reindex(common).values
            diff = va - vb
            if np.allclose(diff, 0):
                stat, p = 0.0, 1.0
            else:
                try:
                    res = stats.wilcoxon(va, vb, zero_method="zsplit", alternative="two-sided")
                    stat, p = float(res.statistic), float(res.pvalue)
                except ValueError:
                    stat, p = float("nan"), float("nan")
            p_corr = min(1.0, p * n_pairs) if (bonferroni and not np.isnan(p)) else p
            # Rank-biserial effect size approximation: r = z / sqrt(N).
            # We approximate z from the normal-CDF inverse of p (two-sided).
            try:
                z = float(stats.norm.isf(p / 2.0)) if (not np.isnan(p) and p > 0) else float("nan")
                r = z / np.sqrt(len(common)) if not np.isnan(z) else float("nan")
            except (ValueError, ZeroDivisionError):
                r = float("nan")
            rows.append(
                {
                    pair_within: var,
                    "model_a": a,
                    "model_b": b,
                    "metric": metric,
                    "n_pairs": int(len(common)),
                    "mean_a": float(np.mean(va)),
                    "mean_b": float(np.mean(vb)),
                    "mean_diff": float(np.mean(diff)),
                    "statistic": stat,
                    "p_value": p,
                    "p_value_bonferroni": p_corr,
                    "effect_size_r": r,
                }
            )
    return pd.DataFrame(rows)
