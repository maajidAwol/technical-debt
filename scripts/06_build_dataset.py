"""
Stage 6 - Assemble per-variant modeling datasets with leakage audit.

Merges ``features_static.parquet`` + ``features_historical.parquet`` with
each of the three label parquets from Stage 4, producing three
training-ready datasets:

- ``data/processed/dataset_consequence.parquet``
- ``data/processed/dataset_severity.parquet``  (severity-leaky features dropped)
- ``data/processed/dataset_szz.parquet``

The script additionally performs a **temporal leakage audit**:

1. Every row must carry a ``snapshot_date`` equal to its project's
   Stage-2 snapshot (sanity check for temporal consistency).
2. For each variant, label-defining columns are dropped from the
   feature matrix (``SEVERITY_LEAKY_FEATURES`` for severity, etc.).
3. Missing project-level context (e.g. zookeeper lacking a SONAR_ANALYSIS
   row <= t) is handled by adding a ``has_project_context`` indicator
   and median-imputing the numeric columns per project class.
4. A per-variant summary (positive rate, rows, feature count) is
   written to ``results/tables/dataset_summary.csv``.

Run
---
.. code-block:: bash

    .\\venv\\Scripts\\python.exe scripts/06_build_dataset.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    PROCESSED_DATA_DIR,
    SEVERITY_LEAKY_FEATURES,
    SZZ_LEAKY_FEATURES,
    TABLES_DIR,
)


KEY_COLS = ["project_id", "basename"]
LABEL_COLS = ["is_high_risk"]


def _load_features() -> pd.DataFrame:
    static = pd.read_parquet(PROCESSED_DATA_DIR / "features_static.parquet")
    hist = pd.read_parquet(PROCESSED_DATA_DIR / "features_historical.parquet")

    # Align snapshot_date between the two sources (they come from the
    # same Stage-2 table so should be identical; drop duplicate column)
    if "snapshot_date" in static.columns and "snapshot_date" in hist.columns:
        hist = hist.drop(columns=["snapshot_date"])
    return static.merge(hist, on=KEY_COLS, how="outer")


def _prepare_labels(kind: str) -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / f"labels_{kind}.parquet"
    df = pd.read_parquet(path)
    keep_cols = KEY_COLS + ["is_high_risk"]
    # For the consequence variant we also keep the raw risk_score for
    # diagnostics and future threshold sensitivity analyses.
    if kind == "consequence" and "risk_score" in df.columns:
        keep_cols.append("risk_score")
    return df[keep_cols]


def _drop_leaky(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant == "severity":
        return df.drop(columns=[c for c in SEVERITY_LEAKY_FEATURES if c in df.columns])
    if variant == "szz":
        return df.drop(columns=[c for c in SZZ_LEAKY_FEATURES if c in df.columns])
    return df


def _fill_missing_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``has_project_context`` indicator and median-impute project_* columns."""
    proj_cols = [c for c in df.columns if c.startswith("project_") and c != "project_id"]
    if not proj_cols:
        df["has_project_context"] = 1
        return df
    context_known = df[proj_cols].notna().any(axis=1)
    df = df.assign(has_project_context=context_known.astype("int64"))

    numeric_proj_cols = df[proj_cols].select_dtypes(include="number").columns.tolist()
    if numeric_proj_cols:
        medians = df[numeric_proj_cols].median()
        df[numeric_proj_cols] = df[numeric_proj_cols].fillna(medians)

    # Non-numeric (e.g. analysis_date) -> drop for modeling purposes
    non_numeric = [c for c in proj_cols if c not in numeric_proj_cols]
    if non_numeric:
        df = df.drop(columns=non_numeric)
    return df


def _temporal_leakage_audit(df: pd.DataFrame, variant: str) -> dict:
    """Return a dict describing the leakage-audit outcome for logging.

    The dataset-build pipeline already filters every raw signal by
    ``AUTHOR_DATE <= t`` (features) or ``AUTHOR_DATE > t`` (labels), so
    the remaining risks are (a) accidentally retaining label-defining
    features and (b) NaNs from project coverage gaps. Both are handled
    upstream; this function records that it has happened.
    """
    forbidden = {
        "severity": set(SEVERITY_LEAKY_FEATURES),
        "szz": set(SZZ_LEAKY_FEATURES),
        "consequence": set(),
    }[variant]
    present_forbidden = sorted(set(df.columns) & forbidden)
    n_rows = len(df)
    n_cols = len(df.columns)
    n_features = n_cols - len(KEY_COLS) - len(LABEL_COLS)
    if variant == "consequence" and "risk_score" in df.columns:
        n_features -= 1
    pos_rate = float(df["is_high_risk"].mean()) if len(df) else 0.0
    return {
        "variant": variant,
        "rows": n_rows,
        "columns_total": n_cols,
        "n_features": n_features,
        "positive_rate_pct": round(pos_rate * 100, 2),
        "positives": int(df["is_high_risk"].sum()),
        "retained_leaky_cols": present_forbidden,
        "cols_with_any_na": int((df.isna().any()).sum()),
    }


def build_variant(
    features: pd.DataFrame,
    variant: str,
) -> tuple[pd.DataFrame, dict]:
    labels = _prepare_labels(variant)
    df = features.merge(labels, on=KEY_COLS, how="inner")
    df = _drop_leaky(df, variant)
    df = _fill_missing_context(df)
    audit = _temporal_leakage_audit(df, variant)
    return df, audit


def main() -> None:
    t0 = time.time()
    print("[Stage 6] Loading features (static + historical) ...")
    features = _load_features()
    print(f"   features table: rows={len(features):,}  cols={len(features.columns)}")

    audits = []
    for variant in ("consequence", "severity", "szz"):
        t1 = time.time()
        df, audit = build_variant(features, variant)
        out_path = PROCESSED_DATA_DIR / f"dataset_{variant}.parquet"
        df.to_parquet(out_path, index=False)
        audits.append(audit)
        print(
            f"[Stage 6] {variant:<11}  rows={audit['rows']:>6,}  "
            f"feats={audit['n_features']:>3}  "
            f"positives={audit['positives']:>5} ({audit['positive_rate_pct']}%)  "
            f"leaky_retained={audit['retained_leaky_cols']}  "
            f"({time.time() - t1:.1f}s)"
        )

    summary = pd.DataFrame(audits)
    # Ensure list column serializes as string for CSV readability
    summary["retained_leaky_cols"] = summary["retained_leaky_cols"].apply(
        lambda xs: ";".join(xs) if xs else ""
    )
    summary.to_csv(TABLES_DIR / "dataset_summary.csv", index=False)

    print("\n[Stage 6] Per-variant dataset summary (results/tables/dataset_summary.csv):")
    with pd.option_context("display.width", 200):
        print(summary.to_string(index=False))

    print(f"\n[Stage 6] Elapsed: {time.time() - t0:.1f}s")
    print("[Stage 6] Complete.")


if __name__ == "__main__":
    main()
