"""
Stage 12b - Held-out case study using the persisted best model.

Loads ``models/best_model.pkl`` + scaler + feature names + threshold, picks
one project from ``dataset_final.parquet`` (default: org.apache:zookeeper),
scores its files, and prints the top-10 highest-risk basenames plus the
top-20% review-budget summary that appears in Section 14 of the Colab
notebook.

This is intentionally separate from Stage 11 (which persists the model
artefacts) and Stage 13 (which scores an arbitrary GitHub repo): Stage 12b
exercises the *full* 27-feature path against held-out training data, so
it's a direct anchor for the CE@20 number reported in the model card.

Run
---
.. code-block:: bash

    python scripts/12b_case_study.py
    python scripts/12b_case_study.py --project org.apache:codec --top 20

Outputs
-------
- ``results/tables/case_study_<project>.csv``: per-file scores
  (basename, risk_score, predicted_high_risk, is_high_risk).
- Console: top-K table + CE@20 summary.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    ALL_FEATURES,
    COST_EFFECTIVENESS_AT,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    TABLES_DIR,
)


DEFAULT_PROJECT = "org.apache:zookeeper"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default=DEFAULT_PROJECT, help=f"Project ID to score (default: {DEFAULT_PROJECT})")
    p.add_argument("--top", type=int, default=10, help="Number of top-ranked files to print (default: 10)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    t0 = time.time()
    print(f"[Stage 12b] Project       : {args.project}")
    print(f"[Stage 12b] Top-K display : {args.top}")

    ds = pd.read_parquet(PROCESSED_DATA_DIR / "dataset_final.parquet")
    demo = ds[ds["project_id"] == args.project].copy().reset_index(drop=True)
    if demo.empty:
        available = sorted(ds["project_id"].unique())
        raise SystemExit(f"No rows for project {args.project!r}. Available:\n  " + "\n  ".join(available))

    pos_rate = 100 * float(demo["is_high_risk"].mean())
    print(f"[Stage 12b] Files         : {len(demo)}")
    print(f"[Stage 12b] Positive rate : {pos_rate:.2f}%")

    # Load persisted artefacts
    best_model = joblib.load(MODELS_DIR / "best_model.pkl")
    scaler = joblib.load(MODELS_DIR / "feature_scaler.pkl")
    feat_names = pd.read_csv(MODELS_DIR / "feature_names.csv")["feature"].tolist()
    threshold = float((MODELS_DIR / "optimal_threshold.txt").read_text())
    assert feat_names == list(ALL_FEATURES), "feature_names.csv does not match config.ALL_FEATURES"

    # dataset_final.parquet is already log1p-transformed (Stage 6), so we do
    # NOT re-apply log1p here. The persisted scaler was fitted on the same
    # log1p-transformed matrix, so feeding the parquet directly preserves
    # train/inference symmetry.
    X = demo[feat_names].fillna(0).astype(float).values
    X_scaled = scaler.transform(X)
    probs = best_model.predict_proba(X_scaled)[:, 1]

    scored = pd.DataFrame(
        {
            "basename": demo["basename"],
            "risk_score": probs,
            "predicted_high_risk": (probs >= threshold).astype(int),
            "is_high_risk": demo["is_high_risk"].astype(int),
        }
    ).sort_values("risk_score", ascending=False).reset_index(drop=True)

    out_path = TABLES_DIR / f"case_study_{args.project.replace(':', '_')}.csv"
    scored.to_csv(out_path, index=False)
    print(f"\n[Stage 12b] Wrote {out_path}")

    print(f"\n[Stage 12b] Top {args.top} highest-risk files:")
    with pd.option_context("display.width", 160, "display.max_rows", None):
        print(scored.head(args.top).to_string(index=False))

    # CE@20 summary against held-out labels
    n = len(scored)
    k = max(1, int(round(COST_EFFECTIVENESS_AT * n)))
    n_pos_total = int(scored["is_high_risk"].sum())
    n_pos_found = int(scored.head(k)["is_high_risk"].sum())
    ce_at_20 = n_pos_found / n_pos_total if n_pos_total else 0.0
    n_high_pred = int(scored["predicted_high_risk"].sum())

    print(f"\n[Stage 12b] Total files                    : {n}")
    print(f"[Stage 12b] Files in top {int(COST_EFFECTIVENESS_AT*100)}% review budget : {k}")
    print(f"[Stage 12b] Truly high-risk files           : {n_pos_total}")
    print(f"[Stage 12b] Found in top {int(COST_EFFECTIVENESS_AT*100)}%               : {n_pos_found}")
    print(f"[Stage 12b] CE@20                           : {ce_at_20:.4f}")
    print(f"[Stage 12b] Predicted high-risk (threshold={threshold}): {n_high_pred}")

    print(f"\n[Stage 12b] Elapsed: {time.time() - t0:.1f}s")
    print("[Stage 12b] Complete.")


if __name__ == "__main__":
    main()
