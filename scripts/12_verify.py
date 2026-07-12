"""
Stage 12 - End-to-end pipeline verification.

Runs every assertion from the acceptance checklist and exits non-zero
on the first failure with a clear message. Designed to be called
manually after a full pipeline run::

    python scripts/12_verify.py

Returns 0 on full pass.
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "OK  " if condition else "FAIL"
    print(f"  {mark}  {name}{(' - ' + detail) if detail else ''}")
    if not condition:
        raise AssertionError(name + (": " + detail if detail else ""))


def main() -> int:
    print("[Stage 12] Running verification checklist ...")

    # ---------------- DATA ----------------
    print("\n[Stage 12] DATA")
    labels = pd.read_parquet("data/processed/labels.parquet")
    check(
        "labels.parquet is_high_risk is binary",
        set(labels["is_high_risk"].unique()) <= {0, 1},
    )
    pos_rate = float(labels["is_high_risk"].mean())
    check(
        "overall positive rate in [8%, 18%]",
        0.08 <= pos_rate <= 0.18,
        f"got {100*pos_rate:.2f}%",
    )
    per_proj_pos = labels.groupby("project_id")["is_high_risk"].sum()
    check(
        ">=20 of 22 projects have >=5 positives",
        int((per_proj_pos >= 5).sum()) >= 20,
        f"got {int((per_proj_pos >= 5).sum())}",
    )

    ds = pd.read_parquet("data/processed/dataset_final.parquet")
    n_feats = ds.drop(columns=["project_id", "basename"]).shape[1]
    check(
        "dataset_final has 27 features + is_high_risk (28 modelling cols)",
        n_feats == 28,
        f"got {n_feats}",
    )
    check("no NaN in dataset_final", int(ds.isna().sum().sum()) == 0)
    check(
        "no leaky severity counts in dataset_final",
        "n_blocker" not in ds.columns and "n_critical" not in ds.columns,
    )
    check(
        "no surrogate column in dataset_final",
        "future_bugfix_count" not in ds.columns,
    )

    weights = json.load(open("data/processed/derived_weights.json"))
    weights_sum = sum(v for k, v in weights.items() if not k.startswith("_"))
    check(
        "derived_weights sums to 1.0 (+- 1e-9)",
        abs(weights_sum - 1.0) < 1e-9,
        f"got {weights_sum}",
    )

    # ---------------- MODELS ----------------
    print("\n[Stage 12] MODELS")
    within = pd.read_csv("results/tables/within_project_results.csv")
    lopo = pd.read_csv("results/tables/lopo_results.csv")
    check("within_project_results has 4 model rows (unique)", within["model"].nunique() == 4)
    check("lopo_results has 4 model rows", lopo["model"].nunique() == 4)
    check("within_project_results has single 'f1' column", "f1" in within.columns and "f1_optimized" not in within.columns)

    mc = pd.read_csv("results/tables/model_comparison.csv")
    check("model_comparison flags exactly one best model", int(mc["is_best"].sum()) == 1)
    best = mc[mc["is_best"]].iloc[0]
    check("best LOPO F1 > 0.55", float(best["lopo_f1"]) > 0.55, f"got {best['lopo_f1']:.4f}")
    check("best within-project F1 > 0.65", float(best["within_f1"]) > 0.65, f"got {best['within_f1']:.4f}")
    check("best LOPO CE@20 > 0.65", float(best["lopo_ce_at_20"]) > 0.65, f"got {best['lopo_ce_at_20']:.4f}")

    # ---------------- PERSISTENCE ----------------
    print("\n[Stage 12] PERSISTENCE")
    import joblib
    model = joblib.load("models/best_model.pkl")
    check("best_model.pkl loads", model is not None)

    card = json.load(open("models/model_card.json"))
    check("model_card.json threshold == 0.5", float(card["threshold"]) == 0.5, f"got {card['threshold']}")
    check("model_card.json n_features == 27", int(card["n_features"]) == 27)

    fn = pd.read_csv("models/feature_names.csv")
    check("feature_names.csv has 27 rows", len(fn) == 27)

    # ---------------- OUTPUTS ----------------
    print("\n[Stage 12] OUTPUTS")
    csv_count = len(glob.glob("results/tables/*.csv"))
    check("results/tables has 10 CSV files", csv_count == 10, f"got {csv_count}")
    png_count = len(glob.glob("results/figures/*.png"))
    check("results/figures has 12 PNG files", png_count == 12, f"got {png_count}")
    check(
        "feature_catalog.csv has 27 rows",
        len(pd.read_csv("data/processed/feature_catalog.csv")) == 27,
    )
    check(
        "corpus_summary.csv has 22 rows",
        len(pd.read_csv("results/tables/corpus_summary.csv")) == 22,
    )
    check(
        "ablation_results.csv has 11 rows",
        len(pd.read_csv("results/tables/ablation_results.csv")) == 11,
    )

    # ---------------- IMPORT GUARD ----------------
    print("\n[Stage 12] IMPORT GUARD")
    res = subprocess.run(
        [
            "grep",
            "-rE",
            r"from src\.analysis\.(significance|sensitivity)|import src\.analysis\.(significance|sensitivity)",
            "scripts/",
            "src/",
            "--include=*.py",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    check(
        "no stale imports of src.analysis.{significance,sensitivity}",
        res.stdout.strip() == "",
        res.stdout.strip() or "ok",
    )

    print("\n[Stage 12] All checks passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"\n[Stage 12] FAILED: {e}", file=sys.stderr)
        sys.exit(1)
