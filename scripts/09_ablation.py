"""
Stage 9 - Feature-family ablation on the best model.

Reads the best model + tuned hyperparams from stage 7b / 8 outputs,
runs the 11-row ablation (1 baseline + 5 families x 2 modes), and
writes ``results/tables/ablation_results.csv``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import TABLES_DIR  # noqa: E402
from src.analysis.ablation import run_family_ablation  # noqa: E402


def main() -> None:
    t0 = time.time()

    mc = pd.read_csv(TABLES_DIR / "model_comparison.csv")
    if mc["is_best"].sum() != 1:
        raise RuntimeError("model_comparison.csv must flag exactly one best model")
    best_model = mc[mc["is_best"]].iloc[0]["model"]
    print(f"[Stage 9] Best model: {best_model}")

    tuned = pd.read_csv(TABLES_DIR / "tuned_hyperparameters.csv")
    params_row = tuned[tuned["model"] == best_model]
    params = json.loads(params_row.iloc[0]["best_params_json"]) if len(params_row) else None
    print(f"[Stage 9] Params: {params}")

    print("[Stage 9] Running ablation (within-project 10-fold CV) ...")
    out = run_family_ablation(best_model, params)
    out_path = TABLES_DIR / "ablation_results.csv"
    out.to_csv(out_path, index=False)
    print(f"[Stage 9] Wrote {out_path}  ({len(out)} rows)")

    print("\n[Stage 9] Ablation table:")
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(out.round(4).to_string(index=False))

    # Quick interpretation: print the marginal F1 drop per family.
    base = out[out["mode"] == "all_features"]["f1"].iloc[0]
    leave = out[out["mode"] == "leave_out_family"][["family", "f1"]].copy()
    leave["f1_drop_vs_all"] = (base - leave["f1"]).round(4)
    print("\n[Stage 9] Marginal F1 drop when family is removed (sorted descending):")
    print(leave.sort_values("f1_drop_vs_all", ascending=False).to_string(index=False))

    print(f"\n[Stage 9] Total elapsed: {time.time() - t0:.1f}s")
    print("[Stage 9] Complete.")


if __name__ == "__main__":
    main()
