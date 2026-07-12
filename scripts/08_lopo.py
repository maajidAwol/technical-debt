"""
Stage 8 - Leave-One-Project-Out validation + best-model selection.

For each of the 4 models, run LOPO with similarity-weighted training
(see src/models/cross_project.py). Threshold optimisation is applied
to the (weighted) training set; the resulting threshold scores the
held-out test project.

ITEM 1 (May 13 enhancements): org.apache:daemon is excluded from
LOPO TEST. It still appears in training folds for the other 21
projects, but is never held out (only 4 positives total, the
fallback threshold could not satisfy min 5 positives). Rows for
daemon carry ``note='training_only'`` in lopo_per_project.csv and
are excluded from the model-level aggregate.

After all four models finish, write ``model_comparison.csv`` and log
the best model (highest LOPO F1_optimized).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*convergence.*")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import MODEL_ORDER, TABLES_DIR  # noqa: E402
from src.models.cross_project import (  # noqa: E402
    lopo_cv,
    lopo_results_to_frame,
    lopo_summary,
)


SKIP_TEST_PROJECTS: tuple[str, ...] = ("org.apache:daemon",)


def _load_tuned_params() -> dict[str, dict]:
    path = TABLES_DIR / "tuned_hyperparameters.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/07b_tune.py before Stage 8."
        )
    df = pd.read_csv(path)
    return {row["model"]: json.loads(row["best_params_json"]) for _, row in df.iterrows()}


def main() -> None:
    t0 = time.time()
    print(f"[Stage 8] Models       : {list(MODEL_ORDER)}")
    print(f"[Stage 8] Skip-test    : {list(SKIP_TEST_PROJECTS)}")

    tuned_params = _load_tuned_params()
    all_rows: list[pd.DataFrame] = []
    for model_name in MODEL_ORDER:
        t1 = time.time()
        params = tuned_params.get(model_name)
        print(f"\n[Stage 8] LOPO {model_name} (params loaded={params is not None}) ...")
        try:
            results = lopo_cv(
                model_name,
                params=params,
                skip_test_projects=SKIP_TEST_PROJECTS,
            )
        except Exception as e:
            print(f"   FAILED: {type(e).__name__}: {e}")
            continue
        if not results:
            print(f"   SKIPPED (estimator unavailable)")
            continue
        df = lopo_results_to_frame(results)
        eligible = df[df["note"] != "training_only"]
        print(
            f"   scored={len(eligible)} of {len(df)} projects  "
            f"mean_F1={eligible['f1'].mean():.3f}  "
            f"mean_ROC={eligible['roc_auc'].mean():.3f}  "
            f"mean_PR={eligible['pr_auc'].mean():.3f}  "
            f"mean_CE@20={eligible['ce_at_20'].mean():.3f}  "
            f"({time.time() - t1:.1f}s)"
        )
        all_rows.append(df)

    per_project = pd.concat(all_rows, ignore_index=True)
    per_project_path = TABLES_DIR / "lopo_per_project.csv"
    per_project.to_csv(per_project_path, index=False)
    print(f"\n[Stage 8] Wrote {per_project_path}  ({len(per_project)} rows)")

    summary = lopo_summary(per_project)
    summary_path = TABLES_DIR / "lopo_results.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[Stage 8] Wrote {summary_path}  ({len(summary)} rows)")

    # ----- Best-model selection -----
    print("\n[Stage 8] Building model_comparison.csv ...")
    within = pd.read_csv(TABLES_DIR / "within_project_results.csv")
    within_agg = (
        within.groupby("model")[["f1", "pr_auc"]]
        .mean()
        .rename(columns={"f1": "within_f1", "pr_auc": "within_pr_auc"})
    )
    lopo_agg = summary.set_index("model")[
        ["f1_mean", "pr_auc_mean", "ce_at_20_mean", "roc_auc_mean"]
    ].rename(
        columns={
            "f1_mean": "lopo_f1",
            "pr_auc_mean": "lopo_pr_auc",
            "ce_at_20_mean": "lopo_ce_at_20",
            "roc_auc_mean": "lopo_roc_auc",
        }
    )
    cmp_df = within_agg.join(lopo_agg, how="outer").reset_index()
    cmp_df["is_best"] = False
    if cmp_df["lopo_f1"].notna().any():
        best_idx = cmp_df["lopo_f1"].idxmax()
        cmp_df.loc[best_idx, "is_best"] = True
        best_model = cmp_df.loc[best_idx, "model"]
        best_f1 = float(cmp_df.loc[best_idx, "lopo_f1"])
        print(f"[Stage 8] Best model: {best_model}, LOPO F1 = {best_f1:.4f}")

    cmp_path = TABLES_DIR / "model_comparison.csv"
    cmp_df.to_csv(cmp_path, index=False)
    print(f"[Stage 8] Wrote {cmp_path}")

    print("\n[Stage 8] Summary table:")
    with pd.option_context("display.width", 220, "display.max_rows", None):
        print(cmp_df.round(4).to_string(index=False))

    print(f"\n[Stage 8] Total elapsed: {time.time() - t0:.1f}s")
    print("[Stage 8] Complete.")


if __name__ == "__main__":
    main()
