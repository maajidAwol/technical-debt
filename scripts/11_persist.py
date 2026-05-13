"""
Stage 11 - Persist the best model + supporting artefacts.

Reads:
- ``results/tables/model_comparison.csv`` to identify the best model
  (where ``is_best == True``).
- ``results/tables/tuned_hyperparameters.csv`` for tuned hyperparams.
- ``data/processed/dataset_final.parquet`` to refit on the full corpus.

Refits the best model on ALL rows, fits a StandardScaler on the same
matrix (for downstream inference symmetry), and writes:

  models/best_model.pkl
  models/feature_scaler.pkl
  models/optimal_threshold.txt   (always "0.5" - see train.py rationale)
  models/feature_names.csv
  models/model_card.json
  models/score_project.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import ALL_FEATURES, LOG1P_FEATURES, MODELS_DIR, TABLES_DIR  # noqa: E402
from src.models.train import CLASSIFICATION_THRESHOLD, _make_model, load_dataset  # noqa: E402


SCORING_FN_TEMPLATE = '''"""
score_project.py - Inference helper for the persisted best model.

Apply log1p to the same columns the training pipeline did, scale with
the persisted StandardScaler, then call predict_proba. A predicted
risk score >= the persisted threshold is flagged as high-risk TD.
"""
import joblib
import pandas as pd
import numpy as np


LOG1P_COLS = __LOG1P_COLS_PLACEHOLDER__


def score_new_project(features_df: pd.DataFrame) -> pd.DataFrame:
    """Score files in a new Apache Java project.

    Input: a DataFrame with the 27 feature columns produced by stage 5
    (in any order; the model's feature_names.csv defines the canonical
    ordering). Output: original rows annotated with ``risk_score`` and
    ``predicted_high_risk``, sorted by descending risk.
    """
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/feature_scaler.pkl")
    threshold = float(open("models/optimal_threshold.txt").read())
    feat_names = pd.read_csv("models/feature_names.csv")["feature"].tolist()

    X = features_df[feat_names].fillna(0).astype(float).copy()
    for col in LOG1P_COLS:
        if col in X.columns:
            X[col] = np.log1p(X[col])

    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]

    result = features_df.copy()
    result["risk_score"] = probs
    result["predicted_high_risk"] = (probs >= threshold).astype(int)
    return result.sort_values("risk_score", ascending=False)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python models/score_project.py <features.parquet>")
        sys.exit(1)
    df_in = pd.read_parquet(sys.argv[1])
    scored = score_new_project(df_in)
    out_path = sys.argv[1].replace(".parquet", "_scored.csv")
    scored.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
'''


def main() -> None:
    t0 = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Identify best model
    cmp_df = pd.read_csv(TABLES_DIR / "model_comparison.csv")
    if cmp_df["is_best"].sum() != 1:
        raise RuntimeError(
            f"model_comparison.csv must flag exactly one best model, got {cmp_df['is_best'].sum()}"
        )
    best_row = cmp_df[cmp_df["is_best"]].iloc[0]
    best_model_name = best_row["model"]
    print(f"[Stage 11] Best model: {best_model_name}")

    # Load tuned hyperparams for the best model
    tuned = pd.read_csv(TABLES_DIR / "tuned_hyperparameters.csv")
    params_row = tuned[tuned["model"] == best_model_name]
    params = json.loads(params_row.iloc[0]["best_params_json"]) if len(params_row) else {}
    print(f"[Stage 11] Tuned params: {params}")

    # Load full corpus + refit
    X, y, _, _ = load_dataset()
    print(f"[Stage 11] Refitting on full corpus: rows={len(X):,} feats={X.shape[1]}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    # For LR (pipeline) we pass through the same _make_model factory but
    # then refit on the scaled matrix to keep the persisted model
    # consistent with the persisted scaler. Tree ensembles ignore the
    # scaling anyway; we still fit on the same scaled X for symmetry.
    est = _make_model(best_model_name, params, y_train=y.values)
    if est is None:
        raise RuntimeError(f"Model {best_model_name!r} unavailable")
    # If the estimator is a Pipeline that already has an internal scaler,
    # we drop it and fit the bare classifier on the externally scaled X.
    if hasattr(est, "named_steps") and "scaler" in getattr(est, "named_steps", {}):
        clf = est.named_steps["clf"]
        clf.fit(X_scaled, y.values)
        final_model = clf
    else:
        est.fit(X_scaled, y.values)
        final_model = est

    # Persist artefacts
    joblib.dump(final_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "feature_scaler.pkl")
    with (MODELS_DIR / "optimal_threshold.txt").open("w", encoding="utf-8") as fh:
        fh.write(f"{CLASSIFICATION_THRESHOLD}")
    pd.DataFrame({"feature": list(ALL_FEATURES)}).to_csv(
        MODELS_DIR / "feature_names.csv", index=False
    )

    # Model card
    card = {
        "model_type": type(final_model).__name__,
        "model_name": best_model_name,
        "n_features": len(ALL_FEATURES),
        "feature_families": 5,
        "label": "dual_signal_combined_weight",
        "threshold": float(CLASSIFICATION_THRESHOLD),
        "lopo_f1": round(float(best_row["lopo_f1"]), 4),
        "lopo_roc_auc": round(float(best_row.get("lopo_roc_auc", float("nan"))), 4),
        "lopo_pr_auc": round(float(best_row["lopo_pr_auc"]), 4),
        "lopo_ce_at_20": round(float(best_row["lopo_ce_at_20"]), 4),
        "within_f1": round(float(best_row["within_f1"]), 4),
        "within_pr_auc": round(float(best_row["within_pr_auc"]), 4),
        "training_projects": 22,
        "training_rows": int(len(X)),
        "tuned_params": params,
        "date_trained": str(pd.Timestamp.now().date()),
    }
    with (MODELS_DIR / "model_card.json").open("w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2)

    # Scoring helper - substitute the log1p column list as a literal
    scoring_src = SCORING_FN_TEMPLATE.replace(
        "__LOG1P_COLS_PLACEHOLDER__", repr(list(LOG1P_FEATURES))
    )
    with (MODELS_DIR / "score_project.py").open("w", encoding="utf-8") as fh:
        fh.write(scoring_src)

    print("\n[Stage 11] Model card:")
    print(json.dumps(card, indent=2))

    print(f"\n[Stage 11] Total elapsed: {time.time() - t0:.1f}s")
    print("[Stage 11] Complete.")


if __name__ == "__main__":
    main()
