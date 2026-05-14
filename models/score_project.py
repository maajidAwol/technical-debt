"""
score_project.py - Inference helper for the persisted best model.

Apply log1p to the same columns the training pipeline did, scale with
the persisted StandardScaler, then call predict_proba. A predicted
risk score >= the persisted threshold is flagged as high-risk TD.
"""
import joblib
import pandas as pd
import numpy as np


LOG1P_COLS = ['ncloc', 'complexity', 'total_commits_pre', 'code_churn_pre', 'recent_churn_90d', 'file_age_days', 'total_debt_minutes', 'bugfix_commits_pre', 'n_jira_bugs_pre']


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
