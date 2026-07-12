"""Regenerate real confusion matrices for LightGBM (within-project + LOPO).

Aggregates out-of-fold predictions and writes a small CSV that the thesis
tables can be re-derived from. Mirrors the existing 07/08 drivers but only
needs the tuned LightGBM and outputs raw (tn, fp, fn, tp) counts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CV_FOLDS, TABLES_DIR  # noqa: E402
from src.models.cross_project import (  # noqa: E402
    _build_sample_weights,
    _project_level_features,
    _similarity_weights,
)
from src.models.train import (  # noqa: E402
    CLASSIFICATION_THRESHOLD,
    _make_model,
    load_dataset,
)


SKIP_TEST_PROJECTS = {"org.apache:daemon"}


def _tuned_params(model_name: str) -> dict:
    df = pd.read_csv(TABLES_DIR / "tuned_hyperparameters.csv")
    row = df[df["model"] == model_name].iloc[0]
    return json.loads(row["best_params_json"])


def regen_within(X, y):
    yv = np.asarray(y, dtype=int)
    params = _tuned_params("lightgbm")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    oof_pred = np.zeros_like(yv)
    oof_true = np.zeros_like(yv)
    for tr, te in skf.split(X, yv):
        est = _make_model("lightgbm", params)
        est.fit(X.iloc[tr], yv[tr])
        prob = est.predict_proba(X.iloc[te])[:, 1]
        oof_pred[te] = (prob >= CLASSIFICATION_THRESHOLD).astype(int)
        oof_true[te] = yv[te]

    tn, fp, fn, tp = confusion_matrix(oof_true, oof_pred, labels=[0, 1]).ravel()
    f1 = f1_score(oof_true, oof_pred)
    return {"mode": "within_project", "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp), "f1": float(f1),
            "n": int(len(oof_true))}


def regen_lopo(X, y, proj):
    params = _tuned_params("lightgbm")
    proj_feats = _project_level_features(X, y, proj)
    projects = sorted(proj.unique())

    all_true, all_pred = [], []
    for test_pid in projects:
        if test_pid in SKIP_TEST_PROJECTS:
            continue
        tr_mask = proj.values != test_pid
        te_mask = proj.values == test_pid
        X_tr = X.iloc[tr_mask].reset_index(drop=True)
        y_tr = np.asarray(y.iloc[tr_mask], dtype=int)
        proj_tr = proj.iloc[tr_mask].reset_index(drop=True)
        X_te = X.iloc[te_mask].reset_index(drop=True)
        y_te = np.asarray(y.iloc[te_mask], dtype=int)

        test_feats = (proj_feats.loc[test_pid].values.astype(float)
                      if test_pid in proj_feats.index
                      else proj_feats.mean().values.astype(float))
        train_pids = sorted(proj_tr.unique())
        w_by_proj = _similarity_weights(proj_feats.loc[train_pids], test_feats)
        sw = _build_sample_weights(proj_tr, w_by_proj)

        est = _make_model("lightgbm", params, y_train=y_tr)
        est.fit(X_tr, y_tr, sample_weight=sw)
        prob = est.predict_proba(X_te)[:, 1]
        pred = (prob >= CLASSIFICATION_THRESHOLD).astype(int)
        all_true.append(y_te)
        all_pred.append(pred)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    f1 = f1_score(y_true, y_pred)
    return {"mode": "lopo", "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp), "f1": float(f1),
            "n": int(len(y_true))}


def main():
    print("Loading dataset ...")
    X, y, proj = load_dataset()
    print(f"  rows={len(X):,}  pos={int(y.sum()):,}  pos_rate={float(y.mean()):.4f}")

    print("\n[1/2] Within-project 10-fold LightGBM (tuned) ...")
    within = regen_within(X, y)
    print(f"  TN={within['tn']:,}  FP={within['fp']:,}  "
          f"FN={within['fn']:,}  TP={within['tp']:,}  F1={within['f1']:.4f}")

    print("\n[2/2] LOPO LightGBM (tuned, similarity-weighted) ...")
    lopo = regen_lopo(X, y, proj)
    print(f"  TN={lopo['tn']:,}  FP={lopo['fp']:,}  "
          f"FN={lopo['fn']:,}  TP={lopo['tp']:,}  F1={lopo['f1']:.4f}")

    out = pd.DataFrame([within, lopo])
    out_path = TABLES_DIR / "confusion_matrices.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
