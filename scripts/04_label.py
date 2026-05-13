# =================================================
# COMBINED-WEIGHT HIGH-RISK TD LABEL
# =================================================
# Binary output: is_high_risk = 1 or 0
#
# Six binary signals from two independent sources:
#   Static (SonarQube at t):   S1, S2, S3
#   History (Git before t):    S4, S5, S6
#
# Weights derived empirically via point-biserial
# correlation with post-snapshot bug-fix activity
# (Kamei et al. TSE 2013 weighting approach).
# Confirmed against theoretically motivated baseline.
#
# Label is binary 1/0 - not probability, not score.
# risk_score is internal computation only.
# =================================================
"""
Stage 4 - Compute the single dual-signal binary label.

Outputs
-------
- ``data/processed/labels.parquet``
    (project_id, basename, is_high_risk, risk_score, S1..S6)
- ``data/processed/label_statistics.csv``
    (project_id, n_files, n_positive, positive_rate_pct,
     n_s1_only, n_s4_only, n_both_static_history, threshold_used)
- ``data/processed/derived_weights.json``
- ``data/processed/weight_comparison.csv``
- ``results/tables/label_summary.csv``
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    LABEL_SURROGATE_WINDOW_MONTHS,
    PROCESSED_DATA_DIR,
    TABLES_DIR,
    THEORETICAL_WEIGHTS,
)
from src.data.labeling import (  # noqa: E402
    SIGNAL_COLUMNS,
    choose_weights,
    compute_dual_signal_labels,
    compute_dual_signal_signals,
    derive_weights_empirically,
)
from src.data.snapshot import load_snapshots  # noqa: E402


# SNAPSHOT: t = median commit date per project
# Rationale: ensures equal proportional history
# across all 22 projects for LOPO comparability.
# Follows Tsoukalas 2020 and Jiang 2024 convention.
# 6-month label window follows Kamei et al. 2013.
# Data after t+6 months discarded: beyond this
# window, activity reflects new debt after t.


def _load_clean() -> dict[str, pd.DataFrame]:
    commits = pd.read_parquet(PROCESSED_DATA_DIR / "clean_git_commits.parquet")
    changes = pd.read_parquet(PROCESSED_DATA_DIR / "clean_git_commits_changes.parquet")
    sonar_issues = pd.read_parquet(PROCESSED_DATA_DIR / "clean_sonar_issues.parquet")
    for df, cols in (
        (commits, ["AUTHOR_DATE", "COMMITTER_DATE"]),
        (changes, ["DATE"]),
        (sonar_issues, ["CREATION_DATE", "CLOSE_DATE"]),
    ):
        for col in cols:
            if col in df.columns and df[col].dtype.kind == "M" and df[col].dt.tz is None:
                df[col] = df[col].dt.tz_localize("UTC")
    return {"commits": commits, "changes": changes, "sonar_issues": sonar_issues}


def main() -> None:
    t0 = time.time()
    print(f"[Stage 4] Surrogate window     : {LABEL_SURROGATE_WINDOW_MONTHS} months")
    print(f"[Stage 4] Theoretical baseline : {THEORETICAL_WEIGHTS}")

    snaps = load_snapshots(PROCESSED_DATA_DIR / "project_snapshots.parquet")
    eligible = snaps[snaps["eligible"]].copy()
    print(f"[Stage 4] Eligible projects    : {len(eligible)}")

    print("[Stage 4] Loading cleaned parquets ...")
    data = _load_clean()
    print(f"   commits      : {len(data['commits']):>10,}")
    print(f"   changes      : {len(data['changes']):>10,}")
    print(f"   sonar_issues : {len(data['sonar_issues']):>10,}")

    # --- Build per-(project, snapshot) signals (incl. surrogate) ----------
    print("\n[Stage 4] Computing per-(project, snapshot) signals ...")
    parts: list[pd.DataFrame] = []
    for _, row in eligible.sort_values(["project_id", "snapshot_id"]).iterrows():
        pid = row["project_id"]
        sid = row["snapshot_id"]
        t = row["snapshot_date"]
        sub = compute_dual_signal_signals(
            pid, t, data["commits"], data["changes"], data["sonar_issues"]
        )
        # Tag rows with the snapshot identifier; project_id is set by the
        # labeling module already.
        sub["snapshot_id"] = sid
        parts.append(sub)
        print(
            f"   {pid:<35} [{sid}]  N={len(sub):>6}  "
            f"S1={int(sub['S1_severity'].sum()):>4}  "
            f"S4={int(sub['S4_bugfix'].sum()):>4}  "
            f"surrogate={int(sub['future_bugfix_count'].sum()):>5}"
        )
    signals_all = pd.concat(parts, ignore_index=True)

    # --- Empirical weight derivation --------------------------------------
    print("\n[Stage 4] Deriving empirical weights via point-biserial ...")
    derived = derive_weights_empirically(signals_all, signals_all[["future_bugfix_count"]])
    chosen_weights, source = choose_weights(derived, THEORETICAL_WEIGHTS, tolerance=0.05)
    if source == "theoretical":
        print("[Stage 4] Empirical weights confirmed theoretical baseline (max diff <= 0.05).")
    else:
        print("[Stage 4] Empirical weights diverge from theoretical (max diff > 0.05); using empirical.")

    print("[Stage 4] Weights (in use):")
    for s in SIGNAL_COLUMNS:
        print(f"   {s:<20} theoretical={THEORETICAL_WEIGHTS[s]:.2f}  derived={derived[s]:.2f}  chosen={chosen_weights[s]:.2f}")

    weights_path = PROCESSED_DATA_DIR / "derived_weights.json"
    with weights_path.open("w", encoding="utf-8") as fh:
        json.dump({**chosen_weights, "_source": source}, fh, indent=2)

    comparison = pd.DataFrame(
        {
            "signal": list(SIGNAL_COLUMNS),
            "theoretical": [THEORETICAL_WEIGHTS[s] for s in SIGNAL_COLUMNS],
            "derived": [derived[s] for s in SIGNAL_COLUMNS],
            "chosen": [chosen_weights[s] for s in SIGNAL_COLUMNS],
        }
    )
    comparison["difference"] = (comparison["derived"] - comparison["theoretical"]).round(3)

    def _interpret(diff: float) -> str:
        # Magnitude bands chosen so anything beyond 0.05 (the
        # theoretical->empirical fallback trigger) gets called out
        # explicitly, with sign telling the reader whether the surrogate
        # placed more or less weight on the signal than literature did.
        ad = abs(diff)
        if ad <= 0.02:
            return "matches theoretical baseline (no meaningful divergence)"
        if diff < 0:
            kind = "weaker"
        else:
            kind = "stronger"
        if ad <= 0.05:
            magnitude = "modestly"
        elif ad <= 0.10:
            magnitude = "noticeably"
        else:
            magnitude = "substantially"
        return f"{magnitude} {kind} empirical correlation with post-snapshot bug-fix activity than Kamei 2013 baseline"

    comparison["interpretation"] = comparison["difference"].apply(_interpret)
    comparison.to_csv(PROCESSED_DATA_DIR / "weight_comparison.csv", index=False)

    # --- Compute labels using chosen weights -------------------------------
    print("\n[Stage 4] Scoring + thresholding per (project, snapshot) ...")
    # Drop surrogate before label scoring so it never leaks into the saved frame.
    signals_only = signals_all.drop(columns=["future_bugfix_count"])
    labeled = compute_dual_signal_labels(
        signals_only, chosen_weights, group_by=("project_id", "snapshot_id")
    )
    assert "future_bugfix_count" not in labeled.columns, "surrogate leaked into labels"
    assert set(labeled["is_high_risk"].unique()) <= {0, 1}

    # --- Per-project label_statistics --------------------------------------
    static_flag = (labeled[["S1_severity", "S2_debt", "S3_smells"]].sum(axis=1) >= 1).astype("int64")
    history_flag = (labeled[["S4_bugfix", "S5_churn", "S6_contributors"]].sum(axis=1) >= 1).astype("int64")
    labeled["_static_any"] = static_flag
    labeled["_history_any"] = history_flag

    stats_rows = []
    for (pid, sid), sub in labeled.groupby(["project_id", "snapshot_id"]):
        s1_only = int(((sub["S1_severity"] == 1) & (sub["_history_any"] == 0)).sum())
        s4_only = int(((sub["S4_bugfix"] == 1) & (sub["_static_any"] == 0)).sum())
        both = int(((sub["_static_any"] == 1) & (sub["_history_any"] == 1) & (sub["is_high_risk"] == 1)).sum())
        thr = float(sub["threshold_used"].iloc[0])
        stats_rows.append(
            {
                "project_id": pid,
                "snapshot_id": sid,
                "n_files": int(len(sub)),
                "n_positive": int(sub["is_high_risk"].sum()),
                "positive_rate_pct": round(100 * float(sub["is_high_risk"].mean()), 2),
                "n_s1_only": s1_only,
                "n_s4_only": s4_only,
                "n_both_static_history": both,
                "threshold_used": thr,
                "min_positives_satisfied": int(sub["min_positives_satisfied"].iloc[0]),
            }
        )
    stats = pd.DataFrame(stats_rows).sort_values(["project_id", "snapshot_id"]).reset_index(drop=True)
    stats.to_csv(PROCESSED_DATA_DIR / "label_statistics.csv", index=False)

    # Concise per-(project, snapshot) summary for the thesis tables/ folder.
    label_summary = stats[
        ["project_id", "snapshot_id", "n_files", "n_positive", "positive_rate_pct", "threshold_used"]
    ].copy()
    label_summary.to_csv(TABLES_DIR / "label_summary.csv", index=False)

    # --- Persist labels (drop helper cols) ---------------------------------
    keep_cols = [
        "project_id",
        "snapshot_id",
        "basename",
        "is_high_risk",
        "risk_score",
        *SIGNAL_COLUMNS,
    ]
    labels_to_save = labeled[keep_cols].copy()
    labels_to_save.to_parquet(PROCESSED_DATA_DIR / "labels.parquet", index=False)

    # --- Stage summary -----------------------------------------------------
    print("\n[Stage 4] Per-(project, snapshot) label statistics:")
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(stats.to_string(index=False))

    overall_rate = float(labeled["is_high_risk"].mean()) * 100
    print(f"\n[Stage 4] Overall positive rate : {overall_rate:.2f}%  "
          f"({int(labeled['is_high_risk'].sum()):,} of {len(labeled):,})")

    low_pos = stats[stats["min_positives_satisfied"] == 0]
    if len(low_pos) > 0:
        print(f"[Stage 4] WARNING: {len(low_pos)} (project, snapshot) rows have < 5 positives even at lowest threshold:")
        print(low_pos[["project_id", "n_positive", "threshold_used"]].to_string(index=False))

    print(f"\n[Stage 4] Total elapsed : {time.time() - t0:.1f}s")
    print("[Stage 4] Complete.")


if __name__ == "__main__":
    main()
