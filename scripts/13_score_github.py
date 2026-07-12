"""
Stage 13 - Score any Apache Java GitHub repository with the persisted model.

This is an *optional* stage. It is not part of the canonical end-to-end run
(it needs a network call and a user-supplied URL); invoke it explicitly:

    python scripts/13_score_github.py \\
        --url https://github.com/apache/commons-cli \\
        --name commons-cli

Outputs
-------
- ``results/tables/risk_report_<name>.csv``: per-file ranked risk scores
  (basename, risk_score, predicted_high_risk).
- Console: top-20 files, summary counts, and the top-20% review-budget line.

Reduced-accuracy notice
-----------------------
SonarQube features (Families 1 & 2) default to zero when scoring an
arbitrary GitHub repo. The model still runs on the remaining 14 git-based
features. The ranking signal is preserved; absolute scores are deflated.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import TABLES_DIR  # noqa: E402
from src.inference.run_scoring import Step, run_scoring  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="GitHub repository URL (https://github.com/<org>/<repo>)")
    p.add_argument("--name", required=True, help="Short repo name; used for output filename and clone directory")
    p.add_argument("--depth", type=int, default=500, help="Number of most-recent commits to traverse (default: 500)")
    p.add_argument("--cleanup", action="store_true", help="Delete the clone directory after scoring")
    p.add_argument("--out-dir", type=Path, default=TABLES_DIR, help="Where to write risk_report_<name>.csv (default: results/tables)")
    return p.parse_args()


def _print_progress(step: Step, detail: str, pct) -> None:
    tag = f"[Stage 13] [{step.value}]"
    if pct is not None and step in (Step.CLONING, Step.READING_HISTORY):
        print(f"{tag} {detail} ({pct:.0f}%)", flush=True)
    else:
        print(f"{tag} {detail}", flush=True)


def main() -> None:
    args = _parse_args()
    t0 = time.time()
    print(f"[Stage 13] URL          : {args.url}")
    print(f"[Stage 13] Name         : {args.name}")
    print(f"[Stage 13] Clone depth  : {args.depth}")

    result = run_scoring(
        args.url,
        repo_name=args.name,
        depth=args.depth,
        cleanup=args.cleanup,
        progress_cb=_print_progress,
    )
    scored = result.scored

    n = result.summary["n_files"]
    k = result.summary["review_budget_k"]
    n_high = result.summary["n_high_risk"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"risk_report_{args.name}.csv"
    scored.to_csv(out_path, index=False)
    print(f"\n[Stage 13] Wrote {out_path}  ({n} files)")

    print("\n[Stage 13] Top 20 highest-risk files:")
    with pd.option_context("display.width", 160, "display.max_rows", None):
        print(scored.head(20).to_string(index=False))

    print(f"\n[Stage 13] Total Java files analyzed    : {n}")
    print(f"[Stage 13] Files in top {result.summary['review_budget_pct']}% review budget: {k}")
    print(f"[Stage 13] High-risk files predicted    : {n_high}")
    print(f"\n[Stage 13] Note: SonarQube features default to zero. Ranking is preserved;")
    print(f"           absolute scores are deflated (~80% of full-model accuracy).")
    print(f"\n[Stage 13] Elapsed: {time.time() - t0:.1f}s")
    print("[Stage 13] Complete.")


if __name__ == "__main__":
    main()
