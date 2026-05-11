"""
Sequential driver for the Technical Debt prediction pipeline.

Runs every numbered script under ``scripts/`` in order, tee-ing each
stage's stdout/stderr into ``results/run_logs/<stage>.log``. The script
streams output to the console live (unbuffered) so progress is visible
while a long stage is running.

Usage
-----
Run everything end-to-end::

    python run_pipeline.py

Skip stages that are already cached::

    python run_pipeline.py --skip 1,2,3,4

Start from a particular stage::

    python run_pipeline.py --from 7

Run only a specific subset::

    python run_pipeline.py --only 7b,7c,7d

List the stages without running them::

    python run_pipeline.py --list

Stop on first failure (default) or keep going::

    python run_pipeline.py --continue-on-error

By default every stage runs. Stages 1-4 only need to run when the raw
``td_V2.db`` changes; their outputs (cleaned parquets, snapshots,
labels) are otherwise deterministic. Pass ``--from 5`` (or
``--skip 1,2,3,4``) once those caches exist.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LOGS_DIR = PROJECT_ROOT / "results" / "run_logs"


@dataclass(frozen=True)
class Stage:
    key: str
    script: str
    name: str
    optional: bool = False


# Order matters. Keys are short tags used by --skip / --from / --only.
PIPELINE: tuple[Stage, ...] = (
    Stage("1",  "01_inspect_db.py",      "Inspect raw DB schema and counts"),
    Stage("2",  "02_profile_projects.py","Profile projects and select snapshots"),
    Stage("3",  "03_clean.py",           "Clean raw tables, drop tests/generated paths"),
    Stage("4",  "04_label.py",           "Build three label variants"),
    Stage("5",  "05_features.py",        "Engineer static, historical, graph, prior-defect features"),
    Stage("6",  "06_build_dataset.py",   "Join features + labels into modelling tables"),
    Stage("7",  "07_train.py",           "Within-project 10-fold CV, 6 models, persist preds, CIs, Wilcoxon"),
    Stage("7b", "07b_tune.py",           "Optuna hyperparameter tuning (tree ensembles)", optional=True),
    Stage("7c", "07c_calibrate.py",      "Probability calibration sweep (Platt, isotonic)", optional=True),
    Stage("7d", "07d_resample.py",       "SMOTE vs class_weight resampling sweep", optional=True),
    Stage("7e", "07e_temporal.py",       "Temporal within-project split T1 -> T2", optional=True),
    Stage("8",  "08_lopo.py",            "Leave-One-Project-Out cross-project CV"),
    Stage("9",  "09_sensitivity.py",     "Sensitivity grid, feature-family ablation, SHAP, permutation"),
    Stage("10", "10_report.py",          "Render figures and compose docs/06_results.md, 07_discussion.md"),
)


def _parse_keys(value: str) -> set[str]:
    return {k.strip() for k in value.split(",") if k.strip()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip", type=str, default="", help="Comma-separated stage keys to skip (e.g. 1,2,3,4).")
    p.add_argument("--from", dest="from_stage", type=str, default=None, help="Start from this stage key.")
    p.add_argument("--only", type=str, default=None, help="Comma-separated stage keys to run, ignoring others.")
    p.add_argument("--list", action="store_true", help="List stages and exit.")
    p.add_argument("--continue-on-error", action="store_true", help="Keep going after a stage fails.")
    p.add_argument("--python", type=str, default=None, help="Path to the Python interpreter to use (default: sys.executable).")
    return p.parse_args()


def _select_stages(args: argparse.Namespace) -> list[Stage]:
    if args.only:
        only = _parse_keys(args.only)
        return [s for s in PIPELINE if s.key in only]

    skip = _parse_keys(args.skip)
    from_idx = 0
    if args.from_stage:
        keys = [s.key for s in PIPELINE]
        if args.from_stage not in keys:
            raise SystemExit(f"--from: unknown stage key {args.from_stage!r}. Valid: {keys}")
        from_idx = keys.index(args.from_stage)
    return [s for i, s in enumerate(PIPELINE) if i >= from_idx and s.key not in skip]


def _print_header(label: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{label}\n{bar}", flush=True)


def _run_stage(stage: Stage, python: str) -> tuple[int, float]:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"stage{stage.key}.log"
    err_path = LOGS_DIR / f"stage{stage.key}.err"

    _print_header(f"[Stage {stage.key}] {stage.name}\n  script: scripts/{stage.script}\n  log:    {log_path.relative_to(PROJECT_ROOT)}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as fout, err_path.open("w", encoding="utf-8") as ferr:
        proc = subprocess.Popen(
            [python, "-u", str(SCRIPTS_DIR / stage.script)],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fout.write(line)
            fout.flush()
        rc = proc.wait()
    elapsed = time.time() - t0

    if err_path.stat().st_size == 0:
        err_path.unlink(missing_ok=True)

    return rc, elapsed


def main() -> int:
    args = parse_args()
    python = args.python or sys.executable

    stages = _select_stages(args)
    if not stages:
        print("No stages selected.", file=sys.stderr)
        return 1

    if args.list:
        print("Stages that will run:")
        for s in stages:
            tag = " (optional)" if s.optional else ""
            print(f"  [{s.key:>2}] {s.name}{tag}  -- scripts/{s.script}")
        return 0

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Pipeline runner using interpreter: {python}")
    print(f"Logs dir: {LOGS_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Will run {len(stages)} stage(s): {[s.key for s in stages]}")

    overall_t0 = time.time()
    failed: list[tuple[str, int]] = []
    for stage in stages:
        rc, elapsed = _run_stage(stage, python)
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"\n[Stage {stage.key}] {status}  elapsed={elapsed/60:.2f} min", flush=True)
        if rc != 0:
            failed.append((stage.key, rc))
            if not args.continue_on_error:
                break

    overall = time.time() - overall_t0
    _print_header(f"Pipeline finished. Total wall-clock: {overall/60:.2f} min")
    if failed:
        print(f"Failed stages: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
