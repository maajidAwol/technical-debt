"""
Orchestrated scoring entry point with step-by-step progress reporting.

``run_scoring`` wraps the proven stage-13 flow (clone -> git features ->
score) behind a typed step enum and an optional ``progress_cb`` so callers
(the FastAPI backend, the stage-13 CLI) can render live progress. It also
adds the optional SonarCloud fetch that fills Families 1-2 when the target
project is already analyzed on https://sonarcloud.io.

Feature computation and scoring are unchanged from ``github_scorer``:
the same clone strategy (shallow at ``depth`` then unshallow so PyDriller
can diff every commit), the same ``_extract_git_features``, the same
``_score_dataframe``. Output is identical to ``scripts/13_score_github.py``.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import COST_EFFECTIVENESS_AT, PROJECT_ROOT  # noqa: E402
from src.inference.github_scorer import (  # noqa: E402
    _extract_git_features,
    _score_dataframe,
)


class Step(str, Enum):
    VALIDATING = "VALIDATING"
    CLONING = "CLONING"
    READING_HISTORY = "READING_HISTORY"
    BUILDING_FEATURES = "BUILDING_FEATURES"
    SONAR_FETCH = "SONAR_FETCH"
    SCORING = "SCORING"
    DONE = "DONE"


# progress_cb(step, detail, pct) — pct in [0, 100] or None when indeterminate.
ProgressCallback = Callable[[Step, str, Optional[float]], None]

GITHUB_URL_RE = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+$")

# The 10 SonarQube feature columns (Families 1-2) that default to zero and
# are overwritten when a SonarCloud fetch succeeds.
SONAR_FEATURE_COLS = [
    "ncloc", "complexity", "cognitive_complexity", "functions", "classes",
    "n_code_smells", "n_bugs", "total_debt_minutes", "issue_density",
    "duplicated_lines_density",
]

FEATURE_FAMILIES: dict[str, list[str]] = {
    "Size & complexity": ["ncloc", "complexity", "cognitive_complexity", "functions", "classes"],
    "Static debt": ["n_code_smells", "n_bugs", "total_debt_minutes", "issue_density", "duplicated_lines_density"],
    "Historical": [
        "total_commits_pre", "code_churn_pre", "recent_churn_90d", "commit_frequency_30d",
        "file_age_days", "days_since_last_change", "contributor_count", "ownership_ratio",
    ],
    "Co-change graph": ["cocg_degree", "cocg_pagerank", "cocg_betweenness", "cocg_entropy"],
    "Prior defect": ["bugfix_commits_pre", "bugfix_commits_90d", "bug_density_pre", "n_jira_bugs_pre", "jira_blocker_flag"],
}

REDUCED_ACCURACY_NOTICE = (
    "SonarQube features default to zero. Ranking is preserved; "
    "absolute scores are deflated (~80% of full-model accuracy)."
)


@dataclass
class ScoringResult:
    scored: pd.DataFrame          # basename, risk_score, predicted_high_risk (desc)
    features: pd.DataFrame        # one row per basename with all 27 feature columns
    summary: dict = field(default_factory=dict)


def _noop(step: Step, detail: str, pct: Optional[float]) -> None:  # pragma: no cover
    pass


def normalize_repo_url(repo_url: str) -> str:
    """Strip whitespace, a trailing slash, and a ``.git`` suffix."""
    url = repo_url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[: -len(".git")]
    return url


def _git_progress_run(
    cmd: list[str],
    emit: Callable[[str, Optional[float]], None],
    phase: str,
    timeout_s: float = 600.0,
) -> None:
    """Run a git command, streaming ``Receiving objects: NN%`` as progress."""
    pct_re = re.compile(r"(Receiving objects|Resolving deltas|Counting objects):\s+(\d+)%")
    proc = subprocess.Popen(
        cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    tail: list[str] = []
    deadline = time.time() + timeout_s
    buf = ""
    assert proc.stderr is not None
    try:
        while True:
            ch = proc.stderr.read(1)
            if ch == "" and proc.poll() is not None:
                break
            if time.time() > deadline:
                proc.kill()
                raise TimeoutError(f"git timed out after {timeout_s:.0f}s during: {phase}")
            if ch in ("\r", "\n"):
                line = buf.strip()
                buf = ""
                if not line:
                    continue
                tail.append(line)
                tail = tail[-5:]
                m = pct_re.search(line)
                if m and m.group(1) == "Receiving objects":
                    emit(f"{phase}… {line}", float(m.group(2)))
            else:
                buf += ch
    finally:
        if proc.poll() is None:
            proc.kill()
    if proc.returncode != 0:
        detail = "; ".join(tail[-3:]) or "no error output"
        raise RuntimeError(f"git failed during {phase.lower()} (exit {proc.returncode}): {detail}")


def _clone_repo_with_progress(
    github_url: str,
    clone_dir: Path,
    depth: int,
    emit: Callable[[str, Optional[float]], None],
) -> None:
    """Same strategy as ``github_scorer._clone_repo`` (shallow then unshallow),
    with live progress parsed from git's stderr."""
    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    if not clone_dir.exists():
        emit("Cloning repository… (partial clone, full history)", None)
        try:
            _git_progress_run(
                ["git", "clone", f"--depth={depth}", "--progress", github_url, str(clone_dir)],
                emit, phase="Cloning",
            )
        except BaseException:
            # A killed/failed clone can leave a partial directory that would
            # be mistaken for a valid cached clone on retry.
            shutil.rmtree(clone_dir, ignore_errors=True)
            raise
    if (clone_dir / ".git" / "shallow").exists():
        emit("Fetching full history (unshallow)…", None)
        _git_progress_run(
            ["git", "-C", str(clone_dir), "fetch", "--unshallow", "--progress"],
            emit, phase="Fetching history",
        )


def head_sha(clone_dir: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(clone_dir), "rev-parse", "HEAD"]
    ).decode().strip()


def ls_remote_head(github_url: str, timeout_s: float = 10.0) -> Optional[str]:
    """HEAD sha of a remote repo without cloning; None if unreachable."""
    try:
        out = subprocess.check_output(
            ["git", "ls-remote", github_url, "HEAD"],
            timeout=timeout_s, stderr=subprocess.DEVNULL,
        ).decode()
        return out.split()[0] if out.split() else None
    except (subprocess.SubprocessError, OSError):
        return None


def _merge_sonar_features(
    feats: pd.DataFrame,
    sonar: dict,
    emit: Callable[[str, Optional[float]], None],
) -> tuple[pd.DataFrame, bool]:
    """Overwrite the zeroed Families 1-2 columns with SonarCloud metrics.

    The project must already be analyzed on SonarCloud; this fetches
    existing measures (no scanner run). Matches rows on the true file
    basename. Returns (features, sonar_connected).
    """
    from src.inference.sonarcloud_client import fetch_file_metrics, to_feature_columns

    base_url = sonar.get("base_url", "https://sonarcloud.io")
    emit(f"Fetching SonarCloud metrics for {sonar['project_key']}…", None)
    raw = fetch_file_metrics(base_url, sonar["project_key"], sonar["token"], verbose=False)
    sf = to_feature_columns(raw, verbose=False)
    if sf.empty:
        emit("SonarCloud returned no file metrics; continuing with zeros.", None)
        return feats, False

    def true_basename(s: pd.Series) -> pd.Series:
        return s.str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1]

    sf = sf.assign(_bn=true_basename(sf["basename"]))
    sf = sf.drop_duplicates(subset="_bn", keep="first")
    merged = feats.drop(columns=SONAR_FEATURE_COLS).assign(_bn=true_basename(feats["basename"]))
    merged = merged.merge(sf[["_bn"] + SONAR_FEATURE_COLS], on="_bn", how="left")
    merged[SONAR_FEATURE_COLS] = merged[SONAR_FEATURE_COLS].fillna(0.0)
    n_matched = int((merged["ncloc"] > 0).sum())
    emit(f"SonarCloud metrics merged for {n_matched}/{len(merged)} files.", None)
    return merged.drop(columns="_bn"), True


def run_scoring(
    repo_url: str,
    repo_name: Optional[str] = None,
    *,
    depth: int = 500,
    sonar: Optional[dict] = None,
    progress_cb: Optional[ProgressCallback] = None,
    clone_root: Optional[Path] = None,
    cleanup: bool = False,
) -> ScoringResult:
    """Clone ``repo_url``, build the 27 features, and score every Java file.

    Parameters mirror ``github_scorer.score_github_repo``; additionally
    ``sonar`` is an optional ``{project_key, token, organization?}`` dict
    that fills Families 1-2 from an existing SonarCloud analysis, and
    ``progress_cb(step, detail, pct)`` receives live step updates.
    """
    t0 = time.time()
    cb = progress_cb or _noop

    cb(Step.VALIDATING, "Checking repository URL…", None)
    url = normalize_repo_url(repo_url)
    if not GITHUB_URL_RE.match(url):
        raise ValueError(
            f"Not a valid GitHub repository URL: {repo_url!r} "
            "(expected https://github.com/<org>/<repo>)"
        )
    repo_name = repo_name or url.rsplit("/", 1)[-1]

    clone_root = clone_root or (PROJECT_ROOT / "data" / "repos")
    clone_dir = clone_root / repo_name

    cb(Step.CLONING, "Cloning repository… (partial clone, full history)", None)
    try:
        _clone_repo_with_progress(url, clone_dir, depth, lambda d, p: cb(Step.CLONING, d, p))

        def feature_cb(stage: str, detail: str, pct: Optional[float]) -> None:
            step = Step.READING_HISTORY if stage == "history" else Step.BUILDING_FEATURES
            cb(step, detail, pct)

        cb(Step.READING_HISTORY, "Walking commit history…", None)
        feats = _extract_git_features(clone_dir, depth, verbose=False, progress_cb=feature_cb)
        if feats.empty:
            raise RuntimeError(
                "No Java files found in the requested commit window — "
                "the model is trained on Java projects only."
            )

        sonar_connected = False
        if sonar and sonar.get("project_key") and sonar.get("token"):
            cb(Step.SONAR_FETCH, "Fetching SonarCloud metrics…", None)
            try:
                feats, sonar_connected = _merge_sonar_features(
                    feats, sonar, lambda d, p: cb(Step.SONAR_FETCH, d, p)
                )
            except Exception as exc:  # reduced-accuracy fallback, not a job failure
                cb(Step.SONAR_FETCH, f"SonarCloud fetch failed ({exc}); continuing with zeros.", None)

        cb(Step.SCORING, f"Scoring {len(feats)} files with LightGBM…", None)
        scored = _score_dataframe(feats)
        sha = head_sha(clone_dir)
    finally:
        if cleanup:
            shutil.rmtree(clone_dir, ignore_errors=True)

    n = len(scored)
    n_high = int(scored["predicted_high_risk"].sum())
    k = max(1, int(round(COST_EFFECTIVENESS_AT * n)))
    summary = {
        "repo_url": url,
        "repo_name": repo_name,
        "head_sha": sha,
        "n_files": n,
        "n_high_risk": n_high,
        "review_budget_k": k,
        "review_budget_pct": int(COST_EFFECTIVENESS_AT * 100),
        "sonar_connected": sonar_connected,
        "elapsed_s": round(time.time() - t0, 1),
    }
    cb(Step.DONE, f"Report ready — {n} files, {n_high} predicted high-risk", 100.0)
    return ScoringResult(scored=scored, features=feats, summary=summary)
