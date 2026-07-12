"""
Score the Java files of any GitHub repository using the persisted best model.

The training pipeline produces a 27-feature LightGBM classifier across 5
feature families (size/complexity, static debt, historical, co-change graph,
prior defect). For an arbitrary GitHub clone we can compute Families 3-5
from git history alone; Families 1-2 (SonarQube) default to zero. The
model still runs, with somewhat reduced accuracy (~80% of the full
pipeline) — the ranking signal is preserved even if absolute scores are
deflated.

Usage
-----
.. code-block:: python

    from src.inference.github_scorer import score_github_repo
    scored = score_github_repo(
        github_url="https://github.com/apache/commons-cli",
        repo_name="commons-cli",
        clone_depth=500,
    )
    scored.head(20)

The function returns a DataFrame sorted by descending ``risk_score`` with
columns ``basename``, ``risk_score``, ``predicted_high_risk``.
"""
from __future__ import annotations

import math
import re
import shutil
import subprocess
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Optional

import igraph as ig
import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    ALL_FEATURES,
    BUGFIX_REGEX,
    LOG1P_FEATURES,
    MODELS_DIR,
    PATH_EXCLUSION_PATTERNS,
    PROJECT_ROOT,
)


_BUGFIX_RE = re.compile(BUGFIX_REGEX, re.IGNORECASE)


# ---------------------------------------------------------------------------
# Clone helpers
# ---------------------------------------------------------------------------
def _clone_repo(github_url: str, clone_dir: Path, depth: int) -> None:
    """Shallow-clone then unshallow so PyDriller can diff every commit.

    A shallow clone leaves a "grafted parent" boundary at the oldest
    commit; ``git diff-tree`` against that fake parent crashes with exit
    128, breaking PyDriller's ``commit.modified_files`` access. The
    initial ``--depth`` keeps the first fetch fast, then we run
    ``git fetch --unshallow`` to fill in the rest of the history.
    """
    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    if not clone_dir.exists():
        subprocess.run(
            ["git", "clone", f"--depth={depth}", github_url, str(clone_dir)],
            check=True,
        )
    is_shallow = (clone_dir / ".git" / "shallow").exists()
    if is_shallow:
        subprocess.run(
            ["git", "-C", str(clone_dir), "fetch", "--unshallow"],
            check=True,
        )


def _target_hashes(clone_dir: Path, depth: int) -> set[str]:
    """Return the set of the most-recent ``depth`` commit hashes on HEAD."""
    out = subprocess.check_output(
        ["git", "-C", str(clone_dir), "rev-list", f"--max-count={depth}", "HEAD"]
    ).decode()
    return set(out.splitlines())


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _is_java_target(path: str) -> bool:
    if not path or not path.endswith(".java"):
        return False
    p = path.replace("\\", "/").lower()
    return not any(bad in p for bad in PATH_EXCLUSION_PATTERNS)


def _extract_git_features(
    clone_dir: Path,
    depth: int,
    *,
    verbose: bool = True,
    progress_cb=None,
) -> pd.DataFrame:
    """Compute Families 3, 4, 5 over the latest ``depth`` commits.

    Returns one row per Java basename with the 18 git-derived columns the
    training pipeline produced; Families 1-2 are added as zeros downstream.

    ``progress_cb(stage, detail, pct)`` (optional) receives live updates:
    ``stage`` is ``"history"`` during commit traversal and ``"features"``
    during feature assembly; ``pct`` is 0-100 or None.
    """
    from pydriller import Repository  # local import: pydriller startup is slow

    emit = progress_cb or (lambda stage, detail, pct: None)
    target = _target_hashes(clone_dir, depth)
    if verbose:
        print(f"[GitHub-Score] Traversing latest {len(target)} commits ...", flush=True)
    emit("history", f"Walking commit history… (0/{len(target)} commits processed)", 0.0)

    file_records: dict[str, dict] = {}
    authors_per_file: dict[str, Counter] = {}
    cochange_edges: Counter = Counter()
    all_basenames: set[str] = set()

    t_now = pd.Timestamp.utcnow()
    t_30d = t_now - pd.Timedelta(days=30)
    t_90d = t_now - pd.Timedelta(days=90)

    n_seen = 0
    n_skipped = 0
    for commit in Repository(str(clone_dir)).traverse_commits():
        if commit.hash not in target:
            continue
        n_seen += 1
        if n_seen % 25 == 0:
            emit(
                "history",
                f"Walking commit history… ({n_seen}/{len(target)} commits processed)",
                100.0 * n_seen / max(1, len(target)),
            )
        if commit.author_date.tzinfo is None:
            commit_date = pd.Timestamp(commit.author_date, tz="UTC")
        else:
            commit_date = pd.Timestamp(commit.author_date).tz_convert("UTC")
        msg = commit.msg or ""
        is_bugfix = bool(_BUGFIX_RE.search(msg))
        author = commit.author.name if commit.author else "unknown"
        try:
            modified_files = commit.modified_files
        except Exception:
            n_skipped += 1
            continue

        touched_in_commit: list[str] = []
        for mf in modified_files:
            path = mf.new_path or mf.old_path or ""
            if not _is_java_target(path):
                continue
            bn = path.rsplit("/", 1)[-1]
            added = mf.added_lines or 0
            removed = mf.deleted_lines or 0
            rec = file_records.setdefault(
                bn,
                {
                    "total_commits_pre": 0,
                    "code_churn_pre": 0,
                    "recent_churn_90d": 0,
                    "commit_frequency_30d": 0,
                    "first_commit": commit_date,
                    "last_commit": commit_date,
                    "bugfix_commits_pre": 0,
                    "bugfix_commits_90d": 0,
                },
            )
            rec["total_commits_pre"] += 1
            rec["code_churn_pre"] += added + removed
            if commit_date >= t_90d:
                rec["recent_churn_90d"] += added + removed
                if is_bugfix:
                    rec["bugfix_commits_90d"] += 1
            if commit_date >= t_30d:
                rec["commit_frequency_30d"] += 1
            if commit_date < rec["first_commit"]:
                rec["first_commit"] = commit_date
            if commit_date > rec["last_commit"]:
                rec["last_commit"] = commit_date
            if is_bugfix:
                rec["bugfix_commits_pre"] += 1
            authors_per_file.setdefault(bn, Counter())[author] += 1
            touched_in_commit.append(bn)
            all_basenames.add(bn)
        if len(touched_in_commit) > 1:
            for a, b in combinations(sorted(set(touched_in_commit)), 2):
                cochange_edges[(a, b)] += 1

    if verbose:
        print(
            f"[GitHub-Score] Traversed {n_seen} commits ({n_skipped} skipped); "
            f"collected {len(all_basenames)} Java basenames.",
            flush=True,
        )

    emit(
        "history",
        f"Traversed {n_seen} commits; collected {len(all_basenames)} Java files.",
        100.0,
    )

    # ---- Co-change graph features via igraph (parity with training pipeline) ----
    emit("features", "Computing co-change graph features (degree, PageRank, betweenness)…", None)
    node_list = sorted(all_basenames)
    node_idx = {n: i for i, n in enumerate(node_list)}
    edges = [(node_idx[a], node_idx[b]) for (a, b) in cochange_edges]
    weights = list(cochange_edges.values())
    G = ig.Graph(n=len(node_list), edges=edges, directed=False)
    if len(node_list) > 2 and edges:
        bc_raw = G.betweenness(directed=False)
        denom = (len(node_list) - 1) * (len(node_list) - 2) / 2.0
        bc_norm = [b / denom for b in bc_raw]
    else:
        bc_norm = [0.0] * len(node_list)
    try:
        pr = G.pagerank(damping=0.85, weights=weights if weights else None)
    except Exception:
        pr = G.pagerank(damping=0.85)
    deg = G.degree()

    neighbour_weights: dict[str, list[int]] = {n: [] for n in node_list}
    for (a, b), w in cochange_edges.items():
        neighbour_weights[a].append(w)
        neighbour_weights[b].append(w)
    entropy: dict[str, float] = {}
    for n in node_list:
        ws = neighbour_weights[n]
        s = sum(ws)
        if s <= 0:
            entropy[n] = 0.0
            continue
        h = 0.0
        for w in ws:
            p = w / s
            if p > 0:
                h -= p * math.log2(p)
        entropy[n] = h

    emit("features", f"Computing 27 features across 5 families for {len(node_list)} files…", None)
    rows: list[dict] = []
    for bn in node_list:
        rec = file_records[bn]
        age_days = max(0, (rec["last_commit"] - rec["first_commit"]).days)
        days_since = max(0, (t_now - rec["last_commit"]).days)
        authors = authors_per_file.get(bn, Counter())
        n_authors = len(authors)
        max_author = max(authors.values()) if authors else 0
        ownership = (max_author / rec["total_commits_pre"]) if rec["total_commits_pre"] else 1.0
        bug_density = (rec["bugfix_commits_pre"] / rec["total_commits_pre"]) if rec["total_commits_pre"] else 0.0
        i = node_idx[bn]
        rows.append(
            {
                "basename": bn,
                # Family 1: size/complexity — SonarQube features default to 0
                "ncloc": 0.0, "complexity": 0.0, "cognitive_complexity": 0.0,
                "functions": 0.0, "classes": 0.0,
                # Family 2: static debt — SonarQube features default to 0
                "n_code_smells": 0, "n_bugs": 0, "total_debt_minutes": 0.0,
                "issue_density": 0.0, "duplicated_lines_density": 0.0,
                # Family 3: historical
                "total_commits_pre": rec["total_commits_pre"],
                "code_churn_pre": rec["code_churn_pre"],
                "recent_churn_90d": rec["recent_churn_90d"],
                "commit_frequency_30d": rec["commit_frequency_30d"],
                "file_age_days": age_days,
                "days_since_last_change": days_since,
                "contributor_count": n_authors,
                "ownership_ratio": ownership,
                # Family 4: co-change graph
                "cocg_degree": deg[i],
                "cocg_pagerank": pr[i],
                "cocg_betweenness": bc_norm[i],
                "cocg_entropy": entropy[bn],
                # Family 5: prior defect
                "bugfix_commits_pre": rec["bugfix_commits_pre"],
                "bugfix_commits_90d": rec["bugfix_commits_90d"],
                "bug_density_pre": bug_density,
                "n_jira_bugs_pre": 0,
                "jira_blocker_flag": 0,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _score_dataframe(features_df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p, scale, and call predict_proba using the persisted model."""
    best_model = joblib.load(MODELS_DIR / "best_model.pkl")
    scaler = joblib.load(MODELS_DIR / "feature_scaler.pkl")
    feat_names = pd.read_csv(MODELS_DIR / "feature_names.csv")["feature"].tolist()
    threshold = float((MODELS_DIR / "optimal_threshold.txt").read_text())

    X = features_df.reindex(columns=feat_names, fill_value=0).fillna(0).astype(float)
    assert list(X.columns) == feat_names, "Feature column mismatch"

    for col in LOG1P_FEATURES:
        if col in X.columns:
            X[col] = np.log1p(X[col])

    X_scaled = scaler.transform(X.values)
    probs = best_model.predict_proba(X_scaled)[:, 1]
    return pd.DataFrame(
        {
            "basename": features_df["basename"],
            "risk_score": probs,
            "predicted_high_risk": (probs >= threshold).astype(int),
        }
    ).sort_values("risk_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def score_github_repo(
    github_url: str,
    repo_name: str,
    *,
    clone_depth: int = 500,
    clone_root: Optional[Path] = None,
    cleanup: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Clone ``github_url``, compute the git-based features, and score every Java file.

    Parameters
    ----------
    github_url :
        Full https URL of a GitHub repository, e.g.
        ``https://github.com/apache/commons-cli``.
    repo_name :
        Short name used as the local clone directory and as the report
        filename suffix.
    clone_depth :
        Number of most-recent commits to score over. The clone is fetched
        shallow at this depth, then unshallowed (so PyDriller can compute
        diffs), then traversal is limited back to these commits to keep
        runtime bounded on large repos.
    clone_root :
        Parent directory for the clone. Defaults to ``<repo>/data/repos/``.
    cleanup :
        Remove the clone directory after scoring (off by default so a
        re-run is fast).
    verbose :
        Print progress.

    Returns
    -------
    DataFrame sorted by descending ``risk_score`` with columns
    ``basename``, ``risk_score``, ``predicted_high_risk``.
    """
    clone_root = clone_root or (PROJECT_ROOT / "data" / "repos")
    clone_dir = clone_root / repo_name

    if verbose:
        print(f"[GitHub-Score] Repo: {github_url}", flush=True)
        print(f"[GitHub-Score] Clone target: {clone_dir}", flush=True)

    _clone_repo(github_url, clone_dir, clone_depth)

    feats = _extract_git_features(clone_dir, clone_depth, verbose=verbose)
    if feats.empty:
        raise RuntimeError("No Java files found in the requested commit window.")

    scored = _score_dataframe(feats)

    if cleanup:
        shutil.rmtree(clone_dir, ignore_errors=True)

    return scored
