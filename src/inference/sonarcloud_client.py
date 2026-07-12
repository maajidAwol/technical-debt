"""
SonarCloud / self-hosted SonarQube client used by Section 16 of the Colab
notebook and by ``scripts/sonar_local_integration.py``.

The trained model uses 10 SonarQube features (Families 1 & 2 of the 27-
feature catalogue). This module wraps the four pieces of behaviour needed
to populate those features on an arbitrary Java repo:

1. ``run_scanner`` — invoke the ``sonar-scanner`` CLI on a clone.
2. ``wait_for_analysis`` — poll the Compute Engine API until the scan is
   processed.
3. ``fetch_file_metrics`` — paginate ``api/measures/component_tree`` and
   return one row per file with the raw SonarQube metric columns.
4. ``to_feature_columns`` — rename + derive the 10 training feature
   columns (``ncloc``, ``complexity``, ``cognitive_complexity``,
   ``functions``, ``classes``, ``n_code_smells``, ``n_bugs``,
   ``total_debt_minutes``, ``issue_density``, ``duplicated_lines_density``)
   and warn on basename collisions.

Both SonarCloud (https://sonarcloud.io) and a self-hosted SonarQube at
``http://localhost:9000`` speak the same REST API; the ``base_url``
parameter selects between them.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Metric mapping — SonarQube metric key -> training feature column
# ---------------------------------------------------------------------------
# Order matches the training catalogue's Family 1 then Family 2.
SONAR_METRIC_KEYS: tuple[str, ...] = (
    "ncloc",
    "complexity",
    "cognitive_complexity",
    "functions",
    "classes",
    "code_smells",
    "bugs",
    "sqale_index",
    "violations",
    "duplicated_lines_density",
)

_METRIC_TO_FEATURE = {
    "ncloc": "ncloc",
    "complexity": "complexity",
    "cognitive_complexity": "cognitive_complexity",
    "functions": "functions",
    "classes": "classes",
    "code_smells": "n_code_smells",
    "bugs": "n_bugs",
    "sqale_index": "total_debt_minutes",
    "duplicated_lines_density": "duplicated_lines_density",
    # `violations` is consumed by `to_feature_columns` to derive `issue_density`
    # and dropped afterwards; no direct rename.
}


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------
def run_scanner(
    repo_path: Path | str,
    project_key: str,
    token: str,
    host_url: str,
    *,
    organization: Optional[str] = None,
    sonar_scanner_bin: str = "sonar-scanner",
    extra_args: Optional[List[str]] = None,
) -> int:
    """Invoke ``sonar-scanner`` and stream its output.

    Parameters
    ----------
    repo_path :
        Path to the cloned Java repo (used as ``-Dsonar.sources``).
    project_key :
        SonarQube/SonarCloud project key.
    token :
        Auth token (Bearer).
    host_url :
        SonarQube host. ``https://sonarcloud.io`` for SonarCloud,
        ``http://localhost:9000`` for a self-hosted instance.
    organization :
        Required for SonarCloud, omitted for self-hosted SonarQube.
    sonar_scanner_bin :
        Path/name of the ``sonar-scanner`` binary. Default picks it up
        from ``PATH``.
    extra_args :
        Extra ``-Dkey=value`` flags appended verbatim.

    Returns
    -------
    Scanner exit code (0 on success).
    """
    repo_path = Path(repo_path)
    cmd: List[str] = [
        sonar_scanner_bin,
        f"-Dsonar.projectKey={project_key}",
        f"-Dsonar.sources={repo_path}",
        f"-Dsonar.host.url={host_url}",
        f"-Dsonar.token={token}",
        "-Dsonar.java.binaries=.",
    ]
    if organization:
        cmd.append(f"-Dsonar.organization={organization}")
    if extra_args:
        cmd.extend(extra_args)

    proc = subprocess.run(cmd, cwd=str(repo_path))
    return proc.returncode


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------
def wait_for_analysis(
    base_url: str,
    project_key: str,
    token: str,
    *,
    max_retries: int = 20,
    sleep_s: float = 10.0,
    verbose: bool = True,
) -> None:
    """Poll the Compute Engine API until the latest analysis finishes.

    Raises
    ------
    TimeoutError
        If ``current.status`` is not ``SUCCESS`` (or the queue is not empty)
        after ``max_retries`` attempts.
    RuntimeError
        If the API call fails for a non-transient reason.
    """
    url = f"{base_url.rstrip('/')}/api/ce/component"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"component": project_key}

    for attempt in range(1, max_retries + 1):
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code == 404:
            # CE has no record for this project key yet; analysis may not
            # have started, give it a chance to appear.
            if verbose:
                print(f"  [poll {attempt}/{max_retries}] not yet visible, waiting...", flush=True)
            time.sleep(sleep_s)
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"CE poll failed ({r.status_code}): {r.text[:200]}")
        data = r.json()
        queue = data.get("queue", [])
        current = data.get("current") or {}
        status = current.get("status")
        if not queue and status == "SUCCESS":
            if verbose:
                print(f"  [poll {attempt}/{max_retries}] analysis SUCCESS", flush=True)
            return
        if status == "FAILED":
            raise RuntimeError(f"SonarQube analysis FAILED: {current!r}")
        if verbose:
            print(
                f"  [poll {attempt}/{max_retries}] queue={len(queue)} status={status or 'pending'}",
                flush=True,
            )
        time.sleep(sleep_s)

    raise TimeoutError(
        f"SonarQube analysis did not finish after {max_retries} polls "
        f"({max_retries * sleep_s:.0f}s). Check the scan logs."
    )


# ---------------------------------------------------------------------------
# Metric fetcher
# ---------------------------------------------------------------------------
def fetch_file_metrics(
    base_url: str,
    project_key: str,
    token: str,
    *,
    metric_keys: Optional[Iterable[str]] = None,
    page_size: int = 500,
    verbose: bool = True,
) -> pd.DataFrame:
    """Paginate ``api/measures/component_tree`` and return one row per file.

    Each row carries ``path`` (the repo-relative path, with the
    ``projectKey:`` prefix stripped), ``basename``, and one numeric column
    per requested metric. Missing metric values are NaN; ``to_feature_columns``
    later fills them with 0.
    """
    metric_keys = list(metric_keys or SONAR_METRIC_KEYS)
    url = f"{base_url.rstrip('/')}/api/measures/component_tree"
    headers = {"Authorization": f"Bearer {token}"}
    rows: list[dict] = []
    page = 1

    while True:
        params = {
            "component": project_key,
            "metricKeys": ",".join(metric_keys),
            "qualifiers": "FIL",
            "ps": page_size,
            "p": page,
        }
        r = requests.get(url, headers=headers, params=params, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"measures fetch failed ({r.status_code}): {r.text[:200]}")
        body = r.json()
        components = body.get("components", [])
        if not components:
            break

        for comp in components:
            raw_key = comp.get("key", "")
            # Strip "projectKey:" prefix if present
            path = raw_key.split(":", 1)[1] if ":" in raw_key else raw_key
            row = {
                "path": path,
                "basename": path.rsplit("/", 1)[-1],
            }
            for m in comp.get("measures", []):
                key = m.get("metric")
                val = m.get("value")
                try:
                    row[key] = float(val) if val is not None else None
                except (TypeError, ValueError):
                    row[key] = None
            rows.append(row)

        # Pagination: SonarCloud honours `paging.total`; bail when we've seen them all.
        paging = body.get("paging", {})
        total = int(paging.get("total", 0))
        seen = page * page_size
        if verbose:
            print(f"  [fetch page {page}] {len(components)} files (cumulative {min(seen, total)} / {total})", flush=True)
        if seen >= total:
            break
        page += 1

    df = pd.DataFrame(rows)
    # Filter to .java files for parity with the training pipeline.
    if not df.empty:
        df = df[df["basename"].str.endswith(".java", na=False)].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Feature column derivation
# ---------------------------------------------------------------------------
def to_feature_columns(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """Rename SonarQube metric columns to training feature names and derive ``issue_density``.

    Input is the DataFrame from ``fetch_file_metrics``. Output keeps
    ``path`` and ``basename`` plus exactly the 10 training feature columns
    (Families 1 + 2). Basename collisions are detected and warned about;
    when present, the first row wins.
    """
    if df.empty:
        cols = ["path", "basename"] + list(_METRIC_TO_FEATURE.values()) + ["issue_density"]
        return pd.DataFrame(columns=cols)

    out = df.copy()
    # Make sure every expected metric column exists (missing -> 0).
    for k in SONAR_METRIC_KEYS:
        if k not in out.columns:
            out[k] = 0.0
        out[k] = out[k].fillna(0).astype(float)

    # Compute issue_density before renaming `violations` away.
    ncloc_safe = out["ncloc"].where(out["ncloc"] > 0, other=float("nan"))
    out["issue_density"] = (out["violations"] / ncloc_safe).fillna(0.0)

    # Rename to training feature names, drop intermediates.
    out = out.rename(columns=_METRIC_TO_FEATURE).drop(columns=["violations"])

    # Basename collision check
    dup_mask = out["basename"].duplicated(keep=False)
    if verbose and dup_mask.any():
        offenders = out.loc[dup_mask, ["basename", "path"]].sort_values("basename")
        n_unique = offenders["basename"].nunique()
        print(
            f"[sonar] WARNING: {n_unique} basename(s) appear in multiple SonarCloud paths; "
            "keeping the first occurrence per basename.",
            flush=True,
        )
        print(offenders.to_string(index=False), flush=True)
        out = out.drop_duplicates(subset=["basename"], keep="first").reset_index(drop=True)

    feature_cols = list(_METRIC_TO_FEATURE.values()) + ["issue_density"]
    return out[["path", "basename"] + feature_cols].reset_index(drop=True)


__all__ = [
    "SONAR_METRIC_KEYS",
    "fetch_file_metrics",
    "run_scanner",
    "to_feature_columns",
    "wait_for_analysis",
]
