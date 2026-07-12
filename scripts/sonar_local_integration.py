"""
Local SonarQube integration — fetch the 10 SonarQube features for an
arbitrary Java repo using a self-hosted SonarQube instance.

This script is the local counterpart of Section 16 of the Colab notebook.
It expects a SonarQube server already running at ``--sonar-url``
(default ``http://localhost:9000``) and the ``sonar-scanner`` CLI
available on PATH.

Usage
-----
.. code-block:: bash

    python scripts/sonar_local_integration.py \\
        --repo-path /path/to/cloned/repo \\
        --project-key my-project \\
        --token <local-token> \\
        --sonar-url http://localhost:9000 \\
        --output results/live/sonar_metrics.csv

Setup (one-time)
----------------
1. Unzip SonarQube into a directory of your choice.
2. Start the server:
     - Windows: ``<sonarqube>\\bin\\windows-x86-64\\StartSonar.bat``
     - macOS / Linux: ``<sonarqube>/bin/<os>/sonar.sh console``
3. Open http://localhost:9000 and log in with ``admin`` / ``admin``
   (change the password on first login).
4. Create a project from the UI (or via the API) and note the project key.
5. Generate a user token at
   ``My Account -> Security -> Generate Tokens`` and pass it via ``--token``.
6. Install the ``sonar-scanner`` CLI and make sure it is on PATH:
   https://docs.sonarqube.org/latest/analyzing-source-code/scanners/sonarscanner/
7. Run this script with the project key + token.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.inference.sonarcloud_client import (  # noqa: E402
    fetch_file_metrics,
    run_scanner,
    to_feature_columns,
    wait_for_analysis,
)


SETUP_HINTS = """\
[sonar-local] Reminder — local SonarQube setup checklist:
  1. SonarQube server is running at the URL passed via --sonar-url
  2. The project key passed via --project-key exists on that server
  3. A user token has been generated (My Account -> Security)
  4. `sonar-scanner` is installed and on PATH (or pass --sonar-scanner-path)
"""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--repo-path", type=Path, required=True, help="Path to a cloned Java repo (used as -Dsonar.sources)")
    p.add_argument("--project-key", required=True, help="SonarQube project key")
    p.add_argument("--token", required=True, help="SonarQube auth token (user token)")
    p.add_argument("--sonar-url", default="http://localhost:9000", help="SonarQube base URL (default: http://localhost:9000)")
    p.add_argument("--sonar-scanner-path", default="sonar-scanner", help="sonar-scanner binary (default: PATH lookup)")
    p.add_argument("--output", type=Path, default=Path("results/live/sonar_metrics.csv"), help="Output CSV path (default: results/live/sonar_metrics.csv)")
    p.add_argument("--skip-scan", action="store_true", help="Skip running sonar-scanner; assume the latest analysis is already published")
    p.add_argument("--max-retries", type=int, default=20, help="Max CE-poll retries (default: 20)")
    p.add_argument("--sleep-s", type=float, default=10.0, help="Seconds between CE polls (default: 10)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    t0 = time.time()
    print(SETUP_HINTS)
    print(f"[sonar-local] Repo path     : {args.repo_path}")
    print(f"[sonar-local] Project key   : {args.project_key}")
    print(f"[sonar-local] Sonar URL     : {args.sonar_url}")
    print(f"[sonar-local] Output CSV    : {args.output}")

    if not args.repo_path.exists():
        print(f"[sonar-local] ERROR: --repo-path does not exist: {args.repo_path}", file=sys.stderr)
        return 2

    if not args.skip_scan:
        print(f"\n[sonar-local] Running scanner ...")
        rc = run_scanner(
            args.repo_path,
            args.project_key,
            args.token,
            args.sonar_url,
            organization=None,  # self-hosted SonarQube does not need an org
            sonar_scanner_bin=args.sonar_scanner_path,
        )
        if rc != 0:
            print(f"[sonar-local] ERROR: sonar-scanner exited {rc}", file=sys.stderr)
            return rc
        print(f"[sonar-local] Scanner exited 0")
    else:
        print(f"[sonar-local] --skip-scan set; using whatever's already on the server")

    print(f"\n[sonar-local] Polling Compute Engine ...")
    wait_for_analysis(
        args.sonar_url,
        args.project_key,
        args.token,
        max_retries=args.max_retries,
        sleep_s=args.sleep_s,
    )

    print(f"\n[sonar-local] Fetching file-level metrics ...")
    raw = fetch_file_metrics(args.sonar_url, args.project_key, args.token)
    feats = to_feature_columns(raw)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(args.output, index=False)
    print(f"\n[sonar-local] Wrote {args.output}  ({len(feats)} files, {feats.shape[1]} columns)")
    if not feats.empty:
        print("\n[sonar-local] First 5 rows:")
        with pd.option_context("display.width", 160, "display.max_columns", None):
            print(feats.head().to_string(index=False))

    print(f"\n[sonar-local] Elapsed: {time.time() - t0:.1f}s")
    print("[sonar-local] Complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
