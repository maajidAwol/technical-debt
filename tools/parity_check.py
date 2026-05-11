"""
End-to-end parity check: igraph centralities vs NetworkX reference.

Runs the Stage 5 graph-feature builder on a single real project
(``archiva`` by default) and compares each centrality column against
a fresh NetworkX reference computation on the same graph. Fails loudly
if any column drifts beyond the documented tolerances.

This script is **not** part of the pipeline. Invoke manually after
changing ``src/features/graph_features.py`` or upgrading igraph /
networkx::

    .\\venv\\Scripts\\python.exe tools/parity_check.py
    .\\venv\\Scripts\\python.exe tools/parity_check.py --project org.apache:cocoon

Exits with non-zero status on any parity violation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR  # noqa: E402
from src.data.snapshot import load_snapshots  # noqa: E402
from src.features.graph_features import (  # noqa: E402
    _build_graph_from_changes,
    build_graph_features_for_project,
)


# Reference tolerances mirror the in-module parity test.
TOL_BC = 1e-9
TOL_CL = 1e-9
TOL_CC = 1e-9
TOL_PR = 5e-6


def _load_changes() -> pd.DataFrame:
    ch = pd.read_parquet(PROCESSED_DATA_DIR / "clean_git_commits_changes.parquet")
    if ch["DATE"].dtype.kind == "M" and ch["DATE"].dt.tz is None:
        ch["DATE"] = ch["DATE"].dt.tz_localize("UTC")
    return ch


def _compute_networkx_reference(G: nx.Graph) -> dict[str, dict[str, float]]:
    """Compute the same four centralities the production code returns,
    but using NetworkX's pure-Python implementations as ground truth."""
    bc = nx.betweenness_centrality(G, normalized=True)
    cl = nx.closeness_centrality(G)
    cc = nx.clustering(G)
    try:
        pr = nx.pagerank(G, weight="weight")
    except nx.PowerIterationFailedConvergence:
        pr = nx.pagerank(G)
    return {"betweenness": bc, "closeness": cl, "clustering": cc, "pagerank": pr}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="org.apache:archiva")
    args = parser.parse_args()

    print(f"[parity] project: {args.project}")

    snaps = load_snapshots(PROCESSED_DATA_DIR / "project_snapshots.parquet")
    row = snaps[snaps["project_id"] == args.project]
    if row.empty:
        print(f"[parity] FAIL: project {args.project} not in snapshots", file=sys.stderr)
        return 2
    snapshot = row.iloc[0]["snapshot_date"]

    changes = _load_changes()

    print("[parity] building production features (igraph) ...")
    feats = build_graph_features_for_project(args.project, snapshot, changes, verbose=False)
    print(f"   rows={len(feats)} columns={list(feats.columns)[:6]}...")

    print("[parity] reconstructing NetworkX reference ...")
    ch = changes[
        (changes["PROJECT_ID"] == args.project) & (changes["DATE"] <= snapshot)
    ]
    G = _build_graph_from_changes(ch[["COMMIT_HASH", "basename"]])
    print(f"   graph V={G.number_of_nodes()} E={G.number_of_edges()}")
    ref = _compute_networkx_reference(G)

    pairs = [
        ("cocg_betweenness", "betweenness", TOL_BC),
        ("cocg_closeness", "closeness", TOL_CL),
        ("cocg_clustering_coef", "clustering", TOL_CC),
        ("cocg_pagerank", "pagerank", TOL_PR),
    ]
    failures: list[tuple[str, float, float]] = []
    for col, key, tol in pairs:
        # Only nodes that actually exist in the graph have a meaningful
        # NetworkX reference; basenames that exist at the snapshot but
        # never co-changed are zero-filled and trivially match.
        ig_vals = feats.set_index("basename")[col]
        diff = 0.0
        worst_node = None
        for node, ref_val in ref[key].items():
            v = float(ig_vals.get(node, 0.0))
            d = abs(v - float(ref_val))
            if d > diff:
                diff = d
                worst_node = node
        ok = diff <= tol
        status = "ok" if ok else "FAIL"
        print(
            f"   {col:<24} max_abs_diff={diff:.3e}  tol={tol:.0e}  worst_node={worst_node}  [{status}]"
        )
        if not ok:
            failures.append((col, diff, tol))

    if failures:
        print(f"\n[parity] FAIL: {len(failures)} columns drifted beyond tolerance")
        return 1
    print(f"\n[parity] PASS for {args.project}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
