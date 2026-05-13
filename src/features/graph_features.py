"""
Family 4 - Co-change graph features at (project, basename) granularity.

For each project at snapshot ``t``, an undirected weighted co-change
graph is built using commits with ``DATE <= t``:

- Nodes: distinct basenames touched by any pre-``t`` commit.
- Edges: for every commit that touches multiple files, every pair of
  basenames gets +1 edge weight (so weight = number of shared commits).

Four features (per spec):
    cocg_degree       - number of co-change neighbours
    cocg_pagerank     - weighted PageRank (damping 0.85)
    cocg_betweenness  - normalised betweenness centrality
    cocg_entropy      - co-change scattering entropy
                        (Ethari & Bhardwaj 2025):
                        H = -sum(p_i * log2(p_i)) over normalised
                        edge weights to neighbours; 0.0 if degree==0.

igraph is used for the centralities (PRPACK + Brandes); a parity check
against NetworkX runs at module import on a 50-node BA graph.
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.szz import basename_universe_at_snapshot  # noqa: E402


_PARITY_TOL_BC = 1e-9
_PARITY_TOL_PR = 5e-6

GRAPH_FEATURE_COLS: tuple[str, ...] = (
    "cocg_degree",
    "cocg_pagerank",
    "cocg_betweenness",
    "cocg_entropy",
)


def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts


def _empty_features_for_universe(universe: pd.DataFrame) -> pd.DataFrame:
    out = universe.copy()
    out["cocg_degree"] = 0
    out["cocg_pagerank"] = 0.0
    out["cocg_betweenness"] = 0.0
    out["cocg_entropy"] = 0.0
    out["cocg_degree"] = out["cocg_degree"].astype("int64")
    return out


def _build_graph_from_changes(ch: pd.DataFrame) -> nx.Graph:
    by_commit = ch.groupby("COMMIT_HASH")["basename"].unique()
    edge_counts: Counter[tuple[str, str]] = Counter()
    all_nodes: set[str] = set()
    for files in by_commit:
        if files is None:
            continue
        all_nodes.update(files)
        if len(files) < 2:
            continue
        sorted_files = sorted(files)
        edge_counts.update(combinations(sorted_files, 2))

    G = nx.Graph()
    G.add_nodes_from(all_nodes)
    if edge_counts:
        G.add_weighted_edges_from(((u, v, w) for (u, v), w in edge_counts.items()))
    return G


def _compute_centralities_igraph(G: nx.Graph) -> tuple[dict[str, float], dict[str, float]]:
    """Return (betweenness, pagerank) dicts keyed by basename."""
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}, {}

    idx = {nm: i for i, nm in enumerate(nodes)}
    edge_list: list[tuple[int, int]] = []
    weights: list[float] = []
    for u, v, d in G.edges(data=True):
        edge_list.append((idx[u], idx[v]))
        weights.append(float(d.get("weight", 1.0)))
    ig_G = ig.Graph(n=n, edges=edge_list, directed=False)

    if n > 2:
        bc_raw = ig_G.betweenness(directed=False)
        denom = (n - 1) * (n - 2) / 2.0
        betweenness = {nodes[i]: float(bc_raw[i] / denom) for i in range(n)}
    else:
        betweenness = {nodes[i]: 0.0 for i in range(n)}

    pr_weights = weights if weights else None
    try:
        pr = ig_G.pagerank(damping=0.85, weights=pr_weights)
    except Exception:
        pr = ig_G.pagerank(damping=0.85)
    pagerank = {nodes[i]: float(pr[i]) for i in range(n)}
    return betweenness, pagerank


def _compute_entropy(G: nx.Graph) -> dict[str, float]:
    """Per-node Shannon entropy over normalised edge weights to neighbours.

    H_v = -sum_i p_i * log2(p_i) where p_i = w_i / sum(w).
    Returns 0.0 for isolated nodes (degree == 0). Single-neighbour nodes
    have p_1 = 1.0 -> 0.0 entropy as well.
    """
    out: dict[str, float] = {}
    for node in G.nodes():
        neighbours = list(G[node])
        if not neighbours:
            out[node] = 0.0
            continue
        weights = [float(G[node][nb].get("weight", 1.0)) for nb in neighbours]
        total = sum(weights)
        if total <= 0:
            out[node] = 0.0
            continue
        h = 0.0
        for w in weights:
            p = w / total
            if p > 0:
                h -= p * math.log2(p)
        out[node] = float(h)
    return out


def _parity_test() -> None:
    """Assert igraph centralities match NetworkX on a small graph at import."""
    rng = np.random.default_rng(42)
    G = nx.barabasi_albert_graph(50, 3, seed=42)
    for u, v in G.edges():
        G[u][v]["weight"] = float(rng.integers(1, 10))

    ig_bc, ig_pr = _compute_centralities_igraph(G)
    nx_bc = nx.betweenness_centrality(G, normalized=True)
    nx_pr = nx.pagerank(G, weight="weight")

    def _max_diff(a: dict, b: dict) -> float:
        return max(abs(a[k] - b[k]) for k in a)

    diff_bc = _max_diff(ig_bc, nx_bc)
    diff_pr = _max_diff(ig_pr, nx_pr)
    failures = []
    if diff_bc > _PARITY_TOL_BC:
        failures.append(f"betweenness: {diff_bc:.3e} > tol {_PARITY_TOL_BC:.0e}")
    if diff_pr > _PARITY_TOL_PR:
        failures.append(f"pagerank: {diff_pr:.3e} > tol {_PARITY_TOL_PR:.0e}")
    if failures:
        raise RuntimeError(
            "graph_features parity test FAILED (igraph vs networkx):\n  "
            + "\n  ".join(failures)
        )


_parity_test()


def build_graph_features_for_project(
    project_id: str,
    snapshot: pd.Timestamp,
    changes: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Per-basename co-change graph features for one project at snapshot ``t``."""
    import time as _time

    def _log(msg: str) -> None:
        if verbose:
            print(f"      [graph:{project_id}] {msg}", flush=True)

    t = _ensure_utc(snapshot)
    universe = basename_universe_at_snapshot(changes, project_id, t)
    if universe.empty:
        return universe

    ch = changes[(changes["PROJECT_ID"] == project_id) & (changes["DATE"] <= t)]
    if ch.empty:
        return _empty_features_for_universe(universe)

    _ts = _time.time()
    G = _build_graph_from_changes(ch[["COMMIT_HASH", "basename"]])
    _log(f"graph built V={G.number_of_nodes()} E={G.number_of_edges()} ({_time.time()-_ts:.1f}s)")
    if G.number_of_edges() == 0:
        return _empty_features_for_universe(universe)

    _ts = _time.time()
    betweenness, pagerank = _compute_centralities_igraph(G)
    _log(f"centralities ({_time.time()-_ts:.1f}s)")

    _ts = _time.time()
    entropy = _compute_entropy(G)
    _log(f"entropy ({_time.time()-_ts:.1f}s)")

    degree = dict(G.degree())

    rows = []
    for node in G.nodes():
        rows.append(
            {
                "basename": node,
                "cocg_degree": int(degree.get(node, 0)),
                "cocg_pagerank": float(pagerank.get(node, 0.0)),
                "cocg_betweenness": float(betweenness.get(node, 0.0)),
                "cocg_entropy": float(entropy.get(node, 0.0)),
            }
        )
    feats = pd.DataFrame(rows)

    out = universe.merge(feats, on="basename", how="left")
    out["cocg_degree"] = out["cocg_degree"].fillna(0).astype("int64")
    for col in ("cocg_pagerank", "cocg_betweenness", "cocg_entropy"):
        out[col] = out[col].fillna(0.0).astype(float)
    return out
