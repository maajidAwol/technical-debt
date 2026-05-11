"""
Snapshot-aware co-change graph features at (project, basename) granularity.

For each project at snapshot ``t``, we build an undirected weighted
co-change graph using only commits with ``DATE <= t``:

- Nodes: distinct basenames touched by any pre-``t`` commit.
- Edges: for every commit that touches multiple files, every pair of
  basenames in that commit gets +1 edge weight (so weight = number of
  shared commits).

From this graph we compute, per basename, a small set of structural
features that capture how *coupled* and *central* the file is in the
project's evolutionary history:

- ``cocg_degree`` - number of distinct neighbours (co-changing files).
- ``cocg_strength_sum`` - sum of incident edge weights (total coupling).
- ``cocg_strength_mean`` - mean edge weight (average coupling intensity).
- ``cocg_strength_max`` - max single-edge weight.
- ``cocg_betweenness`` - betweenness centrality (approximate via
  ``k=500`` sampling on graphs > 5,000 nodes).
- ``cocg_closeness`` - closeness centrality.
- ``cocg_clustering_coef`` - local clustering coefficient (unweighted).
- ``cocg_pagerank`` - weighted PageRank.
- ``cocg_neighbour_count_30d`` / ``cocg_neighbour_count_90d`` -
  neighbour count restricted to commits in the recent 30 / 90 days
  before ``t`` (recency-aware coupling).

Why these features
------------------
Jiang et al. (2024, 2025) and Robredo et al. (2025) report that
graph-based metrics improve technical-debt and defect prediction by
encoding inter-file coupling that per-file aggregates cannot. The MSc
research proposal (Section 2.2 / refs [7], [8]) explicitly flags this
as an under-explored direction. This module adds those features
without leaving the dataset's own commit log.

References
----------
- Jiang, Y., et al. (2024). Graph-based defect prediction. EMSE.
- Robredo, R., et al. (2025). Co-change networks for software quality. JSS.
- Newman, M. E. J. (2003). The structure and function of complex networks.
"""
from __future__ import annotations

import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Optional

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.szz import basename_universe_at_snapshot  # noqa: E402


# Centrality computation strategy (post 2026-04-29 overhaul):
# - Betweenness, closeness, clustering, and weighted PageRank are
#   computed via ``igraph`` (C-backed). NetworkX's pure-Python
#   implementations are 30-100x slower and were the dominant cost of
#   Stage 5. Same exact algorithms (Brandes for betweenness, BFS for
#   closeness, triangle counting for clustering, PRPACK for PageRank);
#   identical output up to float rounding (verified at module import
#   via :func:`_parity_test`).
# - Wasserman-Faust correction for closeness (NetworkX default since
#   2.0) is applied manually after igraph because igraph does not
#   apply WF by default.
# Parity tolerances against NetworkX reference. Betweenness, closeness,
# and clustering use deterministic graph traversals on both libraries
# and agree to float rounding (1e-9). PageRank has a slightly looser
# tolerance because NetworkX uses power iteration with ``tol=1e-6``
# default while igraph's PRPACK solver is tighter; the typical
# PageRank value on a 50-node graph is O(1/n) ~= 0.02, so 5e-6
# absolute corresponds to ~2.5e-4 relative - scientifically negligible.
_PARITY_TOL_BC = 1e-9
_PARITY_TOL_CL = 1e-9
_PARITY_TOL_CC = 1e-9
_PARITY_TOL_PR = 5e-6

# Output column names. Listed here so config.py / ablation can match
# them by name without import order issues.
GRAPH_FEATURE_COLS: tuple[str, ...] = (
    "cocg_degree",
    "cocg_strength_sum",
    "cocg_strength_mean",
    "cocg_strength_max",
    "cocg_betweenness",
    "cocg_closeness",
    "cocg_clustering_coef",
    "cocg_pagerank",
    "cocg_neighbour_count_30d",
    "cocg_neighbour_count_90d",
)


def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts


def _empty_features_for_universe(universe: pd.DataFrame) -> pd.DataFrame:
    out = universe.copy()
    for col in GRAPH_FEATURE_COLS:
        out[col] = 0.0
    # Integer-typed columns
    for col in ("cocg_degree", "cocg_neighbour_count_30d", "cocg_neighbour_count_90d"):
        out[col] = out[col].astype("int64")
    return out


def _build_graph_from_changes(ch: pd.DataFrame) -> nx.Graph:
    """Build a weighted undirected co-change graph from a per-(commit, file) frame.

    Nodes are basenames; edge ``(u, v).weight`` = number of commits in
    which ``u`` and ``v`` both appeared.

    Implementation note: edge counting is vectorised via a single
    ``Counter`` over ``itertools.combinations`` and pushed into NetworkX
    in one ``add_weighted_edges_from`` call. This avoids ~10x of pure-
    Python ``has_edge`` / ``add_edge`` overhead per pair, which is the
    dominant cost for projects with large refactor commits.
    """
    by_commit = ch.groupby("COMMIT_HASH")["basename"].unique()

    edge_counts: Counter[tuple[str, str]] = Counter()
    all_nodes: set[str] = set()
    for files in by_commit:
        if files is None or len(files) < 2:
            if files is not None:
                all_nodes.update(files)
            continue
        # Sort once so (u, v) is canonical: avoids double-counting and
        # keeps the Counter key space minimal.
        sorted_files = sorted(files)
        all_nodes.update(sorted_files)
        edge_counts.update(combinations(sorted_files, 2))

    G = nx.Graph()
    G.add_nodes_from(all_nodes)
    if edge_counts:
        G.add_weighted_edges_from(
            ((u, v, w) for (u, v), w in edge_counts.items())
        )
    return G


def _compute_centralities_igraph(G: nx.Graph) -> dict[str, dict[str, float]]:
    """Compute betweenness, closeness, clustering, weighted PageRank via igraph.

    Returns a dict with four sub-dicts keyed by metric name, each
    mapping basename -> value. Outputs match NetworkX's
    ``betweenness_centrality(normalized=True)``,
    ``closeness_centrality()`` (with ``wf_improved=True``),
    ``clustering()``, and ``pagerank(weight='weight')`` to within the
    tolerances declared at module top (verified by
    :func:`_parity_test`).

    Implementation notes
    --------------------
    - Edge construction is a single linear pass over ``G.edges(data=True)``.
    - Betweenness: igraph returns raw counts for undirected; we apply
      the standard Brandes normalisation factor ``2 / ((n-1)(n-2))``.
    - Closeness: igraph's ``normalized=True`` yields
      ``(c-1) / sum_within_component_distances(v)``. NetworkX with
      ``wf_improved=True`` then multiplies by ``(c-1)/(n-1)`` so that
      nodes in small components are not artificially favoured. We
      apply that factor explicitly here.
    - Clustering: ``transitivity_local_undirected(mode='zero')`` returns
      0 for nodes with fewer than two neighbours, matching NetworkX.
    - PageRank uses igraph's PRPACK solver with the same damping
      (0.85) NetworkX uses. Numerical drift between PRPACK and power
      iteration is below ``_PARITY_TOL_PR``.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {k: {} for k in ("betweenness", "closeness", "clustering", "pagerank")}

    idx = {nm: i for i, nm in enumerate(nodes)}
    edge_list: list[tuple[int, int]] = []
    weights: list[float] = []
    for u, v, d in G.edges(data=True):
        edge_list.append((idx[u], idx[v]))
        weights.append(float(d.get("weight", 1.0)))

    ig_G = ig.Graph(n=n, edges=edge_list, directed=False)

    # ----- betweenness (unweighted, normalised) -----
    if n > 2:
        bc_raw = ig_G.betweenness(directed=False)
        denom = (n - 1) * (n - 2) / 2.0
        betweenness = {nodes[i]: float(bc_raw[i] / denom) for i in range(n)}
    else:
        betweenness = {nodes[i]: 0.0 for i in range(n)}

    # ----- closeness with Wasserman-Faust correction -----
    cl_raw = ig_G.closeness(normalized=True)
    components = ig_G.connected_components()
    membership = components.membership
    sizes = components.sizes()
    closeness: dict[str, float] = {}
    for i, val in enumerate(cl_raw):
        c = sizes[membership[i]]
        if c <= 1 or n <= 1 or val is None:
            closeness[nodes[i]] = 0.0
            continue
        v_float = float(val)
        if not np.isfinite(v_float):
            closeness[nodes[i]] = 0.0
        else:
            closeness[nodes[i]] = v_float * ((c - 1) / (n - 1))

    # ----- local clustering coefficient -----
    cl_local = ig_G.transitivity_local_undirected(mode="zero")
    clustering = {nodes[i]: float(cl_local[i]) for i in range(n)}

    # ----- weighted PageRank -----
    pr_weights = weights if weights else None
    try:
        pr = ig_G.pagerank(damping=0.85, weights=pr_weights)
    except Exception:
        pr = ig_G.pagerank(damping=0.85)
    pagerank = {nodes[i]: float(pr[i]) for i in range(n)}

    return {
        "betweenness": betweenness,
        "closeness": closeness,
        "clustering": clustering,
        "pagerank": pagerank,
    }


def _parity_test() -> None:
    """Assert that igraph centralities match NetworkX within tolerance.

    Runs once at module import on a deterministic 50-node Barabasi-Albert
    graph with random integer weights. Catches normalisation drift if
    a future igraph or NetworkX release changes conventions. Total
    cost <100ms.
    """
    rng = np.random.default_rng(42)
    G = nx.barabasi_albert_graph(50, 3, seed=42)
    for u, v in G.edges():
        G[u][v]["weight"] = float(rng.integers(1, 10))

    ig_out = _compute_centralities_igraph(G)
    nx_bc = nx.betweenness_centrality(G, normalized=True)
    nx_cl = nx.closeness_centrality(G)
    nx_cc = nx.clustering(G)
    nx_pr = nx.pagerank(G, weight="weight")

    def _max_diff(a: dict, b: dict) -> float:
        return max(abs(a[k] - b[k]) for k in a)

    diffs = {
        "betweenness": (_max_diff(ig_out["betweenness"], nx_bc), _PARITY_TOL_BC),
        "closeness": (_max_diff(ig_out["closeness"], nx_cl), _PARITY_TOL_CL),
        "clustering": (_max_diff(ig_out["clustering"], nx_cc), _PARITY_TOL_CC),
        "pagerank": (_max_diff(ig_out["pagerank"], nx_pr), _PARITY_TOL_PR),
    }
    failures = [(m, d, tol) for m, (d, tol) in diffs.items() if d > tol]
    if failures:
        msg = "\n".join(
            f"  {m}: max abs diff {d:.3e} > tol {tol:.0e}" for m, d, tol in failures
        )
        raise RuntimeError(
            "graph_features parity test FAILED (igraph vs networkx):\n"
            f"{msg}\n"
            "Refusing to import; pipeline would produce inconsistent "
            "centrality values vs the documented baselines."
        )


# Run parity check exactly once at module import. Cheap (<100ms),
# fail-loud, and protects against silent normalisation regressions in
# future versions of igraph or networkx.
_parity_test()


def _neighbour_counts_in_window(
    ch_window: pd.DataFrame,
) -> dict[str, int]:
    """For each basename, count the *distinct* basenames it co-changes with
    inside ``ch_window``.

    Computed independently of the full graph so that pre-``t`` recency
    windows do not inherit older co-change history.
    """
    by_commit = ch_window.groupby("COMMIT_HASH")["basename"].unique()
    neighbours: dict[str, set[str]] = {}
    for files in by_commit:
        if files is None or len(files) < 2:
            continue
        for i in range(len(files)):
            fi = files[i]
            s = neighbours.setdefault(fi, set())
            for j in range(len(files)):
                if i == j:
                    continue
                s.add(files[j])
    return {f: len(s) for f, s in neighbours.items()}


def build_graph_features_for_project(
    project_id: str,
    snapshot: pd.Timestamp,
    changes: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Per-basename co-change graph features for one project at snapshot ``t``.

    Parameters
    ----------
    project_id :
        Project key to compute for.
    snapshot :
        Snapshot timestamp ``t`` (UTC-aware). Only commits with
        ``DATE <= t`` participate in the graph.
    changes :
        ``clean_git_commits_changes`` DataFrame with at least
        ``PROJECT_ID``, ``COMMIT_HASH``, ``DATE``, ``basename``.

    Returns
    -------
    DataFrame with columns ``project_id``, ``basename`` and all of
    :data:`GRAPH_FEATURE_COLS`. Every basename in the project's
    snapshot-time universe gets a row; files that never co-changed get
    zeros.
    """
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

    # ----- per-node aggregates -----
    _ts = _time.time()
    degree = dict(G.degree())
    strength_sum = dict(G.degree(weight="weight"))

    strength_max: dict[str, int] = {}
    strength_mean: dict[str, float] = {}
    for node in G.nodes():
        edges = G[node]
        if not edges:
            strength_max[node] = 0
            strength_mean[node] = 0.0
            continue
        weights = [edges[nb]["weight"] for nb in edges]
        strength_max[node] = int(max(weights))
        strength_mean[node] = float(np.mean(weights))
    _log(f"degree+strength ({_time.time()-_ts:.1f}s)")

    _ts = _time.time()
    centralities = _compute_centralities_igraph(G)
    betweenness = centralities["betweenness"]
    closeness = centralities["closeness"]
    clustering = centralities["clustering"]
    pagerank = centralities["pagerank"]
    _log(f"centralities (igraph) ({_time.time()-_ts:.1f}s)")

    _ts = _time.time()
    t_30 = t - pd.Timedelta(days=30)
    t_90 = t - pd.Timedelta(days=90)
    ch_30 = ch[ch["DATE"] > t_30]
    ch_90 = ch[ch["DATE"] > t_90]
    nb_30 = _neighbour_counts_in_window(ch_30[["COMMIT_HASH", "basename"]])
    nb_90 = _neighbour_counts_in_window(ch_90[["COMMIT_HASH", "basename"]])
    _log(f"neighbour windows ({_time.time()-_ts:.1f}s)")

    # ----- assemble into per-basename DataFrame -----
    rows = []
    for node in G.nodes():
        rows.append(
            {
                "basename": node,
                "cocg_degree": int(degree.get(node, 0)),
                "cocg_strength_sum": float(strength_sum.get(node, 0.0)),
                "cocg_strength_mean": float(strength_mean.get(node, 0.0)),
                "cocg_strength_max": float(strength_max.get(node, 0.0)),
                "cocg_betweenness": float(betweenness.get(node, 0.0)),
                "cocg_closeness": float(closeness.get(node, 0.0)),
                "cocg_clustering_coef": float(clustering.get(node, 0.0)),
                "cocg_pagerank": float(pagerank.get(node, 0.0)),
                "cocg_neighbour_count_30d": int(nb_30.get(node, 0)),
                "cocg_neighbour_count_90d": int(nb_90.get(node, 0)),
            }
        )
    feats = pd.DataFrame(rows)

    # Merge onto the snapshot universe so files that never co-changed
    # get zeros (universe = all files that exist at t, not just those
    # in the graph).
    out = universe.merge(feats, on="basename", how="left")

    zero_int_cols = ("cocg_degree", "cocg_neighbour_count_30d", "cocg_neighbour_count_90d")
    for col in zero_int_cols:
        out[col] = out[col].fillna(0).astype("int64")
    zero_float_cols = (
        "cocg_strength_sum",
        "cocg_strength_mean",
        "cocg_strength_max",
        "cocg_betweenness",
        "cocg_closeness",
        "cocg_clustering_coef",
        "cocg_pagerank",
    )
    for col in zero_float_cols:
        out[col] = out[col].fillna(0.0).astype(float)

    return out
