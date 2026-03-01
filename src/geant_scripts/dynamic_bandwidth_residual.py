#!/usr/bin/env python3
"""
Residual bandwidth bottlenecks (Gbps) from Switzerland to selected countries, per 15-min traffic matrix step.

Output format (JSON)
--------------------
{
  "0": { "Italy": 12.34, "Germany": 8.91, ... },
  "1": { "Italy": 11.22, "Germany": 9.10, ... },
  ...
}

Where:
- outer key = timestep index as a string ("0", "1", ...)
- inner key = destination country (no "Switzerland" key)
- value = bottleneck residual bandwidth along the hop-shortest path, in Gbps

Dependencies:
  pip install networkx
"""

from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob
from typing import Any, Dict, List, Tuple

import networkx as nx


# ----------------------------
# Topology parsing (GML-like)
# ----------------------------

def _parse_scalar(s: str) -> Any:
    s = s.strip()
    if not s:
        return ""
    if s.startswith('"'):
        return s.strip().strip('"')
    try:
        if any(c in s for c in (".", "e", "E")):
            return float(s)
        return int(s)
    except ValueError:
        return s


def parse_gml_nodes_edges(path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    stack: List[str] = []
    current: Dict[str, Any] | None = None

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if line == "graph [":
                stack.append("graph")
                continue
            if line == "node [":
                stack.append("node")
                current = {}
                continue
            if line == "edge [":
                stack.append("edge")
                current = {}
                continue

            if line == "]":
                if not stack:
                    continue
                ended = stack.pop()
                if ended == "node" and current is not None:
                    nodes.append(current)
                    current = None
                elif ended == "edge" and current is not None:
                    edges.append(current)
                    current = None
                continue

            if not stack or stack[-1] not in ("node", "edge"):
                continue

            if " " not in line:
                continue

            key, rest = line.split(None, 1)
            if current is None:
                current = {}
            current[key] = _parse_scalar(rest)

    return nodes, edges


def build_capacity_graph(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> nx.Graph:
    """
    Undirected topology graph with edge attribute:
      capacity_bps: float
    """
    G = nx.Graph()

    for n in nodes:
        if "id" not in n:
            continue
        nid = int(n["id"])
        G.add_node(nid, **n)

    for e in edges:
        if "source" not in e or "target" not in e:
            continue
        u, v = int(e["source"]), int(e["target"])
        if u not in G or v not in G:
            continue

        if "LinkSpeedRaw" not in e:
            raise ValueError("Topology edge missing LinkSpeedRaw; cannot determine bandwidth.")
        cap = float(e["LinkSpeedRaw"])

        # If duplicates exist, keep maximum capacity
        if G.has_edge(u, v):
            G[u][v]["capacity_bps"] = max(float(G[u][v]["capacity_bps"]), cap)
        else:
            G.add_edge(u, v, capacity_bps=cap)

    return G


# ----------------------------
# Traffic-matrix parsing
# ----------------------------

# Matches ints, floats, scientific notation
_NUM = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def parse_tm_as_last_square_matrix(path: str, n_nodes: int) -> List[List[float]]:
    """
    Extract all numeric tokens and take the LAST (n_nodes^2) values as the matrix.
    Robust to headers containing timestamps and other numbers.

    Returns: tm[i][j] for i,j in 0..n_nodes-1
    """
    text = open(path, "r", encoding="utf-8", errors="replace").read()
    tokens = [float(m.group(0)) for m in _NUM.finditer(text)]

    need = n_nodes * n_nodes
    if len(tokens) < need:
        raise ValueError(
            f"TM file {path} contains {len(tokens)} numeric tokens but needs at least {need} "
            f"for a {n_nodes}x{n_nodes} matrix."
        )

    vals = tokens[-need:]
    tm = [vals[i * n_nodes : (i + 1) * n_nodes] for i in range(n_nodes)]
    return tm


# ----------------------------
# Residual bandwidth computation
# ----------------------------

def norm_country(s: str) -> str:
    return " ".join(s.strip().lower().split())


def country_to_node_id(G: nx.Graph) -> Dict[str, int]:
    m: Dict[str, int] = {}
    for nid, data in G.nodes(data=True):
        if "Country" in data and str(data["Country"]).strip():
            m[norm_country(str(data["Country"]))] = nid
    return m


def all_pairs_loads_on_hop_shortest_paths(
    G: nx.Graph,
    tm_per_second: List[List[float]],
    node_order: List[int],
) -> Dict[Tuple[int, int], float]:
    """
    Accumulate per-link loads by routing each OD along a hop-count shortest path.
    Returns dict keyed by undirected edge (min(u,v), max(u,v)) -> load_bps.
    """
    load: Dict[Tuple[int, int], float] = {}
    n = len(node_order)

    for i in range(n):
        src = node_order[i]
        for j in range(n):
            if i == j:
                continue
            dst = node_order[j]
            d = float(tm_per_second[i][j])
            if d == 0.0:
                continue

            path = nx.shortest_path(G, source=src, target=dst)  # hop count
            for a, b in zip(path, path[1:]):
                e = (a, b) if a < b else (b, a)
                load[e] = load.get(e, 0.0) + d

    return load


def residuals_from_loads(G: nx.Graph, load: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    """
    residual_bps(edge) = capacity_bps - load_bps
    keyed by undirected edge (min(u,v), max(u,v))
    """
    resid: Dict[Tuple[int, int], float] = {}
    for u, v, data in G.edges(data=True):
        e = (u, v) if u < v else (v, u)
        cap = float(data["capacity_bps"])
        ld = float(load.get(e, 0.0))
        resid[e] = cap - ld
    return resid


def bottleneck_residual_on_hop_shortest_path(
    G: nx.Graph,
    resid: Dict[Tuple[int, int], float],
    src: int,
    dst: int,
) -> float:
    """
    Compute hop-shortest path in G, then return bottleneck residual along that path.
    """
    path = nx.shortest_path(G, source=src, target=dst)
    if len(path) < 2:
        return float("inf")

    vals = []
    for a, b in zip(path, path[1:]):
        e = (a, b) if a < b else (b, a)
        vals.append(float(resid[e]))

    return min(vals)


def bps_to_gbps(x: float) -> float:
    return x / 1e9


DEFAULT_TARGETS = [
    "Italy",
    "Germany",
    "France",
    "Spain",
    "United Kingdom",
    "Austria",
    "Netherlands",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", required=True, help="Path to geant_gml.txt")
    ap.add_argument("--tm-glob", required=True, help='Glob for TM files, e.g. "IntraTM-*.xml"')
    ap.add_argument("--n-steps", type=int, default=None, help="Process only first n files after sorting.")
    ap.add_argument("--src-country", default="Switzerland", help="Default: Switzerland")
    ap.add_argument("--dst-countries", nargs="*", default=DEFAULT_TARGETS)
    ap.add_argument("--out", default="residuals.json", help="JSON output path")
    ap.add_argument("--indent", type=int, default=2, help="JSON indent level (default 2)")
    args = ap.parse_args()

    # Topology
    nodes, edges = parse_gml_nodes_edges(args.topology)
    G = build_capacity_graph(nodes, edges)

    node_ids = sorted(G.nodes())
    n = len(node_ids)

    # Country -> node id mapping
    c2id = country_to_node_id(G)
    src_key = norm_country(args.src_country)
    if src_key not in c2id:
        raise SystemExit(f"Source country not found in topology Country fields: {args.src_country}")
    src_id = c2id[src_key]

    dst_list: List[Tuple[str, int]] = []
    for c in args.dst_countries:
        k = norm_country(c)
        if k not in c2id:
            raise SystemExit(f"Destination country not found in topology Country fields: {c}")
        dst_list.append((c, c2id[k]))

    # TM files
    tm_files = sorted(glob(args.tm_glob))
    if not tm_files:
        raise SystemExit(f"No traffic matrix files matched: {args.tm_glob}")
    if args.n_steps is not None:
        tm_files = tm_files[: args.n_steps]

    # Results: step -> {dst_country: bottleneck_gbps}
    results: Dict[str, Dict[str, float]] = {}

    for step, tm_path in enumerate(tm_files):
        tm_15min = parse_tm_as_last_square_matrix(tm_path, n_nodes=n)

        # Convert to per-second
        tm_per_sec = [[float(x) / 900.0 for x in row] for row in tm_15min]

        # Compute loads and residuals
        load = all_pairs_loads_on_hop_shortest_paths(G, tm_per_sec, node_order=node_ids)
        resid = residuals_from_loads(G, load)

        step_dict: Dict[str, float] = {}
        for dst_country, dst_id in dst_list:
            bottleneck_bps = bottleneck_residual_on_hop_shortest_path(G, resid, src_id, dst_id)
            step_dict[dst_country] = bps_to_gbps(bottleneck_bps)

        results[str(step)] = step_dict

    # Write JSON
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=args.indent, sort_keys=True)

    print(f"Wrote {len(results)} steps to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())