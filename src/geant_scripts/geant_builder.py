#!/usr/bin/env python3
"""
Plot the graph from geant_gml.txt:
- node labels = node "country" attribute (fallbacks included)
- edge labels = bandwidth (from LinkSpeedRaw, assumed bps)

Dependencies:
  pip install networkx matplotlib
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx


def parse_scalar(s: str) -> Any:
    s = s.strip()
    if not s:
        return ""

    if s.startswith('"'):
        if len(s) >= 2 and s.endswith('"'):
            return s[1:-1]
        return s.strip('"')

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
        for raw_line in f:
            line = raw_line.strip()
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
            current[key] = parse_scalar(rest)

    return nodes, edges


def get_node_country(n: Dict[str, Any]) -> str:
    # Your file’s key is likely "Country" or "country"; support both + a few common variants.
    for k in ("country", "Country", "COUNTRY"):
        if k in n and str(n[k]).strip():
            return str(n[k]).strip()
    # Fallbacks
    if "label" in n and str(n["label"]).strip():
        return str(n["label"]).strip()
    if "id" in n:
        return str(n["id"])
    return "?"


def get_node_pos(n: Dict[str, Any]) -> Tuple[float, float] | None:
    # Many Topology Zoo GMLs use Longitude/Latitude (sometimes lowercase).
    lon = None
    lat = None
    for lk in ("Longitude", "longitude", "Lon", "lon", "x"):
        if lk in n:
            lon = n[lk]
            break
    for ak in ("Latitude", "latitude", "Lat", "lat", "y"):
        if ak in n:
            lat = n[ak]
            break
    if lon is None or lat is None:
        return None
    try:
        return (float(lon), float(lat))
    except (TypeError, ValueError):
        return None


def build_graph(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    directed: bool = False,
    combine: str = "max",
) -> nx.Graph:
    G = nx.DiGraph() if directed else nx.Graph()

    # add nodes
    for n in nodes:
        if "id" not in n:
            continue
        nid = int(n["id"])
        G.add_node(nid, **n)

    def combine_weight(old: float, new: float) -> float:
        if combine == "last":
            return new
        if combine == "sum":
            return old + new
        return max(old, new)  # default max

    # add edges
    for e in edges:
        if "source" not in e or "target" not in e:
            continue
        u = int(e["source"])
        v = int(e["target"])
        if u not in G or v not in G:
            continue

        if "LinkSpeedRaw" in e:
            w = float(e["LinkSpeedRaw"])
        else:
            # fallback if needed
            sp = float(e.get("LinkSpeed", 0.0))
            unit = str(e.get("LinkSpeedUnits", "")).upper()
            mult = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}.get(unit, 1.0)
            w = sp * mult

        if G.has_edge(u, v):
            G[u][v]["weight"] = combine_weight(float(G[u][v].get("weight", 0.0)), w)
        else:
            G.add_edge(u, v, **e, weight=w)

    return G


def fmt_bandwidth(bps: float, units: str) -> str:
    if units == "bps":
        return f"{bps:.0f} bps"
    if units == "Mbps":
        return f"{bps/1e6:g} Mbps"
    if units == "Gbps":
        return f"{bps/1e9:g} Gbps"
    return str(bps)


def plot_graph(G: nx.Graph, units: str = "Gbps", out: str | None = None, show: bool = True) -> None:
    # positions: use lon/lat if present, else spring layout
    pos_geo: Dict[int, Tuple[float, float]] = {}
    for nid, data in G.nodes(data=True):
        p = get_node_pos(data)
        if p is not None:
            pos_geo[nid] = p

    if len(pos_geo) == G.number_of_nodes() and G.number_of_nodes() > 0:
        pos = pos_geo
    else:
        pos = nx.spring_layout(G, seed=7)

    node_labels = {nid: get_node_country(data) for nid, data in G.nodes(data=True)}
    edge_labels = {(u, v): fmt_bandwidth(float(d.get("weight", 0.0)), units) for u, v, d in G.edges(data=True)}

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="#dbeafe", edgecolors="#1f2937", linewidths=1.0)
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color="#64748b", alpha=0.85)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_color="#111827")

    # Edge labels can get cluttered on dense graphs; this draws them all as requested.
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="#111827")

    plt.axis("off")
    plt.tight_layout()

    if out:
        plt.savefig(out, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to geant_gml.txt")
    ap.add_argument("--directed", action="store_true", help="Treat edges as directed.")
    ap.add_argument("--combine", choices=["max", "sum", "last"], default="max",
                    help="How to combine multiple edges between the same pair.")
    ap.add_argument("--units", choices=["bps", "Mbps", "Gbps"], default="Gbps",
                    help="Units to display for edge labels.")
    ap.add_argument("--out", default=None, help="Save plot to this file (e.g., graph.png).")
    ap.add_argument("--no-show", action="store_true", help="Do not open a window; only save if --out is set.")
    args = ap.parse_args()

    nodes, edges = parse_gml_nodes_edges(args.path)
    G = build_graph(nodes, edges, directed=args.directed, combine=args.combine)
    plot_graph(G, units=args.units, out=args.out, show=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())