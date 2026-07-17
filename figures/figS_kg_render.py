#!/usr/bin/env python3
"""
Artistic-but-faithful rendering of the real biophysical knowledge graph.

Loads the ACTUAL graph (knowledge_graph_reprofiled_32/kg_{nodes,edges}.csv),
lays it out with a force-directed algorithm, and colours nodes by type. Nothing
is invented: every node and edge is drawn from the data. Hotspots form the outer
constellation; the semantic backbone (kingdom / organism / element / parameter)
is the bright, structured core.

  python figures/figS_kg_render.py
  -> figures/figS_kg_render.{pdf,png}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import pandas as pd
import numpy as np

KGDIR = "knowledge_graph_reprofiled_32"
INK, MUTED = "#1a1a19", "#6b7280"

# node-type palette (distinct hues, print-safe)
COL = {
    "kingdom":   "#5b21b6",   # deep violet
    "organism":  "#c2418c",   # magenta  (paper KG colour)
    "element":   "#1f6fd0",   # blue     (elements/profiles)
    "parameter": "#0e9384",   # teal
    "hotspot":   "#f4b8d3",   # pale pink cloud
}
SIZE = {"kingdom": 620, "organism": 300, "element": 360, "parameter": 470, "hotspot": 7}
ZO   = {"hotspot": 1, "organism": 3, "element": 3, "parameter": 3, "kingdom": 4}

nodes = pd.read_csv(f"{KGDIR}/kg_nodes.csv")
edges = pd.read_csv(f"{KGDIR}/kg_edges.csv")

G = nx.Graph()
ntype = {}
for _, r in nodes.iterrows():
    G.add_node(r["node_id"]); ntype[r["node_id"]] = r["node_type"]
for _, r in edges.iterrows():
    if r["source"] in ntype and r["target"] in ntype:
        G.add_edge(r["source"], r["target"], etype=r["edge_type"])

# force-directed layout (fixed seed -> reproducible)
pos = nx.spring_layout(G, k=0.42, iterations=170, seed=7)

fig = plt.figure(figsize=(7.6, 7.4))
ax = fig.add_axes([0.0, 0.0, 1.0, 0.93]); ax.axis("off")

# ---- edges: hotspot edges as faint web, backbone edges bright ----
hot_e, back_e = [], []
for u, v, d in G.edges(data=True):
    (hot_e if (ntype[u] == "hotspot" or ntype[v] == "hotspot") else back_e).append((u, v))
nx.draw_networkx_edges(G, pos, edgelist=hot_e, ax=ax, edge_color="#f0c3d9",
                       width=0.25, alpha=0.35)
nx.draw_networkx_edges(G, pos, edgelist=back_e, ax=ax, edge_color="#b06", width=0.9, alpha=0.65)

# ---- nodes, drawn type by type (hotspots underneath) ----
for t in ["hotspot", "organism", "element", "parameter", "kingdom"]:
    nl = [n for n in G.nodes if ntype[n] == t]
    nx.draw_networkx_nodes(G, pos, nodelist=nl, node_color=COL[t], node_size=SIZE[t],
                           edgecolors="white" if t != "hotspot" else "none",
                           linewidths=0.6 if t != "hotspot" else 0, ax=ax,
                           alpha=0.95 if t != "hotspot" else 0.7)

# ---- label the semantic backbone nodes lightly ----
for n in G.nodes:
    if ntype[n] in ("kingdom", "parameter"):
        name = str(n).split(":")[-1]
        ax.text(pos[n][0], pos[n][1], name, fontsize=5.6, ha="center", va="center",
                color="white", fontweight="bold", zorder=6)

ax.set_xlim(min(p[0] for p in pos.values()) - 0.08, max(p[0] for p in pos.values()) + 0.08)
ax.set_ylim(min(p[1] for p in pos.values()) - 0.08, max(p[1] for p in pos.values()) + 0.08)

# ---- title + legend ----
fig.text(0.02, 0.965, "The biophysical knowledge graph", fontsize=13.5, fontweight="bold", color=INK)
fig.text(0.02, 0.937, "1,757 nodes · 7,265 edges · force-directed layout of the actual 32-organism graph",
         fontsize=8.6, color=MUTED)

handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=COL[t],
                  markeredgecolor="white", markersize=ms, label=f"{lab} ({c})")
           for t, ms, lab, c in [
               ("kingdom", 11, "kingdom", 4), ("organism", 8, "organism", 32),
               ("element", 8.5, "element", 14), ("parameter", 9.5, "parameter", 7),
               ("hotspot", 5, "hotspot", "1,700")]]
leg = ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=7.6,
                handletextpad=0.5, borderpad=0.7, labelspacing=0.55, title="node types")
leg.get_title().set_fontsize(7.8); leg.get_title().set_fontweight("bold")
leg.get_frame().set_edgecolor("#d6dae0"); leg.get_frame().set_alpha(0.95)

ax.text(0.985, 0.015, "relations: BELONGS_TO · HAS_ELEMENT · PRECEDES · MEASURED_BY · "
        "CORRELATES_WITH · HOTSPOT_IN_{ORGANISM, ELEMENT, PARAMETER}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0, color=MUTED)

fig.savefig("figures/figS_kg_render.pdf")
fig.savefig("figures/figS_kg_render.png", dpi=200)
print("wrote figures/figS_kg_render.{pdf,png}")
