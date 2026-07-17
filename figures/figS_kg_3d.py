#!/usr/bin/env python3
"""
3D layout of the real biophysical knowledge graph.

Computes ONE force-directed 3D layout from the actual CSVs and emits:
  - figures/figS_kg_3d.{png,pdf}      a static 3D still (one viewing angle)
  - figures/kg_3d_data.json           node positions + typed edges for the
                                      interactive HTML (same layout, same seed)

Parameter CORRELATES_WITH edges are sign-coloured (blue = +r, red = -r) so
proximity is not mistaken for similarity: an anti-correlated pair (e.g.
bbone-inter, r=-0.56) reads as an opposition, not a kinship.

Integrity: layout is emergent and seed-dependent; distances are NOT quantitative.

  python figures/figS_kg_3d.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D
import networkx as nx
import pandas as pd
import numpy as np
import json

KGDIR = "knowledge_graph_reprofiled_32"
INK, MUTED = "#1a1a19", "#6b7280"
COL = {"kingdom": "#5b21b6", "organism": "#c2418c", "element": "#1f6fd0",
       "parameter": "#0e9384", "hotspot": "#f4b8d3"}
SIZE = {"kingdom": 190, "organism": 70, "element": 90, "parameter": 130, "hotspot": 3}
POS_C, NEG_C, BONE_C, HOT_C = "#1f6fd0", "#d1345b", "#b0367f", "#f0c3d9"

nodes = pd.read_csv(f"{KGDIR}/kg_nodes.csv")
edges = pd.read_csv(f"{KGDIR}/kg_edges.csv")

G = nx.Graph(); ntype = {}; nname = {}
for _, r in nodes.iterrows():
    G.add_node(r["node_id"]); ntype[r["node_id"]] = r["node_type"]
    nname[r["node_id"]] = str(r["node_id"]).split(":")[-1]
for _, r in edges.iterrows():
    if r["source"] in ntype and r["target"] in ntype:
        rv = r.get("pearson_r", None)
        try:
            rv = float(rv)
        except Exception:
            rv = None
        G.add_edge(r["source"], r["target"], etype=r["edge_type"], r=rv)

pos = nx.spring_layout(G, dim=3, k=0.55, iterations=160, seed=7)

def kind(u, v, d):
    if ntype[u] == "hotspot" or ntype[v] == "hotspot":
        return "hotspot"
    if d.get("etype") == "CORRELATES_WITH":
        return "corr_pos" if (d.get("r") or 0) >= 0 else "corr_neg"
    return "backbone"

# ---------- static render ----------
fig = plt.figure(figsize=(7.8, 7.6))
ax = fig.add_subplot(111, projection="3d")
ax.set_box_aspect((1, 1, 1)); ax.axis("off")
ax.view_init(elev=18, azim=32)

def seg(elist):
    return [[(pos[u][0], pos[u][1], pos[u][2]), (pos[v][0], pos[v][1], pos[v][2])] for u, v in elist]

buckets = {"hotspot": [], "backbone": [], "corr_pos": [], "corr_neg": []}
for u, v, d in G.edges(data=True):
    buckets[kind(u, v, d)].append((u, v))

# hotspot edges omitted in the static still (they occlude the core); kept in the
# interactive version where depth can be rotated apart.
ax.add_collection3d(Line3DCollection(seg(buckets["backbone"]), colors=BONE_C, linewidths=0.6, alpha=0.4))
ax.add_collection3d(Line3DCollection(seg(buckets["corr_pos"]), colors=POS_C, linewidths=1.8, alpha=0.95))
ax.add_collection3d(Line3DCollection(seg(buckets["corr_neg"]), colors=NEG_C, linewidths=1.8, alpha=0.95))

for t in ["hotspot", "organism", "element", "parameter", "kingdom"]:
    nl = [n for n in G.nodes if ntype[n] == t]
    xs = [pos[n][0] for n in nl]; ys = [pos[n][1] for n in nl]; zs = [pos[n][2] for n in nl]
    ax.scatter(xs, ys, zs, s=(2 if t == "hotspot" else SIZE[t]), c=COL[t], depthshade=True,
               edgecolors="white" if t != "hotspot" else "none",
               linewidths=0.4 if t != "hotspot" else 0, alpha=0.95 if t != "hotspot" else 0.30)

# label only the interpretable anchors, nudged so they clear their node
for n in G.nodes:
    if ntype[n] in ("kingdom", "parameter"):
        ax.text(pos[n][0], pos[n][1], pos[n][2] + 0.03, nname[n], fontsize=6.4, color=INK,
                fontweight="bold", ha="center", va="bottom", zorder=10,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75))

# zoom into the semantic core (hotspot halo may clip — intended)
bb = np.array([pos[n] for n in G.nodes if ntype[n] != "hotspot"])
c, rad = bb.mean(0), (bb.max(0) - bb.min(0)).max() * 0.62
ax.set_xlim(c[0] - rad, c[0] + rad); ax.set_ylim(c[1] - rad, c[1] + rad); ax.set_zlim(c[2] - rad, c[2] + rad)

fig.text(0.02, 0.965, "The biophysical knowledge graph — 3D", fontsize=13.5, fontweight="bold", color=INK)
fig.text(0.02, 0.938, "1,757 nodes · 7,265 edges · one force-directed 3D layout (static view; rotate the "
         "interactive version)", fontsize=8.2, color=MUTED)

handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=COL[t], markeredgecolor="white",
                  markersize=ms, label=f"{lab} ({c})")
           for t, ms, lab, c in [("kingdom", 10, "kingdom", 4), ("organism", 7, "organism", 32),
                                  ("element", 8, "element", 14), ("parameter", 9, "parameter", 7),
                                  ("hotspot", 4.5, "hotspot", "1,700")]]
handles += [Line2D([0], [0], color=POS_C, lw=2, label="corr  +r"),
            Line2D([0], [0], color=NEG_C, lw=2, label="corr  −r")]
leg = ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=7.0,
                labelspacing=0.4, borderpad=0.6)
leg.get_frame().set_edgecolor("#d6dae0")

fig.text(0.5, 0.02, "Layout reflects connectivity, not similarity — distances are emergent and not "
         "quantitative.", ha="center", fontsize=6.4, color=MUTED, style="italic")

fig.savefig("figures/figS_kg_3d.pdf")
fig.savefig("figures/figS_kg_3d.png", dpi=200)
print("wrote figures/figS_kg_3d.{pdf,png}")

# ---------- JSON for the interactive page ----------
P = np.array([pos[n] for n in G.nodes])
P = (P - P.mean(0)) / (np.abs(P).max() + 1e-9)   # centre + scale to ~[-1,1]
idx = {n: i for i, n in enumerate(G.nodes)}
out_nodes = [{"t": ntype[n], "n": nname[n] if ntype[n] != "hotspot" else "",
              "d": int(G.degree[n]),
              "x": round(float(P[idx[n]][0]), 4), "y": round(float(P[idx[n]][1]), 4),
              "z": round(float(P[idx[n]][2]), 4)} for n in G.nodes]
out_edges = [{"s": idx[u], "t": idx[v], "k": kind(u, v, d)} for u, v, d in G.edges(data=True)]
with open("figures/kg_3d_data.json", "w") as fh:
    json.dump({"nodes": out_nodes, "edges": out_edges,
               "counts": {"kingdom": 4, "organism": 32, "element": 14, "parameter": 7, "hotspot": 1700},
               "colors": COL}, fh)
print(f"wrote figures/kg_3d_data.json  ({len(out_nodes)} nodes, {len(out_edges)} edges)")
