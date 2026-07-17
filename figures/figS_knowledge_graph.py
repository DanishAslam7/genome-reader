#!/usr/bin/env python3
"""
Supplementary Figure - the biophysical knowledge graph and its context branch.

Shows (A) the KG schema with real counts for the 32-organism cohort
(knowledge_graph_reprofiled_32/kg_stats.json), (B) the context-only feature
extraction that deliberately excludes the true element node (anti-leak), and
(C) how the resulting 76-d vector fuses into the profile trunk.

All numbers are READ FROM THE ARTEFACTS:
  - node/edge counts: knowledge_graph_reprofiled_32/kg_stats.json
  - branch shape/feature_dim: train_taxonomy_multitask.py:1804-1861, 2478-2488
    and the ablation_both run_config (kg feature_dim = 76).

COLOUR CONTRACT (paper-wide): blue = profiles/trunk, magenta = knowledge graph,
red = leakage / caveat.

  python figures/figS_knowledge_graph.py
  -> figures/figS_knowledge_graph.{pdf,png}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BIO, KG, RED = "#2a78d6", "#c2418c", "#b91c1c"
INK, MUTED, RULE = "#1a1a19", "#6b7280", "#d6dae0"
KTINT, RTINT, PANEL, BTINT = "#fbe9f3", "#fdeceb", "#f7f8fa", "#eaf2fc"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig = plt.figure(figsize=(7.6, 9.4))
ax = fig.add_axes([0.015, 0.01, 0.97, 0.975])
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, fc="white", ec=RULE, lw=1.0, z=2, r=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def node(cx, cy, w, h, label, sub, fc, ec, tcol=INK, fs=8.2):
    box(cx - w / 2, cy - h / 2, w, h, fc=fc, ec=ec, lw=1.3, z=4, r=0.8)
    ax.text(cx, cy + (1.0 if sub else 0), label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=tcol, zorder=5)
    if sub:
        ax.text(cx, cy - 1.7, sub, ha="center", va="center", fontsize=6.6,
                color=MUTED, zorder=5)


def edge(p1, p2, label="", color=MUTED, lw=1.2, ls="-", curve=0.0, lab_dy=0.9,
         lab_fs=6.3, tcol=None, arrow=True):
    style = "-|>" if arrow else "-"
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls,
                                mutation_scale=8, shrinkA=6, shrinkB=6,
                                connectionstyle=f"arc3,rad={curve}"))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + lab_dy, label, ha="center", va="center", fontsize=lab_fs,
                color=tcol or color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


def panel(x, w, y, h, letter, title, ec=RULE):
    box(x, y, w, h, fc=PANEL, ec=ec, lw=1.0, r=1.0)
    ax.text(x + 1.6, y + h - 2.3, letter, fontsize=11.5, fontweight="bold", color=INK, va="center")
    ax.text(x + 5.4, y + h - 2.3, title, fontsize=9.0, fontweight="bold", color=INK, va="center")


# ═════════════ header ═════════════
ax.text(0, 98.6, "Supplementary Figure - the knowledge-graph context branch",
        fontsize=11.6, fontweight="bold", color=INK)
ax.text(0, 96.2, "Active in the canonical and profiles+sequence+KG configurations; withheld from the "
        "cross-kingdom transfer experiments,", fontsize=8.0, color=MUTED)
ax.text(0, 94.4, "because cohort-derived features are fitted using every organism and would leak "
        "across a held-out kingdom.", fontsize=8.0, color=MUTED)

# ═════════════ PANEL A - the graph ═════════════
panel(0.5, 99.0, 60.0, 32.0, "A", "The knowledge graph  ·  1,757 nodes, 7,265 edges (32-organism cohort)", ec=KG)

# taxonomy tier (Kingdom over Organism) + biophysics chain (Element -> Parameter),
# with the hotspot layer as a hub BELOW, so no edge crosses the graph.
K = (17, 86.0); O = (17, 76.5); E = (45, 76.5); P = (73, 76.5); H = (45, 66.5)
node(*K, 21, 6.0, "Kingdom", "4", KTINT, KG)
node(*O, 21, 6.0, "Organism", "32", KTINT, KG)
node(*E, 21, 6.0, "Element", "14", KTINT, KG)
node(*P, 21, 6.0, "Parameter", "7", KTINT, KG)
node(*H, 23, 6.0, "Hotspot", "1,700 derived", "white", KG, tcol=KG)

# taxonomy + chain
edge((O[0], O[1] + 3.0), (K[0], K[1] - 3.0), "BELONGS_TO", KG, 1.3)
edge((O[0] + 10.5, O[1]), (E[0] - 10.5, E[1]), "HAS_ELEMENT", KG, 1.3)
edge((E[0] + 10.5, E[1]), (P[0] - 10.5, P[1]), "MEASURED_BY", KG, 1.4)
ax.text((E[0] + P[0]) / 2, 71.4, "carries per-organism\nprofile statistics", ha="center",
        va="center", fontsize=5.7, color=MUTED, style="italic", linespacing=1.0,
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.9), zorder=6)
# self-relations, sitting on top of their nodes
edge((E[0] - 6, E[1] + 3.0), (E[0] + 6, E[1] + 3.0), "PRECEDES", KG, 1.1, curve=-0.6, lab_dy=1.8)
edge((P[0] - 6, P[1] + 3.0), (P[0] + 6, P[1] + 3.0), "CORRELATES_WITH", KG, 1.0, ls="--",
     curve=-0.6, lab_dy=1.8)
# hotspot hub -> organism / element / parameter (clean fan, no crossings)
edge((H[0] - 9, H[1] + 2.4), (O[0] + 4, O[1] - 3.0), "", KG, 0.9, ls=":", curve=-0.08)
edge((H[0], H[1] + 3.0), (E[0], E[1] - 3.0), "", KG, 0.9, ls=":")
edge((H[0] + 9, H[1] + 2.4), (P[0] - 4, P[1] - 3.0), "", KG, 0.9, ls=":", curve=0.08)
ax.text(H[0] + 14.5, H[1], "HOTSPOT_IN_{ORGANISM,\nELEMENT, PARAMETER}", ha="left", va="center",
        fontsize=5.9, color=KG)

# ═════════════ PANEL B - context-only extraction ═════════════
panel(0.5, 99.0, 30.0, 28.0, "B", "Per-window features - context nodes only (anti-leak)", ec=RED)

wy = 53.5
ax.text(4.0, wy, "one training window:", fontsize=7.6, color=INK, fontweight="bold")
node(24, wy - 4.6, 20, 5.4, "kingdom node", "used", KTINT, KG, fs=7.2)
node(24, wy - 11.2, 20, 5.4, "organism node", "used", KTINT, KG, fs=7.2)
node(24, wy - 17.8, 22, 5.4, "true element node", "EXCLUDED", RTINT, RED, tcol=RED, fs=7.2)
# cross out element
ax.plot([13.4, 34.6], [wy - 20.5, wy - 15.1], color=RED, lw=1.6, zorder=6)
ax.text(24, wy - 22.6, "including it would leak the label", fontsize=6.6, color=RED,
        ha="center", va="center")

# feature composition box
box(50, wy - 20.0, 48, 20.5, fc="white", ec=RULE, lw=1.0, r=0.9)
ax.text(52, wy - 1.9, "per node, keep:", fontsize=7.6, fontweight="bold", color=INK)
ax.text(52, wy - 4.7, "• 10 structural  (per-relation degree, genomic order,", fontsize=6.9, color=INK)
ax.text(54, wy - 6.9, "n_organisms / n_elements / n_sequences, …)", fontsize=6.9, color=MUTED)
ax.text(52, wy - 9.6, "• 28 profile summary  (mean / std / peak-pos /", fontsize=6.9, color=INK)
ax.text(54, wy - 11.8, "peak-val  ×  7 biophysical parameters)", fontsize=6.9, color=MUTED)
ax.text(52, wy - 14.6, "drop identity cols (kingdom_idx, organism_idx)", fontsize=6.7, color=RED)
box(52, wy - 19.4, 44, 3.4, fc=KTINT, ec=KG, lw=1.2, r=0.7)
ax.text(74, wy - 17.7, "kg_features  =  38 × {kingdom, organism}  =  76-d",
        ha="center", va="center", fontsize=7.6, fontweight="bold", color=KG)
edge((35, wy - 11.2), (50, wy - 10.0), "", KG, 1.2)

# ═════════════ PANEL C - fusion ═════════════
panel(0.5, 99.0, 6.0, 22.0, "C", "Fusion into the profile trunk", ec=KG)

fy = 19.0
# KG branch chain
chain = [(4, "kg_features", "76", KTINT, KG),
         (20, "LayerNorm", "", "white", KG),
         (33, "Dense", "96", "white", KG),
         (44, "Dropout", "", "white", KG),
         (55, "Dense", "48", "white", KG)]
for i, (cx, lab, sub, fc, ec) in enumerate(chain):
    node(cx + 4, fy, 12 if lab != "kg_features" else 13, 5.2, lab, sub, fc, ec,
         tcol=(KG if fc == KTINT else INK), fs=7.0)
    if i:
        edge((chain[i-1][0] + 10, fy), (cx - 2, fy), "", KG, 1.1)

# profile trunk box
node(24, fy - 9.0, 26, 5.2, "profile trunk", "256-d shared", BTINT, BIO, tcol=BIO, fs=7.4)
# concat
node(72, fy - 4.5, 12, 5.2, "⊕ concat", "", "white", INK, fs=7.4)
edge((61, fy), (72, fy - 1.9), "", KG, 1.2)
edge((37, fy - 9.0), (72, fy - 6.6), "", BIO, 1.2)
node(88, fy - 4.5, 15, 5.2, "Dense", "192", "white", INK, fs=7.4)
edge((78, fy - 4.5), (80.5, fy - 4.5), "", INK, 1.2)
# to heads
ax.text(88, fy - 9.4, "→ shared representation → element head (+ aux heads)",
        ha="right", va="center", fontsize=6.9, color=MUTED)

ax.text(4, 2.4, "Result: the KG branch adds +1.38 pp element accuracy in-distribution (residual over "
        "sequence), contingent on the sequence branch;", fontsize=6.9, color=INK)
ax.text(4, 0.7, "it is switched off wherever a kingdom is held out, so no reported transfer number "
        "depends on it.", fontsize=6.9, color=INK)

fig.savefig("figures/figS_knowledge_graph.pdf")
fig.savefig("figures/figS_knowledge_graph.png", dpi=200)
print("wrote figures/figS_knowledge_graph.{pdf,png}")
