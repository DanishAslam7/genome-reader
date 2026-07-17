#!/usr/bin/env python3
"""
Figure 6 - mechanism: which physical property encodes which element.

(a) drop-one channel ablation, 7 parameters x 5 elements, 3 seeds. DIVERGING scale
    centred on zero because the negatives are real (removing H-bond IMPROVES exon
    boundaries). Two hues + neutral gray midpoint; no hue at the midpoint.
(b) keep-only: every parameter alone collapses to chance -> the code is distributed.

  python figures/fig6_mechanism.py
  -> figures/fig6_mechanism.{pdf,png}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

# ── data: 3-seed mean / sd (param_ablation_out/*.json; seeds 42/123/777) ──────
PARAMS = ["bbone", "bp", "hbond", "inter", "intra", "sol", "stack"]
PNAMES = ["backbone", "base pair", "H-bond", "inter*", "intra", "solvation", "stacking"]
# Fixed input channel weights (Section 2.6 / S1). Shown on the y-axis so the two
# controls in Section 3.5 read straight off the figure: stack shares H-bond's ×2.0 but
# is worth ~0 at promoters; inter is weighted lowest (×1.0) yet dominates start codons.
WEIGHT = {"bbone": 1.4, "bp": 1.4, "hbond": 2.0, "inter": 1.0,
          "intra": 1.4, "sol": 1.4, "stack": 2.0}
YLAB = [f"{n}  ×{WEIGHT[p]}" for n, p in zip(PNAMES, PARAMS)]
ELEMS  = ["exon\nboundary", "gene\nboundary", "promoter", "start\ncodon", "stop\ncodon"]

MEAN = np.array([
    [0.339, 0.299, 0.037, 0.275, 0.206],   # bbone
    [0.157, 0.106, 0.008, 0.141, 0.174],   # bp
    [-0.109, 0.301, 0.447, 0.180, 0.197],  # hbond
    [0.103, -0.038, 0.151, 0.518, 0.199],  # inter
    [0.177, -0.043, 0.148, 0.379, 0.202],  # intra
    [0.348, 0.220, 0.140, 0.084, 0.286],   # sol
    [0.176, 0.037, 0.035, 0.110, 0.123],   # stack
])
SD = np.array([
    [0.016, 0.024, 0.012, 0.118, 0.102],
    [0.014, 0.029, 0.027, 0.047, 0.024],
    [0.024, 0.064, 0.058, 0.057, 0.037],
    [0.022, 0.029, 0.058, 0.107, 0.056],
    [0.034, 0.012, 0.040, 0.072, 0.036],
    [0.055, 0.071, 0.161, 0.022, 0.030],
    [0.018, 0.022, 0.031, 0.029, 0.020],
])
KEEP_M = {"bbone": 0.253, "bp": 0.213, "hbond": 0.191, "inter": 0.169,
          "intra": 0.199, "sol": 0.170, "stack": 0.251}
KEEP_S = {"bbone": 0.008, "bp": 0.041, "hbond": 0.022, "inter": 0.002,
          "intra": 0.044, "sol": 0.004, "stack": 0.013}
CHANCE = 0.20
CLEAN = (2, 2)   # hbond x promoter - the only cell with no error-bar overlap

INK, MUTED, GRID = "#1a1a19", "#6b7280", "#e5e7eb"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8.5, "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# diverging: blue (removing helps) -> neutral gray (no effect) -> orange (element depends on it)
cmap = LinearSegmentedColormap.from_list("div", [
    (0.00, "#1c5cab"), (0.35, "#86b6ef"), (0.50, "#eceef1"),
    (0.65, "#f2a878"), (0.85, "#eb6834"), (1.00, "#a83616"),
])
norm = TwoSlopeNorm(vmin=-0.12, vcenter=0.0, vmax=0.52)

fig = plt.figure(figsize=(7.4, 3.95))
gs = fig.add_gridspec(1, 2, width_ratios=[2.62, 1.0], wspace=0.34,
                      left=0.088, right=0.975, top=0.80, bottom=0.20)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])

# ── (a) heatmap ───────────────────────────────────────────────────────────────
axA.imshow(MEAN, cmap=cmap, norm=norm, aspect="auto")

def lum(rgba):
    r, g, b = rgba[:3]
    f = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)

for i in range(len(PARAMS)):
    for j in range(len(ELEMS)):
        v, s = MEAN[i, j], SD[i, j]
        txt = "#ffffff" if lum(cmap(norm(v))) < 0.42 else INK
        axA.text(j, i - 0.13, f"{'−' if v < 0 else ''}{abs(v):.3f}", ha="center", va="center",
                 fontsize=7.9, color=txt, fontweight="bold")
        axA.text(j, i + 0.21, f"± {s:.3f}", ha="center", va="center",
                 fontsize=6.2, color=txt, alpha=0.82)

# grid lines between cells (2px surface gap equivalent)
axA.set_xticks(np.arange(-0.5, len(ELEMS), 1), minor=True)
axA.set_yticks(np.arange(-0.5, len(PARAMS), 1), minor=True)
axA.grid(which="minor", color="white", linewidth=1.6)
axA.tick_params(which="minor", length=0)

# the one clean claim
axA.add_patch(Rectangle((CLEAN[1] - 0.5, CLEAN[0] - 0.5), 1, 1, fill=False,
                        edgecolor=INK, lw=2.0, zorder=5))
# (explained in the footnote rather than an in-plot annotation: every position
#  around this cell is occupied by a column header or a neighbouring cell)

axA.set_xticks(range(len(ELEMS))); axA.set_xticklabels(ELEMS, fontsize=8.2, color=INK)
axA.set_yticks(range(len(PARAMS))); axA.set_yticklabels(YLAB, fontsize=7.8, color=INK)
axA.xaxis.set_ticks_position("top")
axA.tick_params(axis="both", length=0, pad=5)
for s in axA.spines.values(): s.set_visible(False)
axA.set_title("a   Drop-one: accuracy lost when a parameter is removed",
              fontsize=9.2, color=INK, loc="left", pad=40, fontweight="bold")
axA.text(0, 1.20, "×n beside each parameter = its fixed input weight (Section 2.6): H-bond and "
                  "stacking ×2.0, inter ×1.0.", transform=axA.transAxes,
         fontsize=6.4, color=MUTED, va="bottom")

# colourbar
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axA,
                  orientation="horizontal", fraction=0.052, pad=0.075,
                  ticks=[-0.10, 0, 0.15, 0.30, 0.45])
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=6.8, colors=MUTED, length=2)
cb.set_label("← removing it HELPS      no effect      removing it HURTS →",
             fontsize=6.9, color=MUTED, labelpad=4)

# ── (b) keep-only ─────────────────────────────────────────────────────────────
order = sorted(PARAMS, key=lambda p: KEEP_M[p])
y = np.arange(len(order))
vals = [KEEP_M[p] for p in order]
errs = [KEEP_S[p] for p in order]
names = [PNAMES[PARAMS.index(p)] for p in order]

axB.barh(y, vals, height=0.62, color="#c9ced6", zorder=2)
axB.errorbar(vals, y, xerr=errs, fmt="none", ecolor=MUTED, elinewidth=0.9,
             capsize=2.2, capthick=0.9, zorder=3)
axB.axvline(CHANCE, color="#b91c1c", ls=(0, (4, 3)), lw=1.3, zorder=4)
# parked low, where the short bars leave the right of the chance line empty
axB.text(CHANCE + 0.005, 0.42, "chance\n0.20", fontsize=7.0, color="#b91c1c",
         ha="left", va="center", fontweight="bold", linespacing=1.25)

# labels clear the error-bar caps, never sit on them
for yi, v, e in zip(y, vals, errs):
    axB.text(v + e + 0.006, yi, f"{v:.3f}", va="center", ha="left", fontsize=7.2, color=INK)

axB.set_yticks(y); axB.set_yticklabels(names, fontsize=7.8, color=INK)
axB.set_xlim(0.15, 0.30)
axB.set_ylim(-0.62, len(order) - 0.38)
axB.set_xticks([0.15, 0.20, 0.25, 0.30])
axB.tick_params(axis="x", labelsize=7, colors=MUTED, length=3, width=0.7)
axB.tick_params(axis="y", length=0, pad=4)
axB.set_xlabel("element accuracy, this parameter alone", fontsize=7.4, color=MUTED, labelpad=5)
axB.grid(axis="x", color=GRID, lw=0.6, zorder=0)
axB.set_axisbelow(True)
for s in ("top", "right", "left"): axB.spines[s].set_visible(False)
axB.spines["bottom"].set_color(GRID)
axB.set_title("b   Keep-only: no parameter\n     carries the code alone",
              fontsize=9.2, color=INK, loc="left", pad=10, fontweight="bold")

fig.text(0.088, 0.055, "Outlined cell (H-bond × promoter, 0.447 ± 0.058) is the only one whose error bars clear every rival - every other top-ranked cell overlaps its runner-up.",
         fontsize=6.6, color=INK)
fig.text(0.088, 0.012, "* inter is the only tetranucleotide-derived parameter; the other six come from the trinucleotide table.",
         fontsize=6.6, color=MUTED)

for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig6_mechanism.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
print("wrote figures/fig6_mechanism.pdf / .png")
