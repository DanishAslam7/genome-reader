#!/usr/bin/env python3
"""
Figure 3 — cross-kingdom transfer: biophysics ties sequence.

Leave-one-kingdom-out, 3 seeds, both arms, all leak-safe + transductive-norm.
The claim is a TIE (0.5422 +/- 0.0044 vs 0.5465 +/- 0.0038), so the figure must
make "these are the same" legible — paired points per kingdom with seed spread
shown, not bars that invite reading a winner.

Numbers verified from each run's metrics.json (see RESULTS_LEDGER.md).

  python figures/fig3_loko_tie.py
  -> figures/fig3_loko_tie.{pdf,png}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ── data: per kingdom, per seed (42, 123, 777) ────────────────────────────────
BIO = {
    "plantae":  [0.6397, 0.6566, 0.6354],
    "animalia": [0.6058, 0.5922, 0.5943],
    "fungi":    [0.4928, 0.4881, 0.4697],   # all warmup=8
    "protista": [0.4457, 0.4357, 0.4501],
}
SEQ = {
    "plantae":  [0.6447, 0.6411, 0.6371],
    "animalia": [0.5990, 0.5920, 0.5934],
    "fungi":    [0.5142, 0.4959, 0.5048],
    "protista": [0.4455, 0.4506, 0.4397],
}
ORDER = ["plantae", "animalia", "fungi", "protista"]      # hardest last
CHANCE = 0.20

BIO_C, SEQ_C = "#2a78d6", "#008300"          # documented categorical slots 1-2
INK, MUTED, GRID = "#1a1a19", "#6b7280", "#e5e7eb"
RED = "#b91c1c"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8.5, "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.4, 4.2),
                               gridspec_kw={"width_ratios": [2.5, 1.0], "wspace": 0.42})
fig.subplots_adjust(left=0.085, right=0.965, top=0.83, bottom=0.175)

# ── left: per-kingdom paired points ───────────────────────────────────────────
y = np.arange(len(ORDER))[::-1]
off = 0.15
axL.axvline(CHANCE, color=RED, ls=(0, (4, 3)), lw=1.1, zorder=1)
axL.text(CHANCE, len(ORDER) - 0.42, " chance 0.20", fontsize=7.0, color=RED,
         ha="left", va="center", fontweight="bold")

for yi, k in zip(y, ORDER):
    b, s = np.array(BIO[k]), np.array(SEQ[k])
    # connector showing the gap between arm means
    axL.plot([b.mean(), s.mean()], [yi + off, yi - off], color=MUTED, lw=0.8, ls=":", zorder=2)
    for vals, yy, c in ((b, yi + off, BIO_C), (s, yi - off, SEQ_C)):
        axL.plot([vals.min(), vals.max()], [yy, yy], color=c, lw=1.1, alpha=0.5,
                 solid_capstyle="round", zorder=3)
        axL.plot(vals, [yy] * len(vals), "|", ms=6.5, mec=c, mew=1.1, zorder=4)
        axL.plot(vals.mean(), yy, "o", ms=7, mfc=c, mec="white", mew=1.2, zorder=5)
    axL.annotate(f"{s.mean() - b.mean():+.3f}", xy=(0.662, yi), va="center", ha="left",
                 fontsize=7.4, color=MUTED, annotation_clip=False)

axL.set_yticks(y); axL.set_yticklabels([k.capitalize() for k in ORDER], fontsize=9, color=INK)
axL.set_ylim(-0.62, len(ORDER) - 0.38)
axL.set_xlim(0.19, 0.705)
axL.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6])
axL.set_xlabel("Zero-shot element accuracy on the held-out kingdom", fontsize=8.4,
               color=INK, labelpad=6)
axL.tick_params(axis="x", labelsize=8, colors=MUTED, length=3, width=0.7)
axL.tick_params(axis="y", length=0, pad=4)
axL.grid(axis="x", color=GRID, lw=0.6, zorder=0)
axL.set_axisbelow(True)
for sp in ("top", "right", "left"): axL.spines[sp].set_visible(False)
axL.spines["bottom"].set_color(GRID)
axL.set_title("a   Every kingdom held out in turn", fontsize=9.4, color=INK,
              loc="left", pad=22, fontweight="bold")
axL.text(0, 1.045, "ticks = individual seeds · filled dot = 3-seed mean",
         transform=axL.transAxes, fontsize=7.4, color=MUTED, va="bottom")
axL.text(0.662, len(ORDER) - 0.42, "gap", fontsize=7.0, color=MUTED, ha="left",
         va="center", style="italic", clip_on=False)

# ── right: the overall tie ────────────────────────────────────────────────────
bm = np.array([np.mean([BIO[k][i] for k in ORDER]) for i in range(3)])
sm = np.array([np.mean([SEQ[k][i] for k in ORDER]) for i in range(3)])
for xi, vals, c, lab in ((0, bm, BIO_C, "Biophysics"), (1, sm, SEQ_C, "+ sequence")):
    axR.plot([xi, xi], [vals.mean() - vals.std(ddof=1), vals.mean() + vals.std(ddof=1)],
             color=c, lw=2.4, solid_capstyle="round", zorder=3, alpha=0.85)
    axR.plot([xi] * 3, vals, "o", ms=3.6, mfc="white", mec=c, mew=0.9, zorder=4)
    axR.plot(xi, vals.mean(), "o", ms=9, mfc=c, mec="white", mew=1.4, zorder=5)
    axR.annotate(f"{vals.mean():.4f}\n± {vals.std(ddof=1):.4f}", xy=(xi, vals.mean()),
                 xytext=(xi + 0.17, vals.mean()), fontsize=7.8, color=INK, va="center",
                 ha="left", linespacing=1.35)

axR.set_xlim(-0.42, 1.62)
axR.set_ylim(0.528, 0.560)
axR.set_xticks([0, 1]); axR.set_xticklabels(["Biophysics", "+ sequence"], fontsize=8.4, color=INK)
axR.set_yticks([0.53, 0.54, 0.55, 0.56])
axR.tick_params(axis="y", labelsize=7.6, colors=MUTED, length=3, width=0.7)
axR.tick_params(axis="x", length=0, pad=6)
axR.grid(axis="y", color=GRID, lw=0.6, zorder=0)
axR.set_axisbelow(True)
for sp in ("top", "right"): axR.spines[sp].set_visible(False)
for sp in ("left", "bottom"): axR.spines[sp].set_color(GRID)
axR.set_ylabel("mean over 4 kingdoms", fontsize=7.6, color=MUTED, labelpad=2)
axR.set_title("b   The tie", fontsize=9.4, color=INK, loc="left", pad=22, fontweight="bold")
axR.text(0, 1.045, "3-seed mean ± sd", transform=axR.transAxes,
         fontsize=7.4, color=MUTED, va="bottom")

# the point of the panel: error bars overlap
axR.annotate("", xy=(0.02, 0.5335), xytext=(0.98, 0.5335),
             arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8, shrinkA=0, shrinkB=0,
                             mutation_scale=8))
axR.text(0.5, 0.5315, "+0.004\nwithin error", ha="center", va="top", fontsize=7.4,
         color=INK, fontweight="bold", linespacing=1.3)

handles = [Line2D([], [], color=BIO_C, lw=2.2, marker="o", ms=6, mec="white", mew=1.1,
                  label="Biophysics (profiles only)"),
           Line2D([], [], color=SEQ_C, lw=2.2, marker="o", ms=6, mec="white", mew=1.1,
                  label="Biophysics + sequence")]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.085, 0.005),
           frameon=False, fontsize=8, ncol=2, columnspacing=2.2, handlelength=2.0)

for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig3_loko_tie.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
print("wrote figures/fig3_loko_tie.pdf / .png")
