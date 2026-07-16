#!/usr/bin/env python3
"""
Figure 4 — the normalization confound.

BOTH panels are eval-only on a FIXED trained model: held-out organisms contribute no
training rows, so their normalization statistics can only affect test-time input.
Each model is scored twice, varying nothing but the normalization. This removes the
seam in the earlier version, where each before/after pair came from two separately
trained checkpoints — a real confound, since in the collapsed regime the output is
arbitrary and two equivalent checkpoints land ~6 points apart.

Validation: the own-profile values reproduce each trained run's own test metric to
~4 decimals (with-sequence: 0.5989 vs 0.5990, 0.5141 vs 0.5142, 0.6448 vs 0.6447,
0.4455 vs 0.4455). Same computation, not merely close.

COLOUR CONTRACT: colour encodes the ARM (blue = profiles, green = + sequence), never
the kingdom — kingdoms are direct-labelled. No 4-hue kingdom set clears the all-pairs
CVD floors, and blue/green must mean the same thing in every figure.

  python figures/fig4_normalization.py
  -> figures/fig4_normalization.{pdf,png}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── data: norm_ablation_out/*.json, seed 42, full test sets, eval-only ────────
BIO = {"plantae": (0.1878, 0.6401), "animalia": (0.1648, 0.6069),
       "fungi": (0.2342, 0.4938), "protista": (0.1328, 0.4484)}
SEQ = {"plantae": (0.6116, 0.6448), "animalia": (0.5520, 0.5989),
       "fungi": (0.4879, 0.5141), "protista": (0.4151, 0.4455)}
ORDER = ["plantae", "animalia", "fungi", "protista"]
N = {"plantae": 191254, "animalia": 172711, "fungi": 78897, "protista": 93274}
CHANCE = 0.20

BIO_C, SEQ_C = "#2a78d6", "#008300"
INK, MUTED, GRID = "#1a1a19", "#6b7280", "#e5e7eb"
RED = "#b91c1c"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8.5, "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 4.4), sharey=True)
fig.subplots_adjust(left=0.085, right=0.74, top=0.80, bottom=0.22, wspace=0.52)


def panel(ax, data, color, title, subtitle):
    ax.axhspan(0.10, CHANCE, color=RED, alpha=0.05, lw=0)
    ax.axhline(CHANCE, color=RED, ls=(0, (4, 3)), lw=1.1, zorder=1)
    for k in ORDER:
        y0, y1 = data[k]
        ax.plot([0, 1], [y0, y1], color=color, lw=2.0, marker="o", ms=6.5,
                mfc=color, mec="white", mew=1.3, zorder=4, solid_capstyle="round")
        ax.annotate(f"{k.capitalize()}   {y1 - y0:+.3f}", xy=(1.07, y1), va="center",
                    ha="left", fontsize=7.4, color=INK, annotation_clip=False)
    ax.set_xlim(-0.30, 1.30)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["identity\nfallback", "own-profile\nstatistics"], fontsize=8.2, color=INK)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.set_title(title, fontsize=9.4, color=INK, loc="left", pad=20, fontweight="bold")
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=7.6, color=MUTED, va="bottom")
    mean_d = np.mean([data[k][1] - data[k][0] for k in ORDER])
    ax.text(0.5, 0.135, f"mean {mean_d:+.3f}", ha="center", va="center",
            fontsize=9.0, color=color, fontweight="bold")
    return mean_d


dA = panel(axA, BIO, BIO_C, "a   Profiles only",
           "collapses below chance without the fix")
dB = panel(axB, SEQ, SEQ_C, "b   Profiles + sequence",
           "structurally immune — which hid the bug")

axA.set_ylim(0.10, 0.70)
axA.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
axA.set_ylabel("Element accuracy, held-out kingdom (zero-shot)", fontsize=8.8,
               color=INK, labelpad=6)
axA.tick_params(axis="y", labelsize=8, colors=MUTED, length=3, width=0.7)
axB.tick_params(axis="y", length=0)
# labelled in panel b: in panel a the identity markers (0.133-0.234) sit on this line
axB.text(-0.28, CHANCE + 0.012, "chance 0.20", fontsize=7.0, color=RED,
         ha="left", va="bottom", fontweight="bold")

# the headline: the ASYMMETRY is the diagnosis
fig.text(0.5, 0.905, f"the same correction is worth {dA/dB:.0f}× more to profiles than to sequence",
         ha="center", fontsize=8.6, color=INK, style="italic")

fig.text(0.085, 0.105,
         "Both panels are eval-only on one fixed trained model. Held-out organisms contribute no training rows, so their statistics affect test-time\n"
         "input alone: each model is scored twice, varying nothing else. Seed 42; full test sets (78,897–191,254 windows).",
         fontsize=6.6, color=INK, va="top", linespacing=1.65)
fig.text(0.085, 0.045,
         "The trunk was fitted on standardized input, so held-out organisms arrived at raw scale — the model did not meet an unfamiliar kingdom,\n"
         "it met unfamiliar units. The asymmetry between the panels is the diagnosis; the recovery in (a) alone would not be.",
         fontsize=6.6, color=MUTED, va="top", linespacing=1.65)

for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig4_normalization.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
print(f"wrote figures/fig4_normalization.pdf / .png   (mean {dA:+.4f} vs {dB:+.4f} = {dA/dB:.1f}x)")
