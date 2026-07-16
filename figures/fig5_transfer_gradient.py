#!/usr/bin/env python3
"""
Figure 5 — the transfer-distance gradient (transfer-only, one configuration family).

REBUILT 2026-07-16. The earlier version put in-distribution as the first rung, but
in-dist comes from a different configuration family (segment windows + kingdom
conditioning ON) than the transfer results (leak-safe: both OFF). Mixing them made
part of the 0.685 -> 0.542 drop a config change rather than distance. This version
plots only the three transfer rungs, all from the leak-safe family, so every point on
every line is the same model receding from home.

Three rungs by increasing evolutionary distance:
  held-out kingdom (LOKO, still eukaryotic) -> unseen diatom lineage -> prokaryotes.

The finding is the sign reversal of what sequence is worth (biophys+seq minus biophys):
  +0.004 (LOKO, a tie) -> 0.000 (diatoms) -> -0.065 (prokaryotes).

Palette = documented categorical slots 1-4 (CVD-validated all-pairs); k-mer and chance
are neutral references, not categorical series.

  python figures/fig5_transfer_gradient.py
  -> figures/fig5_transfer_gradient.{pdf,png}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── data (leak-safe family; see RESULTS_LEDGER.md) ────────────────────────────
RUNGS = ["Held-out\nkingdom", "Diatoms", "Prokaryote\npromoters"]
X = [0, 1, 2]

SERIES = [  # label, values, hex (documented slot order), zorder
    ("Biophysics",            [0.542, 0.569, 0.386], "#2a78d6", 6),
    ("Biophysics + sequence", [0.547, 0.569, 0.321], "#008300", 5),
    ("NT-50M, pretrained",    [0.703, 0.627, 0.599], "#e87ba4", 4),
    ("NT-50M, from scratch",  [0.610, 0.520, 0.286], "#eda100", 4),
]
KMER   = [0.246, 0.283, 0.234]        # LOKO mean, diatoms, prok
CHANCE = [0.20, 0.25, 0.20]           # diatoms = 4 classes -> 0.25

INK, MUTED, FAINT, GRID = "#1a1a19", "#6b7280", "#9ca3af", "#e5e7eb"
RED = "#b91c1c"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8.5, "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(7.4, 4.5))
fig.subplots_adjust(left=0.075, right=0.715, top=0.86, bottom=0.30)

# ── references (recessive) ────────────────────────────────────────────────────
ax.fill_between(X, 0.10, CHANCE, color=MUTED, alpha=0.06, lw=0, zorder=0)
ax.plot(X, CHANCE, ls=(0, (4, 3)), lw=1.1, color=MUTED, zorder=1)
ax.plot(X, KMER, ls=(0, (1.5, 2)), lw=1.5, color=FAINT,
        marker="o", ms=4.2, mfc=FAINT, mec="white", mew=0.7, zorder=2)

# ── the crossing: biophysics overtakes NT-scratch between LOKO and diatoms ─────
# biophys 0.542->0.569 vs NT-scratch 0.610->0.520 cross at x ~= 0.76
ax.axvspan(0.55, 0.97, color="#2a78d6", alpha=0.05, lw=0, zorder=0)
ax.plot([0.76], [0.559], marker="o", ms=4, mfc="none", mec="#2a78d6", mew=1.1, zorder=7)
ax.annotate("structure overtakes sequence", xy=(0.76, 0.548), xytext=(0.80, 0.44),
            ha="center", va="top", fontsize=7.6, color="#2a78d6", style="italic",
            arrowprops=dict(arrowstyle="-", color="#2a78d6", lw=0.7, ls=":",
                            shrinkA=1, shrinkB=1))

# ── model series ──────────────────────────────────────────────────────────────
for label, vals, hexc, z in SERIES:
    ax.plot(X, vals, lw=2.0, color=hexc, marker="o", ms=6.2,
            mfc=hexc, mec="white", mew=1.3, zorder=z, solid_capstyle="round")

# ── the finding: sequence HURTS at the far rung ───────────────────────────────
ax.annotate("", xy=(1.94, 0.325), xytext=(1.94, 0.382),
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5,
                            shrinkA=0, shrinkB=0, mutation_scale=10), zorder=8)
ax.text(1.62, 0.352, "adding sequence HURTS\n−0.065", fontsize=7.8, color=RED,
        ha="center", va="center", fontweight="bold", linespacing=1.3, zorder=8)

# ── direct labels ─────────────────────────────────────────────────────────────
for label, vals, hexc, _ in SERIES:
    y = vals[-1]
    ax.annotate(f"{label}  {y:.3f}", xy=(2.05, y), xycoords="data",
                va="center", ha="left", fontsize=8.2, color=INK, annotation_clip=False)
    ax.plot([2.0, 2.03], [y, y], lw=0.7, color=hexc, clip_on=False, zorder=2)
for txt, yv in [(f"k-mer floor  {KMER[-1]:.3f}", KMER[-1]), ("chance  0.20", 0.20)]:
    ax.annotate(txt, xy=(2.05, yv), va="center", ha="left", fontsize=7.6,
                color=MUTED, annotation_clip=False)

# ── axes ──────────────────────────────────────────────────────────────────────
ax.set_xticks(X); ax.set_xticklabels(RUNGS, fontsize=8.6, color=INK)
ax.set_xlim(-0.12, 2.0)
ax.set_ylim(0.155, 0.735)
ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
ax.set_ylabel("Zero-shot element accuracy", fontsize=9, color=INK, labelpad=6)
ax.set_xlabel("increasing distance from training distribution  →",
              fontsize=8, color=MUTED, style="italic", labelpad=9)
ax.tick_params(axis="y", labelsize=8, colors=MUTED, length=3, width=0.7)
ax.tick_params(axis="x", length=0, pad=6)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)

ax.set_title("Beyond the training distribution, structure overtakes sequence",
             fontsize=10.4, color=INK, loc="left", pad=11, fontweight="bold")

# ── legend, parked outside ────────────────────────────────────────────────────
handles = [Line2D([], [], color=h, lw=2.0, marker="o", ms=5.2, mec="white", mew=1.0, label=l)
           for l, _, h, _ in SERIES]
handles += [Line2D([], [], color=FAINT, lw=1.5, ls=(0, (1.5, 2)), label="k-mer floor"),
            Line2D([], [], color=MUTED, lw=1.1, ls=(0, (4, 3)), label="chance (0.25 diatoms)")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.235),
          frameon=False, fontsize=7.6, ncol=3, columnspacing=1.8,
          handlelength=2.2, labelspacing=0.5, borderpad=0)

for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig5_transfer_gradient.{ext}", dpi=600,
                bbox_inches="tight", facecolor="white")
print("wrote figures/fig5_transfer_gradient.pdf / .png  (transfer-only, 3 rungs)")
