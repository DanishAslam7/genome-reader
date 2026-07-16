#!/usr/bin/env python3
"""
Figure 2 — in distribution, structure is as good as sequence.

(a) three input arms, 3 seeds each, on the shared 32-organism test split.
(b) the decomposition: what sequence adds, and what the knowledge graph adds on
    top of it. Both increments are ~1pp on a 68-71% base — i.e. structure alone
    is already doing nearly all the work.

All sd's are SAMPLE sd (ddof=1), matching Figures 3-6. (The ledger previously
quoted population sd for this panel only; corrected 2026-07-15.)

  python figures/fig2_indist_ablation.py
  -> figures/fig2_indist_ablation.{pdf,png}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── data: seeds 42 / 123 / 777 (RESULTS_LEDGER.md, Phase 1) ───────────────────
ARMS = [
    ("Biophysics\n(profiles only)", [0.6807, 0.6847, 0.6889], "#2a78d6"),
    ("+ sequence",                  [0.7024, 0.6928, 0.6914], "#008300"),
    ("+ sequence\n+ knowledge graph", [0.7069, 0.7074, 0.7136], "#e87ba4"),
]
CHANCE = 0.20
INK, MUTED, GRID = "#1a1a19", "#6b7280", "#e5e7eb"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8.5, "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 3.9),
                               gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.40})
fig.subplots_adjust(left=0.10, right=0.965, top=0.80, bottom=0.20)

# ── (a) the three arms ────────────────────────────────────────────────────────
means = [np.mean(v) for _, v, _ in ARMS]
sds = [np.std(v, ddof=1) for _, v, _ in ARMS]

for i, ((lab, vals, c), m, sd) in enumerate(zip(ARMS, means, sds)):
    v = np.array(vals)
    axA.plot([i, i], [m - sd, m + sd], color=c, lw=2.6, solid_capstyle="round",
             alpha=0.85, zorder=3)
    axA.plot([i] * len(v), v, "_", ms=11, mec=c, mew=1.3, zorder=4)
    axA.plot(i, m, "o", ms=9, mfc=c, mec="white", mew=1.4, zorder=5)
    axA.annotate(f"{m:.3f}\n± {sd:.3f}", xy=(i + 0.19, m), fontsize=7.8, color=INK,
                 va="center", ha="left", linespacing=1.35)

axA.set_xlim(-0.45, 2.75)
axA.set_ylim(0.672, 0.722)
axA.set_xticks(range(len(ARMS)))
axA.set_xticklabels([a[0] for a in ARMS], fontsize=8.2, color=INK)
axA.set_yticks([0.68, 0.69, 0.70, 0.71, 0.72])
axA.set_ylabel("Element accuracy (in distribution)", fontsize=8.6, color=INK, labelpad=6)
axA.tick_params(axis="y", labelsize=8, colors=MUTED, length=3, width=0.7)
axA.tick_params(axis="x", length=0, pad=6)
axA.grid(axis="y", color=GRID, lw=0.6, zorder=0)
axA.set_axisbelow(True)
for s in ("top", "right"): axA.spines[s].set_visible(False)
for s in ("left", "bottom"): axA.spines[s].set_color(GRID)
axA.set_title("a   Adding sequence buys ~1 point", fontsize=9.4, color=INK,
              loc="left", pad=22, fontweight="bold")
axA.text(0, 1.045, "dashes = individual seeds · dot = mean, bar = ± sd (n = 3)",
         transform=axA.transAxes, fontsize=7.4, color=MUTED, va="bottom")

# ── (b) decomposition ─────────────────────────────────────────────────────────
seq_gain = 100 * (means[1] - means[0])
kg_gain = 100 * (means[2] - means[1])
base = 100 * (means[0] - CHANCE)

bars = [("Structure alone\nabove chance", base, "#2a78d6"),
        ("Sequence adds", seq_gain, "#008300"),
        ("Knowledge graph\nadds (residual)", kg_gain, "#e87ba4")]
ypos = np.arange(len(bars))[::-1]
for yi, (lab, val, c) in zip(ypos, bars):
    axB.barh(yi, val, height=0.55, color=c, zorder=3)
    axB.text(val + 0.9, yi, f"{val:+.2f} pp", va="center", ha="left",
             fontsize=8, color=INK, fontweight="bold")

axB.set_yticks(ypos); axB.set_yticklabels([b[0] for b in bars], fontsize=8, color=INK)
axB.set_xlim(0, 62)
axB.set_xticks([0, 10, 20, 30, 40, 48.5])
axB.set_xticklabels(["0", "10", "20", "30", "40", "48.5"], fontsize=7.6)
axB.set_xlabel("percentage points of element accuracy", fontsize=7.8, color=MUTED, labelpad=5)
axB.tick_params(axis="x", labelsize=7.6, colors=MUTED, length=3, width=0.7)
axB.tick_params(axis="y", length=0, pad=4)
axB.grid(axis="x", color=GRID, lw=0.6, zorder=0)
axB.set_axisbelow(True)
for s in ("top", "right", "left"): axB.spines[s].set_visible(False)
axB.spines["bottom"].set_color(GRID)
axB.set_title("b   Where the accuracy comes from", fontsize=9.4, color=INK,
              loc="left", pad=22, fontweight="bold")
axB.text(0, 1.045, "relative to chance (0.20)", transform=axB.transAxes,
         fontsize=7.4, color=MUTED, va="bottom")

fig.text(0.10, 0.055,
         "Profiles alone carry 48.5 of the 51.0 points separating chance from the full model: sequence and the knowledge graph add 1.08 and 1.38 points respectively.",
         fontsize=6.9, color=INK)
fig.text(0.10, 0.016,
         "The knowledge graph is admissible here but not for transfer (Section 3.2): several of its features derive from element annotations. A profiles+KG arm without sequence would not converge.",
         fontsize=6.9, color=MUTED)

for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig2_indist_ablation.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
print("wrote figures/fig2_indist_ablation.pdf / .png")
print(f"  structure above chance = {base:.2f}pp | sequence {seq_gain:+.2f} | KG {kg_gain:+.2f}")
