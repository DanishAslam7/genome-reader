#!/usr/bin/env python3
"""
Figure 1 - study design and model.

(a) the cohort: 32 organisms, 8 per kingdom (balanced by construction, which is what
    makes the leave-one-kingdom-out folds comparable), plus the two external
    validation sets that never contribute to training.
(b) the model: profile trunk is the object of study; the sequence and knowledge-graph
    branches are ablation arms, and the knowledge graph is disabled for all transfer.

COLOUR CONTRACT (paper-wide): blue = biophysics/profiles, green = +sequence,
magenta = +knowledge graph. Kingdoms are NEVER encoded by hue - no 4-hue set drawn
from the remaining documented slots clears the all-pairs CVD/normal-vision floors
(best is 6.9/20.8, floor band), so kingdoms are encoded by position and label.

Layout note: both panels use 0-100 data coords on non-square axes, so a y-unit is
~2.5x smaller on screen than an x-unit. Vertical budgets below are in y-units and
assume ~3.9 y-units per line of 8pt text; do not tighten them without re-rendering.

  python figures/fig1_design.py
  -> figures/fig1_design.{pdf,png}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import FancyBboxPatch, Rectangle

# ── cohort (verified from dataset_reprofiled_32/encoders.json + Y_kingdom) ─────
COHORT = {
    "Animalia": ["H. sapiens", "M. musculus", "R. norvegicus", "G. gallus",
                 "X. tropicalis", "D. rerio", "D. melanogaster", "C. elegans"],
    "Plantae":  ["A. thaliana", "O. sativa", "Z. mays", "T. aestivum",
                 "G. max", "S. lycopersicum", "P. trichocarpa", "P. patens"],
    "Fungi":    ["S. cerevisiae", "S. pombe", "K. lactis", "Y. lipolytica",
                 "A. niger", "A. fumigatus", "N. crassa", "C. neoformans"],
    "Protista": ["P. falciparum", "T. gondii", "L. major", "T. brucei",
                 "T. thermophila", "G. intestinalis", "D. discoideum", "C. reinhardtii"],
}

BIO, SEQ, KG = "#2a78d6", "#008300", "#e87ba4"
INK, MUTED, RULE, PANEL = "#1a1a19", "#6b7280", "#d6dae0", "#f4f5f7"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig = plt.figure(figsize=(7.4, 7.7))
gs = fig.add_gridspec(2, 1, height_ratios=[1.30, 0.95], hspace=0.14,
                      left=0.045, right=0.965, top=0.965, bottom=0.075)
axA = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1])
for ax in (axA, axB):
    ax.set_xlim(0, 100); ax.axis("off")
axA.set_ylim(-6, 100); axB.set_ylim(0, 100)


def box(ax, x, y, w, h, fc="white", ec=RULE, lw=0.9, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                fc=fc, ec=ec, lw=lw, zorder=z))


# PhyloPic silhouettes: all public domain (7x CC0, 1x PD Mark) - see
# figures/phylopic/CREDITS.json. Recoloured to the figure ink; extents are computed
# from the axes' true inch dimensions so the shapes are never distorted by the
# non-square 0-100 coordinate system.
SIL = {}
def silhouette(ax, slug, cx, cy, max_dy, max_dx, color="#1a1a19", alpha=0.88):
    """Fit a silhouette inside (max_dx, max_dy) without distorting it. Scaling by
    height alone blows wide shapes (maize is 512x197) out of their column."""
    f = Path("figures/phylopic") / f"{slug}.png"
    if not f.exists():
        return
    if slug not in SIL:
        SIL[slug] = plt.imread(f)
    im = np.array(SIL[slug])
    rgb = np.array([int(color[i:i+2], 16) / 255 for i in (1, 3, 5)])
    im[..., :3] = rgb                      # recolour, keep the alpha mask
    im[..., 3] *= alpha
    h_px, w_px = im.shape[0], im.shape[1]
    bb = ax.get_position()
    W_in, H_in = bb.width * fig.get_figwidth(), bb.height * fig.get_figheight()
    k = (w_px / h_px) * (H_in / W_in)      # x-units of width per y-unit of height
    dy = min(max_dy, max_dx / k)           # fit BOTH dimensions
    dx = dy * k
    ax.imshow(im, extent=(cx - dx / 2, cx + dx / 2, cy, cy + dy),
              aspect="auto", zorder=5, interpolation="antialiased")




def justified_paragraph(fig, x0, y0, width, text, fontsize, color, leading=1.55):
    """Draw `text` justified to `width` (figure fraction), anchored top-left at
    (x0, y0). Matplotlib has no justified text: measure every word, greedy-wrap,
    then distribute the slack across the inter-word gaps. The final line of the
    paragraph is left as-is, per normal typesetting.
    """
    fig.canvas.draw()                       # need a renderer to measure
    rend = fig.canvas.get_renderer()
    px_per_frac = fig.get_figwidth() * fig.dpi

    def w_px(t):
        a = fig.text(0, 0, t, fontsize=fontsize)
        w = a.get_window_extent(renderer=rend).width
        a.remove()
        return w

    space = w_px(" ")
    target = width * px_per_frac
    words = text.split()
    widths = [w_px(t) for t in words]

    lines, cur, cur_w = [], [], 0.0
    for t, wd in zip(words, widths):
        add = wd if not cur else wd + space
        if cur and cur_w + add > target:
            lines.append((cur, cur_w))
            cur, cur_w = [t], wd
        else:
            cur.append(t); cur_w += add
    if cur:
        lines.append((cur, cur_w))

    line_h = fontsize * leading / 72.0 / fig.get_figheight()
    for i, (ws, natural) in enumerate(lines):
        y = y0 - i * line_h
        last = (i == len(lines) - 1)
        if last or len(ws) == 1:
            gap = space                     # never stretch the final line
        else:
            gap = space + (target - natural) / (len(ws) - 1)
        x = x0 * px_per_frac
        for t in ws:
            fig.text(x / px_per_frac, y, t, fontsize=fontsize, color=color, va="top")
            x += w_px(t) + gap



# ══════════════════════ (a) cohort ══════════════════════
axA.text(0, 97, "a", fontsize=11, fontweight="bold", color=INK)
axA.text(3.2, 97, "32 organisms · four kingdoms · eight each", fontsize=9.6,
         fontweight="bold", color=INK)
axA.text(3.2, 91, "Balanced by construction, so each held-out-kingdom fold withholds "
                  "an equal share of the training set.", fontsize=7.6, color=MUTED)

ICONS = {"Animalia": [("human", 13.5), ("fly", 8.0)],
         "Plantae":  [("arabidopsis", 11.5), ("maize", 12.5)],
         "Fungi":    [("yeast", 9.0), ("neurospora", 12.0)],
         "Protista": [("plasmodium", 10.5), ("chlamydomonas", 10.5)]}

colw, x0, gap = 22.0, 1.5, 1.4
for i, (kingdom, orgs) in enumerate(COHORT.items()):
    x = x0 + i * (colw + gap)
    box(axA, x, 22, colw, 62, fc=PANEL, ec=RULE)
    axA.add_patch(Rectangle((x, 79.4), colw, 4.6, fc=INK, ec="none", zorder=3))
    axA.text(x + colw / 2, 81.7, kingdom.upper(), fontsize=7.8, color="white",
             ha="center", va="center", fontweight="bold", zorder=4)
    # representative silhouettes
    for k, (slug, dy) in enumerate(ICONS[kingdom]):
        silhouette(axA, slug, x + colw * (0.28 + 0.44 * k), 64.5, dy, max_dx=colw * 0.40)
    for j, o in enumerate(orgs):
        axA.text(x + 1.6, 58.0 - j * 4.6, o, fontsize=7.1, color=INK, va="center",
                 style="italic", zorder=4)

axA.annotate("", xy=(x0 + 4 * colw + 3 * gap - 0.4, 18.5), xytext=(x0, 18.5),
             arrowprops=dict(arrowstyle="-", color=RULE, lw=1.0))
axA.text(x0, 14.5, "TRAINING COHORT - each kingdom held out in turn (Section 3.2)", fontsize=7.2,
         color=MUTED, fontweight="bold")

# external sets - taller boxes; title and body get their own lines
for i, (title, l1, l2) in enumerate([
        ("2 diatoms", "P. tricornutum · T. pseudonana", "unseen eukaryotic lineage"),
        ("49 prokaryotes", "promoters", "a different domain of life")]):
    x = x0 + i * 48
    box(axA, x, -4, 44, 15, fc="white", ec=INK, lw=1.0)
    axA.text(x + 2.4, 6.8, title, fontsize=8.2, color=INK, fontweight="bold", va="center")
    # species names italic, matching the kingdom columns; "promoters" is not a name
    axA.text(x + 2.4, 2.2, l1, fontsize=6.9, color=INK, va="center",
             style="italic" if i == 0 else "normal")
    axA.text(x + 2.4, -1.6, l2, fontsize=6.5, color=MUTED, va="center", style="italic")
axA.text(x0 + 96.5, 14.5, "NEVER TRAINED ON  (Section 3.4)", fontsize=7.2, color=INK,
         ha="right", va="center", fontweight="bold")

# ══════════════════════ (b) model ══════════════════════
axB.text(0, 97, "b", fontsize=11, fontweight="bold", color=INK)
axB.text(3.2, 97, "Multitask model", fontsize=9.6, fontweight="bold", color=INK)
axB.text(3.2, 91, "The profile trunk is the object of study; the other two inputs are "
                  "ablation arms.", fontsize=7.6, color=MUTED)

# --- inputs: 3 boxes, h=24, title band + 2 body lines ---
def input_box(y, ec, title, body, note):
    box(axB, 1.5, y, 22, 24, fc="white", ec=ec, lw=1.5)
    axB.text(12.5, y + 19.0, title, fontsize=7.2, color=ec, ha="center", fontweight="bold")
    axB.text(12.5, y + 11.0, body, fontsize=7.8, color=INK, ha="center", va="center",
             linespacing=1.45)
    axB.text(12.5, y + 3.6, note, fontsize=6.3, color=MUTED, ha="center", style="italic")

input_box(58, BIO, "PROFILES", "7 parameters\n× 475 positions", "no sequence identity")
input_box(30, SEQ, "SEQUENCE", "one-hot\n501 bp", "ablation arm")
input_box(2, KG, "KNOWLEDGE GRAPH", "organism / kingdom\nnode features", "off for all transfer")

for y in (70, 42, 14):
    axB.annotate("", xy=(31.5, y), xytext=(24.2, y),
                 arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0, mutation_scale=8))

# --- trunk ---
box(axB, 32, 30, 21, 52, fc="#eaf2fc", ec=BIO, lw=1.6)
axB.text(42.5, 77, "TRUNK", fontsize=7.2, color=BIO, ha="center", fontweight="bold")
axB.text(42.5, 60, "3 convolutional\nstages", fontsize=7.8, color=INK, ha="center",
         va="center", linespacing=1.45)
axB.text(42.5, 48, "d = 256", fontsize=8.4, color=INK, ha="center", fontweight="bold")
axB.text(42.5, 35, "per-organism\nz-normalized", fontsize=6.3, color=MUTED, ha="center",
         va="center", linespacing=1.4, style="italic")

axB.annotate("", xy=(59.5, 56), xytext=(53.7, 56),
             arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0, mutation_scale=8))

# --- GRL ---
box(axB, 60, 38, 19, 36, fc="white", ec="#b91c1c", lw=1.4)
axB.text(69.5, 66, "GRADIENT\nREVERSAL", fontsize=6.9, color="#b91c1c", ha="center",
         va="center", fontweight="bold", linespacing=1.35)
axB.text(69.5, 51, "organism λ=0.005\nkingdom λ=0.08", fontsize=7.2, color=INK,
         ha="center", va="center", linespacing=1.5)
axB.text(69.5, 42, "strips taxonomy", fontsize=6.3, color=MUTED, ha="center", style="italic")

axB.annotate("", xy=(84.5, 56), xytext=(79.2, 56),
             arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0, mutation_scale=8))

# --- heads ---
box(axB, 85, 30, 14, 52, fc="white", ec=INK, lw=1.6)
axB.text(92, 77, "ELEMENT", fontsize=7.2, color=INK, ha="center", fontweight="bold")
for j, c in enumerate(["exon boundary", "gene boundary", "promoter",
                       "start codon", "stop codon"]):
    axB.text(92, 68 - j * 6.0, c, fontsize=6.3, color=INK, ha="center")
axB.text(92, 40, "annotated in all\n32 organisms", fontsize=5.9, color=MUTED, ha="center",
         va="center", linespacing=1.4, style="italic")
axB.text(92, 33, "chance = 0.20", fontsize=6.3, color=MUTED, ha="center", style="italic")

box(axB, 85, 1, 14, 24, fc=PANEL, ec=RULE, lw=0.9)
axB.text(92, 21, "AUXILIARY · 6", fontsize=6.4, color=MUTED, ha="center", fontweight="bold")
axB.text(92, 11.5, "coarse element\nbio group · cds\nstart/stop pair\nUTR subtype\n"
                   "human-only UTR",
         fontsize=5.6, color=MUTED, ha="center", va="center", linespacing=1.5)

# footnote at FIGURE level, below both panels - justified so the block reads as a
# tidy rectangle rather than three ragged lines of very unequal length.
justified_paragraph(
    fig, 0.045, 0.058, 0.920,
    "The element head carries loss weight 2.5; the six auxiliary heads carry 3.2 between them and "
    "so shape the shared trunk substantially. UTR and enhancer annotations exist for human alone "
    "in this cohort and are quarantined in a masked human-only head, keeping the five element "
    "classes comparable across all 32 organisms; the element task therefore has no enhancer "
    "class. Held-out organisms are normalized from "
    "their own profiles (Section 2.5). The knowledge graph and the cohort-derived segment windows "
    "are disabled for every transfer evaluation, since both were constructed using the held-out "
    "organisms' annotations. Silhouettes from PhyloPic (public domain): T. M. Keesey, T. Hegna, "
    "J. Warner, W. Decatur, Arcadia Science, J. Wells.",
    fontsize=6.5, color=MUTED)

for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig1_design.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
print("wrote figures/fig1_design.pdf / .png")
