#!/usr/bin/env python3
"""
Supplementary Figure S1 - model architecture (rev 4).

Rev 3 was legible in principle but not in practice: three panels side by side in
7.4in forced type down to 4.6-5.5pt, well below print-readable. Rev 4 keeps the
same one-spine design and spends vertical space instead, so nothing is under ~6.2pt
and body text sits at 7-7.5pt.

Layout: a horizontal spine across the top (traceable left-to-right in one sweep),
then full- or half-width detail panels stacked beneath it.

Every layer, shape and parameter count is READ FROM THE TRAINED CHECKPOINT via
dump_model_summary.py (model_summary_out/*.json).

COLOUR CONTRACT (paper-wide): blue = profiles/trunk, green = sequence branch,
magenta = knowledge graph, red = adversarial + the fixed prior-derived weighting.

  python figures/figS_architecture.py
  -> figures/figS_architecture.{pdf,png}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BIO, SEQ, KG, RED = "#2a78d6", "#008300", "#e87ba4", "#b91c1c"
INK, MUTED, RULE = "#1a1a19", "#6b7280", "#d6dae0"
TINT, RTINT, PANEL = "#eaf2fc", "#fdeceb", "#f7f8fa"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig = plt.figure(figsize=(7.6, 9.9))
ax = fig.add_axes([0.015, 0.01, 0.97, 0.975])
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, fc="white", ec=RULE, lw=1.0, z=2, r=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def flow(x1, x2, y, color=MUTED, lw=1.3):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=9,
                                shrinkA=0, shrinkB=0))


def panel(x, w, y, h, letter, title, ec=RULE):
    box(x, y, w, h, fc=PANEL, ec=ec, lw=1.0, r=1.0)
    ax.text(x + 1.4, y + h - 2.0, letter, fontsize=11, fontweight="bold", color=INK, va="center")
    ax.text(x + 4.8, y + h - 2.0, title, fontsize=8.8, fontweight="bold", color=INK, va="center")


ax.text(0, 98.4, "Supplementary Figure S1 - model architecture", fontsize=13.5,
        fontweight="bold", color=INK)
ax.text(0, 96.0, "Read from the trained checkpoint. The profile trunk is the object of study.",
        fontsize=8.6, color=MUTED)

# ══════════════════ THE SPINE ══════════════════
SY, SH = 87.0, 6.6
CYM = SY + SH / 2
SPINE = [(1.0, 11.0, "profile", "475 × 7", BIO, "white", 1.8, None),
         (14.5, 12.5, "× scale", "fixed prior", RED, RTINT, 1.6, None),
         (29.5, 10.0, "stem", "k=7 → 256", BIO, "white", 1.1, "12,800"),
         (42.0, 16.5, "3 × ConvNeXt stage", "475 → 238 → 119", BIO, "white", 1.5, "1.79 M"),
         (61.0, 12.0, "pool → dense", "→ 128", BIO, "white", 1.1, "98,688"),
         (75.5, 9.0, "⊕ fuse", "+ derived", BIO, TINT, 1.1, "62,288"),
         (87.0, 12.0, "SHARED", "176-dim", INK, "white", 2.0, None)]
for i, (x, w, t, sub, ec, fc, lw, prm) in enumerate(SPINE):
    box(x, SY, w, SH, fc=fc, ec=ec, lw=lw)
    ax.text(x + w / 2, SY + SH * 0.63, t, ha="center", va="center", fontsize=8.4,
            color=RED if ec == RED else INK, zorder=4,
            fontweight="bold" if ec in (RED, INK) else "normal")
    ax.text(x + w / 2, SY + SH * 0.27, sub, ha="center", va="center", fontsize=6.9,
            color=MUTED, zorder=4, family="monospace")
    if prm:
        ax.text(x + w / 2, SY - 1.4, prm, ha="center", va="top", fontsize=6.6, color=MUTED,
                family="monospace")
    if i < len(SPINE) - 1:
        flow(x + w + 0.2, SPINE[i + 1][0] - 0.2, CYM)
ax.text(93.0, SY - 1.4, "→ panel d", ha="center", va="top", fontsize=6.8, color=INK,
        fontweight="bold")
ax.text(50, 94.4, "one continuous path: profile → fixed channel scaling → trunk → shared representation → nine heads",
        ha="center", fontsize=7.6, color=MUTED, style="italic")

# ══════════════════ (a) fixed channel scaling - FULL WIDTH ══════════════════
PX, PW, PY, PH = 1.0, 98.0, 62.5, 19.0
panel(PX, PW, PY, PH, "a", "Fixed channel scaling - the prior this work inherits", ec=RED)
# connector: bottom of the "× scale" node -> top of panel (a). Must touch both,
# or it reads as stray dots floating in the gap.
ax.annotate("", xy=(20.75, PY + PH), xytext=(20.75, SY),
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.1, mutation_scale=8,
                            ls=(0, (2.5, 2)), shrinkA=1, shrinkB=1))
ax.text(PX + 2.0, PY + PH - 5.2,
        "Non-trainable, applied to the profile before the stem. Weights are taken from the descriptive study's\n"
        "start/stop findings and were fixed before any model in this work was trained.",
        fontsize=7.4, color=INK, va="top", linespacing=1.6)

wts = [("hbond", 2.0, RED), ("stack", 2.0, RED), ("sol", 1.4, "#b9c2cc"), ("bp", 1.4, "#b9c2cc"),
       ("intra", 1.4, "#b9c2cc"), ("bbone", 1.4, "#b9c2cc"), ("inter", 1.0, BIO)]
bx, bw, base = 4.0, 5.2, PY + 3.4
for i, (nm, w, c) in enumerate(wts):
    x = bx + i * 7.2
    hh = (w / 2.0) * 6.2
    ax.add_patch(FancyBboxPatch((x, base), bw, hh, boxstyle="round,pad=0,rounding_size=0.3",
                                fc=c, ec="none", zorder=3))
    ax.text(x + bw / 2, base + hh + 0.8, f"×{w}", ha="center", fontsize=7.4,
            color=c if c != "#b9c2cc" else MUTED, fontweight="bold", zorder=4)
    ax.text(x + bw / 2, base - 1.7, nm, ha="center", fontsize=7.0, color=INK, zorder=4)
ax.text(56.0, PY + 11.0,
        "The two channels the prior singles out at\n"
        "start/stop get ×2.0; the four that shift\n"
        "more weakly get ×1.4; inter gets ×1.0,\n"
        "because the prior reports nothing\n"
        "distinctive about it there.",
        fontsize=7.2, color=INK, va="top", linespacing=1.55)
ax.text(56.0, PY + 3.0,
        "Section 3.5 ablates these same\n"
        "channels, so it is disclosed here.",
        fontsize=7.0, color=RED, va="top", style="italic", linespacing=1.55)

# ══════════════════ (b) ConvNeXt block - LEFT HALF ══════════════════
BX, BW, BY, BH = 1.0, 47.0, 39.0, 21.0
panel(BX, BW, BY, BH, "b", "ConvNeXt block  (one of three)")
by = [BY + 12.4, BY + 8.6, BY + 4.8, BY + 1.0]
for (t, s), yy in zip([("depthwise Conv1D  k=7", "groups 256 · no bias · 1,792"),
                       ("LayerNormalization", "512"),
                       ("Dense 1024 · ReLU", "4× expansion · 263,168"),
                       ("Dense 256", "project back · 262,400")], by):
    box(BX + 6.0, yy, BW - 10.0, 3.2, fc="white", ec=RULE, lw=0.9)
    ax.text(BX + BW / 2 + 1.0, yy + 2.15, t, ha="center", fontsize=7.2, color=INK, zorder=4)
    ax.text(BX + BW / 2 + 1.0, yy + 0.85, s, ha="center", fontsize=6.2, color=MUTED, zorder=4,
            family="monospace")
for yy in by[:-1]:
    ax.annotate("", xy=(BX + BW / 2 + 1.0, yy - 0.5), xytext=(BX + BW / 2 + 1.0, yy),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, mutation_scale=6))
ax.add_patch(FancyArrowPatch((BX + 4.0, by[0] + 3.2), (BX + 4.0, by[-1] + 1.6),
                             connectionstyle="arc3,rad=0.5", arrowstyle="-|>",
                             color=MUTED, lw=0.8, mutation_scale=6, zorder=1))
ax.text(BX + 2.2, (by[0] + by[-1]) / 2, "residual", fontsize=6.6, color=MUTED, rotation=90,
        ha="center", va="center")

# ══════════════════ (c) branches - RIGHT HALF ══════════════════
CX2, CW2, CY2, CH2 = 52.0, 47.0, 39.0, 21.0
panel(CX2, CW2, CY2, CH2, "c", "Branches into the shared vector")
rows = [("profile complexity", "14 stats/channel → 98 → 48", BIO, "derived · always on"),
        ("start/stop contrast", "L/centre/R windows → 36 → 32", BIO, "derived · always on"),
        ("sequence", "one-hot 501×4 · k9→k15→k21 → 96", SEQ, "ablation arm · +19.3%"),
        ("knowledge graph", "organism/kingdom nodes → 48", KG, "OFF for all transfer")]
for i, (t, s, c, note) in enumerate(rows):
    yy = CY2 + CH2 - 9.0 - i * 3.6
    ax.add_patch(FancyBboxPatch((CX2 + 2.2, yy), 1.8, 2.8, boxstyle="round,pad=0,rounding_size=0.3",
                                fc=c, ec="none", zorder=3))
    ax.text(CX2 + 5.4, yy + 1.9, t, fontsize=7.4, color=INK, va="center", zorder=4)
    ax.text(CX2 + 5.4, yy + 0.4, s, fontsize=6.2, color=MUTED, va="center", zorder=4,
            family="monospace")
    ax.text(CX2 + CW2 - 1.8, yy + 1.4, note, fontsize=6.4, color=c if c != BIO else MUTED,
            va="center", ha="right", style="italic", zorder=4)
ax.text(CX2 + 2.2, CY2 + CH2 - 4.6, "the two derived branches make the shared vector 176-dim",
        fontsize=6.6, color=MUTED, style="italic")

# ══════════════════ (d) heads - FULL WIDTH ══════════════════
HX, HW, HY, HH = 1.0, 98.0, 12.0, 24.5
panel(HX, HW, HY, HH, "d", "Nine heads, all reading the 176-dim shared representation")
HEADS = [("element", "5", "2.5", INK, "REPORTED - the target"),
         ("organism", "32", "1.0", RED, "adversarial · GRL λ=0.005"),
         ("kingdom", "4", "0.5", RED, "adversarial · GRL λ=0.08"),
         ("human UTR/enhancer", "3", "1.0", MUTED, "auxiliary · masked to human"),
         ("biological group", "4", "0.8", MUTED, "auxiliary"),
         ("coding detector", "2", "0.5", MUTED, "auxiliary"),
         ("start/stop pair", "2", "0.4", MUTED, "auxiliary"),
         ("coarse element", "7", "0.3", MUTED, "auxiliary"),
         ("UTR subtype", "2", "0.2", MUTED, "auxiliary")]
colw = 32.0
for i, (nm, ncls, w, c, role) in enumerate(HEADS):
    col, row = i // 3, i % 3
    x = HX + 2.0 + col * colw
    yy = HY + HH - 9.4 - row * 5.0
    lw = 1.9 if c == INK else (1.4 if c == RED else 0.9)
    box(x, yy, colw - 3.4, 4.2, fc="white", ec=c if c != MUTED else RULE, lw=lw)
    ax.text(x + 1.6, yy + 2.9, nm, fontsize=7.6, color=INK, va="center",
            fontweight="bold" if c == INK else "normal")
    ax.text(x + 1.6, yy + 1.1, role, fontsize=6.3, color=MUTED, va="center", style="italic")
    ax.text(x + colw - 4.8, yy + 2.9, f"w {w}", fontsize=7.4, color=INK, va="center",
            ha="right", family="monospace")
    ax.text(x + colw - 4.8, yy + 1.1, f"{ncls} cls", fontsize=6.3, color=MUTED, va="center",
            ha="right")
ax.text(HX + 2.0, HY + 2.6,
        "Adversarial heads (red) train through gradient reversal: their weight drives the trunk to SUPPRESS taxonomy, not predict it, and they are\n"
        "excluded from the auxiliary subtotal. The six auxiliary heads total 3.2 - more than the element head's 2.5. No taxonomic identity reaches\n"
        "the element head: it is suppressed in the representation and withheld as an input.",
        fontsize=6.9, color=INK, va="center", linespacing=1.6)

# ══════════════════ footer ══════════════════
ax.text(1.0, 8.8,
        "Two controls in Section 3.5 show the panel (a) weighting does not drive the attributions: stack shares hbond's ×2.0 yet is worth 0.035 at\n"
        "promoters against hbond's 0.447; and inter, weighted lowest at ×1.0, carries the largest effect in the whole matrix at the start codon (0.518).",
        fontsize=7.0, color=RED, va="top", linespacing=1.65)
ax.text(1.0, 2.8,
        "Verified against the trained checkpoint (dump_model_summary.py): 2,201,778 parameters profiles-only (81 layers); 2,626,034 with the\n"
        "sequence branch (93 layers).",
        fontsize=6.8, color=MUTED, va="top", linespacing=1.6)

for ext in ("pdf", "png"):
    fig.savefig(f"figures/figS_architecture.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
print("wrote figures/figS_architecture.pdf / .png   (rev 4 - min font ~6.2pt, body 7-7.5pt)")
