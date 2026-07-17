#!/usr/bin/env python3
"""
Real biophysical-profile figure in the source-study style (Sharma et al. 2025):
the mean (+/- SD) of each of the 7 canonical parameters vs position, centered on a
genomic landmark, for one organism / element.

Data: diatom_cache/pt_total_<elem>.npy  — real profiles, shape (n_windows, 7, 475),
for Phaeodactylum tricornutum. Param axis order is the canonical
[bbone, bp, hbond, inter, intra, sol, stack] (eval_diatoms.py:26).

  python figures/profile_source_style.py  ->  figures/profile_source_style.{png,pdf}
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORG   = "Phaeodactylum tricornutum"
ELEM  = "stac"                       # start codon — a clean landmark
ELEM_NAME = "start codon"
PARAMS = ["bbone","bp","hbond","inter","intra","sol","stack"]
PLABEL = {"bbone":"Backbone geometry","bp":"Base-pair axis","hbond":"Hydrogen bonding",
          "inter":"Inter-bp step (tetranucleotide)","intra":"Intra-bp geometry",
          "sol":"Solvation","stack":"Base stacking"}
# de-emphasise / emphasise to echo the prior's start/stop weighting (disclosed)
ACCENT = {"hbond":"#c2418c","stack":"#c2418c"}          # up-weighted x2 in the model
TRACE, BAND, INK, MUTED, DARKRED = "#2b5f9e","#2b5f9e","#1b1a24","#6b6577","#a01b3a"

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],
                     "pdf.fonttype":42,"ps.fonttype":42})

X = np.load(f"diatom_cache/pt_total_{ELEM}.npy", mmap_mode="r")   # (n,7,475)
n = X.shape[0]
mean = np.asarray(X).mean(axis=0)      # (7,475)
std  = np.asarray(X).std(axis=0)       # (7,475)
pos  = np.arange(mean.shape[1]) - mean.shape[1]//2   # centered on landmark
# trim boundary columns: the tetranucleotide 'inter' track is undefined at the
# very edges of the window and reads 0 there — drop 3 positions each side.
mean = mean[:, 3:-3]; std = std[:, 3:-3]; pos = pos[3:-3]

fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.3))
fig.patch.set_facecolor("white")
for i, p in enumerate(PARAMS):
    ax = axes.flat[i]
    c = ACCENT.get(p, TRACE)
    ax.axvline(0, color="#cfc9d8", lw=1.0, ls=(0,(3,3)), zorder=1)
    ax.fill_between(pos, mean[i]-std[i], mean[i]+std[i], color=c, alpha=0.13, lw=0, zorder=2)
    ax.plot(pos, mean[i], color=c, lw=1.7, zorder=3)
    ax.set_title(PLABEL[p] + ("   ×2" if p in ACCENT else ""),
                 color=DARKRED if p in ACCENT else INK, fontsize=9.6, fontweight="bold")
    ax.set_xlabel("position (bp, centered)", fontsize=8, color=MUTED)
    ax.set_ylabel("normalized profile", fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, color="#eeecf3", lw=0.8); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.set_xlim(pos[0], pos[-1])

# caption cell
cap = axes.flat[7]; cap.axis("off")
cap.text(0.02, 0.94,
         f"{ELEM_NAME.capitalize()}\n{ORG}\n(Bacillariophyta / diatom)",
         fontsize=11, fontweight="bold", color=INK, va="top", linespacing=1.4, transform=cap.transAxes)
cap.text(0.02, 0.52,
         f"n = {n:,} windows\nmean ± SD of the real\nbiophysical profile\n\n"
         "H-bonding and stacking\n(magenta, ×2) carry the\nfixed prior weighting the\n"
         "model inherits.",
         fontsize=8.6, color=MUTED, va="top", linespacing=1.5, transform=cap.transAxes)

fig.suptitle(f"Biophysical profile of the {ELEM_NAME} — {ORG}",
             fontsize=14, fontweight="bold", color=INK, y=0.995)
fig.tight_layout(rect=[0,0,1,0.965])
fig.savefig("figures/profile_source_style.png", dpi=300, facecolor="white")
fig.savefig("figures/profile_source_style.pdf", facecolor="white")
print(f"wrote figures/profile_source_style.{{png,pdf}}  (n={n} windows, {ORG}, {ELEM_NAME})")
