#!/usr/bin/env python3
"""
Real biophysical-profile figures (source-study style, Sharma et al. 2025).

2x4 grid: 7 panels, one per biophysical parameter (mean +/- SD vs centered
position), plus an 8th "scale box" that carries the colour key, the meaning of
the x2 flag, and n. No figure heading; identity (organism/element) goes in the
caption. Parameters are labelled by their short codes.

Data (fast path): diatom_cache/<org>_total_<elem>.npy, shape (n, 7, 475),
param axis order [bbone, bp, hbond, inter, intra, sol, stack].

  python figures/profile_source_style.py
  -> figures/element_profiles_enhanced/<org>/<elem>.png
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORGS = {"pt": "Phaeodactylum tricornutum", "tp": "Thalassiosira pseudonana"}
ELEMS = {"gs":"gene start","ge":"gene end","es":"exon start","ee":"exon end",
         "cds":"CDS","stac":"start codon","stoc":"stop codon"}
PARAMS = ["bbone","bp","hbond","inter","intra","sol","stack"]   # data axis order
X2 = {"hbond","stack"}                     # channels up-weighted x2 at the model input
RED, PINK, MUTED, INK = "#c0504d", "#c2418c", "#6b6577", "#1b1a24"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],
                     "pdf.fonttype":42,"ps.fonttype":42})
OUT = "figures/element_profiles_enhanced"

def make(slug, name, elem, elem_name):
    path = f"diatom_cache/{slug}_total_{elem}.npy"
    if not os.path.exists(path):
        return False
    try:
        X = np.load(path, mmap_mode="r")
        mean = np.asarray(X).mean(0); std = np.asarray(X).std(0)
    except Exception as e:
        print(f"  skip {slug}/{elem}: {type(e).__name__}"); return False
    n = X.shape[0]
    mean, std = mean[:, 3:-3], std[:, 3:-3]          # trim tetranucleotide edge cols
    pos = np.arange(mean.shape[1]) - mean.shape[1]//2

    fig, axes = plt.subplots(2, 4, figsize=(13.0, 5.9)); fig.patch.set_facecolor("white")
    for i, p in enumerate(PARAMS):
        ax = axes.flat[i]; c = PINK if p in X2 else RED
        ax.axvline(0, color="#cfc9d8", lw=1.0, ls=(0,(3,3)), zorder=1)
        ax.fill_between(pos, mean[i]-std[i], mean[i]+std[i], color=c, alpha=0.13, lw=0, zorder=2)
        ax.plot(pos, mean[i], color=c, lw=1.7, zorder=3)
        ax.text(0.045, 0.93, p + ("  ×2" if p in X2 else ""), transform=ax.transAxes,
                color=c, fontsize=11, fontweight="bold", va="top", ha="left")
        ax.tick_params(labelsize=7.5); ax.grid(True, color="#eeecf3", lw=0.8); ax.set_axisbelow(True)
        for s in ("top","right"): ax.spines[s].set_visible(False)
        ax.set_xlim(pos[0], pos[-1])
        if i >= 3: ax.set_xlabel("position (bp)", fontsize=8, color=MUTED)
        if i in (0, 4): ax.set_ylabel("normalized profile", fontsize=8, color=MUTED)

    # ---- 8th cell: scale box ----
    sb = axes.flat[7]; sb.axis("off")
    T = sb.transAxes
    sb.add_patch(plt.Rectangle((0.02,0.03),0.96,0.94,transform=T,fill=False,ec="#d6dae0",lw=1.1))
    sb.plot([0.08,0.20],[0.88,0.88],color=PINK,lw=3.2,transform=T,solid_capstyle="round")
    sb.text(0.24,0.88,"hbond, stack",transform=T,fontsize=10,fontweight="bold",color=PINK,va="center")
    sb.text(0.08,0.79,"up-weighted ×2 at the model input",transform=T,fontsize=8,color=MUTED,va="center")
    sb.plot([0.08,0.20],[0.66,0.66],color=RED,lw=3.2,transform=T,solid_capstyle="round")
    sb.text(0.24,0.66,"bbone, bp, inter, intra, sol",transform=T,fontsize=10,fontweight="bold",color=RED,va="center")
    sb.text(0.08,0.57,"the other five parameters",transform=T,fontsize=8,color=MUTED,va="center")
    sb.text(0.08,0.42,"×2 = fixed prior channel weighting\n(Sharma et al. 2025); the four red\n"
            "×1.4, inter ×1.0",transform=T,fontsize=7.6,color=INK,va="top",linespacing=1.4)
    sb.text(0.08,0.20,"shaded band = mean ± SD",transform=T,fontsize=8,color=MUTED,va="center")
    sb.text(0.08,0.11,f"n = {n:,} windows",transform=T,fontsize=9,fontweight="bold",color=INK,va="center")

    fig.tight_layout()
    d = os.path.join(OUT, slug); os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, f"{elem}.png"), dpi=200, facecolor="white"); plt.close(fig)
    return True

if __name__ == "__main__":
    made = 0
    for slug, name in ORGS.items():
        for elem, en in ELEMS.items():
            if make(slug, name, elem, en):
                made += 1; print(f"  {slug}/{elem}.png")
    print(f"wrote {made} figures to {OUT}/")
