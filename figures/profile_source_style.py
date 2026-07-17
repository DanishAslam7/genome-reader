#!/usr/bin/env python3
"""
Real biophysical-profile figures in the source-study style (Sharma et al. 2025):
mean (+/- SD) of each of the 7 canonical parameters vs position, centered on a
genomic landmark. 2x4 grid per (organism, element).

Fast path (this script): the assembled diatom caches diatom_cache/<org>_total_<elem>.npy
(shape n x 7 x 475), param order [bbone,bp,hbond,inter,intra,sol,stack].

For the 32 cohort organisms use plot_element_profiles.py (same style, reads the
per-organism CSV tree) — run it on the cluster; the CSVs are ~270 MB each.

  python figures/profile_source_style.py
  -> figures/element_profiles_enhanced/<organism>/<element>.{png}
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORGS = {"pt": "Phaeodactylum tricornutum", "tp": "Thalassiosira pseudonana"}  # both diatoms
ELEMS = {"gs":"gene start","ge":"gene end","es":"exon start","ee":"exon end",
         "cds":"CDS","stac":"start codon","stoc":"stop codon"}
PARAMS = ["bbone","bp","hbond","inter","intra","sol","stack"]
PLABEL = {"bbone":"Backbone geometry","bp":"Base-pair axis","hbond":"Hydrogen bonding",
          "inter":"Inter-bp step (tetranucleotide)","intra":"Intra-bp geometry",
          "sol":"Solvation","stack":"Base stacking"}
ACCENT = {"hbond","stack"}
TRACE, ACC, INK, MUTED, DARKRED = "#2b5f9e","#c2418c","#1b1a24","#6b6577","#a01b3a"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],
                     "pdf.fonttype":42,"ps.fonttype":42})
OUT = "figures/element_profiles_enhanced"

def make(org_slug, org_name, elem, elem_name):
    path = f"diatom_cache/{org_slug}_total_{elem}.npy"
    if not os.path.exists(path): return False
    X = np.load(path, mmap_mode="r")            # (n,7,475)
    n = X.shape[0]
    mean = np.asarray(X).mean(0); std = np.asarray(X).std(0)
    mean, std = mean[:, 3:-3], std[:, 3:-3]     # trim tetranucleotide edge artefact
    pos = np.arange(mean.shape[1]) - mean.shape[1]//2
    fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.3)); fig.patch.set_facecolor("white")
    for i, p in enumerate(PARAMS):
        ax = axes.flat[i]; c = ACC if p in ACCENT else TRACE
        ax.axvline(0, color="#cfc9d8", lw=1.0, ls=(0,(3,3)), zorder=1)
        ax.fill_between(pos, mean[i]-std[i], mean[i]+std[i], color=c, alpha=0.13, lw=0, zorder=2)
        ax.plot(pos, mean[i], color=c, lw=1.7, zorder=3)
        ax.set_title(PLABEL[p] + ("   ×2" if p in ACCENT else ""),
                     color=DARKRED if p in ACCENT else INK, fontsize=9.6, fontweight="bold")
        ax.set_xlabel("position (bp, centered)", fontsize=8, color=MUTED)
        ax.set_ylabel("normalized profile", fontsize=8, color=MUTED)
        ax.tick_params(labelsize=7.5); ax.grid(True, color="#eeecf3", lw=0.8); ax.set_axisbelow(True)
        for s in ("top","right"): ax.spines[s].set_visible(False)
        ax.set_xlim(pos[0], pos[-1])
    cap = axes.flat[7]; cap.axis("off")
    cap.text(0.02,0.94,f"{elem_name.capitalize()}\n{org_name}\n(Bacillariophyta / diatom)",
             fontsize=11,fontweight="bold",color=INK,va="top",linespacing=1.4,transform=cap.transAxes)
    cap.text(0.02,0.5,f"n = {n:,} windows\nmean ± SD of the real\nbiophysical profile\n\n"
             "H-bonding and stacking\n(magenta, ×2) carry the\nfixed prior weighting the\nmodel inherits.",
             fontsize=8.6,color=MUTED,va="top",linespacing=1.5,transform=cap.transAxes)
    fig.suptitle(f"Biophysical profile of the {elem_name} — {org_name}",
                 fontsize=14,fontweight="bold",color=INK,y=0.995)
    fig.tight_layout(rect=[0,0,1,0.965])
    d = os.path.join(OUT, org_slug); os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, f"{elem}.png"), dpi=200, facecolor="white"); plt.close(fig)
    return True

if __name__ == "__main__":
    made = 0
    for slug, name in ORGS.items():
        for elem, en in ELEMS.items():
            if make(slug, name, elem, en):
                made += 1; print(f"  {slug}/{elem}.png")
    print(f"wrote {made} figures to {OUT}/")
