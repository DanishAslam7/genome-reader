#!/usr/bin/env python3
"""Per-(organism, element) mean biophysical-profile traces — structure-&-dynamics style.

For every (kingdom/organism/element) in the reprofiled legacy tree, plot the mean
(+/-SD band) of each of the 7 canonical biophysical tracks vs centered position —
matching the fig7 aesthetic in claude_opus/make_projection_figures.py, but one
figure per organism x element (all 32 organisms: old, new, and the balancing 6).

Data source: <profile-root>/<kingdom>/<organism>/<element>/<track>_norm_<label>_<organism>.csv
(rows = sequences, cols = positions). Averaging is chunked so huge files
(human/wheat, 100k-500k rows) stay memory-safe.

  python plot_element_profiles.py                         # all organisms/elements
  python plot_element_profiles.py --organisms danio_rerio giardia_intestinalis
  python plot_element_profiles.py --elements cds gs       # element short-codes
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent
KINGDOMS = ["animalia", "fungi", "plantae", "protista"]

# Canonical 7 tracks (matches run_profile_batch CANONICAL_ORDER) + display labels.
TRACKS = ["bbone", "bp", "hbond", "intra", "sol", "stack", "inter"]
TRACK_LABEL = {
    "bbone": "Backbone", "bp": "Base-pair", "hbond": "H-bond", "intra": "Intra-bp",
    "sol": "Solvation", "stack": "Stacking", "inter": "Inter-bp (tetramer)",
}
# element short-code -> readable name
ELEMENT_NAME = {
    "gs": "Gene start", "ge": "Gene end", "es": "Exon start", "ee": "Exon end",
    "cds": "CDS", "stac": "Start codon", "stoc": "Stop codon", "prom": "Promoter",
    "5utrs": "5'UTR start", "5utre": "5'UTR end", "3utrs": "3'UTR start", "3utre": "3'UTR end",
    "ens": "Enhancer start", "ene": "Enhancer end",
}
DARK_RED = "#8B0000"
LINE = "#C0504D"
ACCENT_COL = "#c2418c"     # H-bond & stacking — the channels the model up-weights x2
ACCENT_TITLE = "#a01b3a"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 200,
                     "axes.titleweight": "bold", "axes.titlesize": 10})


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile-root", type=Path, default=REPO,
                    help="Root holding <kingdom>/<organism>/<element>/ (default: repo root).")
    ap.add_argument("--out-dir", type=Path, default=REPO / "figures" / "element_profiles")
    ap.add_argument("--organisms", nargs="*", default=None, help="Subset of organisms.")
    ap.add_argument("--elements", nargs="*", default=None, help="Subset of element short-codes.")
    ap.add_argument("--chunksize", type=int, default=50000)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def mean_std_chunked(path: Path, chunksize: int):
    """Memory-safe per-column mean/std over all rows of a profile CSV."""
    n = 0
    s = ss = None
    for chunk in pd.read_csv(path, index_col=0, chunksize=chunksize):
        v = chunk.to_numpy(dtype=np.float64)
        cs, css = v.sum(0), (v * v).sum(0)
        s = cs if s is None else s + cs
        ss = css if ss is None else ss + css
        n += len(v)
    if n == 0:
        return None, None, 0
    mean = s / n
    var = np.maximum(ss / n - mean ** 2, 0.0)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32), n


def find_track_csv(element_dir: Path, track: str) -> Path | None:
    hits = sorted(element_dir.glob(f"{track}_norm_*.csv"))
    return hits[0] if hits else None


def plot_element(element_dir: Path, kingdom: str, organism: str, element: str,
                 out_path: Path, chunksize: int) -> bool:
    series = {}
    n_seq = None
    for tr in TRACKS:
        csv = find_track_csv(element_dir, tr)
        if csv is None:
            continue
        mean, std, n = mean_std_chunked(csv, chunksize)
        if mean is None:
            continue
        series[tr] = (mean, std)
        n_seq = n
    if not series:
        return False

    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.2))
    axes = axes.ravel()
    for i, tr in enumerate(TRACKS):
        ax = axes[i]
        if tr not in series:
            ax.axis("off")
            continue
        mean, std = series[tr]
        if len(mean) > 8:                             # trim boundary cols (tetranucleotide
            mean, std = mean[3:-3], std[3:-3]          # 'inter' is undefined at the edges)
        x = np.arange(len(mean)) - len(mean) // 2     # centre landmark at 0
        accent = tr in ("hbond", "stack")             # channels the model up-weights x2
        c = ACCENT_COL if accent else LINE
        ax.plot(x, mean, color=c, lw=1.4)
        ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.15, linewidth=0)
        ax.axvline(0, color="#999999", lw=0.6, ls=":")
        ax.set_title(TRACK_LABEL[tr] + ("  ×2" if accent else ""),
                     color=ACCENT_TITLE if accent else DARK_RED)
        ax.set_xlabel("position (centered)")
        ax.set_ylabel("normalized profile")
        ax.grid(color="#eeeeee", lw=0.5)
        ax.set_axisbelow(True)
    axes[7].axis("off")
    axes[7].text(0.5, 0.5,
                 f"{ELEMENT_NAME.get(element, element)}\n{organism}\n({kingdom})\n\nn = {n_seq:,} sequences\nmean ± SD",
                 ha="center", va="center", fontsize=12, color=DARK_RED, fontweight="bold")
    fig.suptitle(f"Mean biophysical profile — {ELEMENT_NAME.get(element, element)} · {organism}",
                 color=DARK_RED, fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def main() -> int:
    args = parse_args()
    root = args.profile_root.resolve()
    made = skipped = 0
    for kingdom in KINGDOMS:
        kdir = root / kingdom
        if not kdir.is_dir():
            continue
        for org_dir in sorted(p for p in kdir.iterdir() if p.is_dir()):
            organism = org_dir.name
            if args.organisms and organism not in args.organisms:
                continue
            for elem_dir in sorted(p for p in org_dir.iterdir() if p.is_dir()):
                element = elem_dir.name
                if args.elements and element not in args.elements:
                    continue
                if not any(elem_dir.glob("*_norm_*.csv")):
                    continue
                out_path = args.out_dir / kingdom / organism / f"{element}.png"
                if out_path.exists() and not args.overwrite:
                    skipped += 1
                    continue
                ok = plot_element(elem_dir, kingdom, organism, element, out_path, args.chunksize)
                if ok:
                    made += 1
                    print(f"  {kingdom}/{organism}/{element} -> {out_path}", flush=True)
    print(f"\nDone. figures made={made}, skipped(existing)={skipped}. Out: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
