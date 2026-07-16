#!/usr/bin/env python3
"""Faithful port of the dataviz skill's validate_palette.js (local node is v12, too
old for the ES module). Same Machado-Oliveira-Fernandes 2009 severity-1.0 matrices,
same OKLab dE*100, same thresholds. Verified to reproduce the skill's own shipped
8-hue report exactly (worst adjacent protan dE 9.1, normal-vision floor 19.6).

Kept in the repo so the paper's figure palettes stay checkable.

  python figures/validate_palette.py "#hex,#hex,..." --mode light [--pairs all]
"""
import sys, math, itertools

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
DEFAULT_SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602],
               [0.004733, 0.691367, 0.303900]],
}

hex2srgb = lambda h: [int(h.strip().lstrip("#")[i:i+2], 16) / 255 for i in (0, 2, 4)]
s2lin = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
lin = lambda h: [s2lin(c) for c in hex2srgb(h)]


def rel_lum(h):
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted([rel_lum(a), rel_lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_from_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708*r + 0.5363325363*g + 0.0514459929*b) ** (1/3)
    m = (0.2119034982*r + 0.6806995451*g + 0.1073969566*b) ** (1/3)
    s = (0.0883024619*r + 0.2817188376*g + 0.6299787005*b) ** (1/3)
    return [0.2104542553*l + 0.7936177850*m - 0.0040720468*s,
            1.9779984951*l - 2.4285922050*m + 0.4505937099*s,
            0.0259040371*l + 0.7827717662*m - 0.8086757660*s]


def oklch(h):
    L, a, b = oklab_from_lin(lin(h))
    return L, math.hypot(a, b)


def simulate(h, kind):
    r, g, b = lin(h); M = MACHADO[kind]
    return [min(1, max(0, M[i][0]*r + M[i][1]*g + M[i][2]*b)) for i in range(3)]


def delta_e(h1, h2, kind=None):
    a = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100 * math.dist(a, b)


def validate(palette, mode="light", surface=None, pairs="adjacent"):
    surface = surface or DEFAULT_SURFACE[mode]
    lo, hi = BAND[mode]
    ok = True
    print(f"\n  mode={mode}  surface={surface}  pairs={pairs}  n={len(palette)}")
    print("  " + "-" * 76)
    off = [(c, round(oklch(c)[0], 3)) for c in palette if not (lo <= oklch(c)[0] <= hi)]
    ok &= not off
    print(f"  {'Lightness band':22s} {'FAIL' if off else 'pass':6s} " +
          (f"outside L {lo}-{hi}: {off}" if off else f"all inside L {lo}-{hi}"))
    lowc = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    ok &= not lowc
    print(f"  {'Chroma floor':22s} {'FAIL' if lowc else 'pass':6s} " +
          (f"reads gray: {lowc}" if lowc else f"all >= {CHROMA_FLOOR}"))
    n = len(palette)
    pl = list(itertools.combinations(range(n), 2)) if pairs == "all" else [(i, i+1) for i in range(n-1)]
    worst = min(((delta_e(palette[i], palette[j], k), k, palette[i], palette[j])
                 for k in ("protan", "deutan") for i, j in pl), key=lambda t: t[0])
    tri = min(delta_e(palette[i], palette[j], "tritan") for i, j in pl)
    wd = worst[0]
    state = "pass" if wd >= CVD_TARGET else ("FLOOR" if wd >= CVD_FLOOR else "FAIL")
    ok &= state != "FAIL"
    print(f"  {'CVD separation':22s} {state:6s} worst {worst[3]}<->{worst[2]} dE {wd:.1f} ({worst[1]}) · tritan {tri:.1f}")
    nworst = min(((delta_e(palette[i], palette[j]), palette[i], palette[j]) for i, j in pl), key=lambda t: t[0])
    nd = nworst[0]
    nstate = "pass" if nd >= NORMAL_FLOOR else "FAIL"
    ok &= nstate == "pass"
    print(f"  {'Normal-vision floor':22s} {nstate:6s} worst {nworst[2]}<->{nworst[1]} dE {nd:.1f}")
    low = [(c, round(contrast(c, surface), 2)) for c in palette if contrast(c, surface) < CONTRAST_MIN]
    print(f"  {'Contrast vs surface':22s} {'RELIEF' if low else 'pass':6s} " +
          (f"below {CONTRAST_MIN}:1 (needs labels/table): {low}" if low else f"all >= {CONTRAST_MIN}:1"))
    print(f"  => {'OK' if ok else 'HAS FAILURES'}")
    return ok


if __name__ == "__main__":
    pal = [c for c in sys.argv[1].split(",") if c.strip()]
    mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "light"
    surface = sys.argv[sys.argv.index("--surface") + 1] if "--surface" in sys.argv else None
    pairs = sys.argv[sys.argv.index("--pairs") + 1] if "--pairs" in sys.argv else "adjacent"
    sys.exit(0 if validate(pal, mode, surface, pairs) else 1)
