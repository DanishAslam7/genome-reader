#!/usr/bin/env python3
"""Re-derive Table S3/S4 parameter-specific segment windows on the current cohort.

Strategy (CESL — Consistent Effect-Size Localization), per (kingdom, parameter, element):

  Pass 1 (selection), per profile position p, pooled over the kingdom's organisms:
    self-saliency  a(p) = |mean_e(p) - baseline_e|          (element's OWN feature)
    discriminability d(p) = |mean_e(p) - mean_rest(p)|       (vs other elements)
    consistency    c(p) = frac. of organisms whose deviation sign agrees
    FDR gate       g(p) = 1 if one-way ANOVA across elements is BH-FDR significant
    score          s(p) = a(p)*d(p)*c(p)*g(p)
  Window = contiguous max-sum interval of s above its in-range baseline, snapped to a
  5-nt grid, >= min length, within landmark +/- search. A flatness gate (mean-profile
  amplitude) yields NO WINDOW for featureless elements (e.g. CDS) -- matching S3/S4,
  which exclude CDS.

  Pass 2 (paper's significance report) on the chosen window: per-sequence segment
  averages -> one-way ANOVA across elements + Tukey-Kramer post-hoc + BH-FDR.

Why effect-size, not p-values: at tens-of-thousands of sequences per element, ANOVA/
Tukey p-values saturate (significant everywhere), so selection is driven by magnitude
+ organism-consistency, with FDR kept only as a gate/report.

Coordinates: profiles are 25-nt-smoothed (length L ~475/474); output windows are mapped
back to the paper's 501-bp / center-250 convention so they slot into TABLE_WINDOWS.

  python derive_table_windows.py            # all kingdoms, default knobs
  python derive_table_windows.py --include-cds
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as fdist
# Tukey-Kramer p-value via the studentized-range distribution (scipy >=1.7);
# fall back to a t-based pairwise p if unavailable. Avoids a statsmodels dependency
# (genome_reader_tf lacks it and the cluster has no internet to pip-install).
try:
    from scipy.stats import studentized_range as _srange
    def tukey_p(q, k, df):
        return float(_srange.sf(q, k, df))
except Exception:                                     # pragma: no cover
    from scipy.stats import t as _tdist
    def tukey_p(q, k, df):
        return float(2.0 * _tdist.sf(q / (2.0 ** 0.5), df))


def bh_reject(pvals, alpha):
    """Benjamini-Hochberg FDR: boolean reject array in original order."""
    import numpy as _np
    p = _np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return _np.zeros(0, dtype=bool)
    order = _np.argsort(p)
    thresh = alpha * (_np.arange(1, n + 1) / n)
    ok = p[order] <= thresh
    rej = _np.zeros(n, dtype=bool)
    if ok.any():
        kmax = int(_np.max(_np.nonzero(ok)))
        rej[order[: kmax + 1]] = True
    return rej


REPO = Path(__file__).resolve().parent
KINGDOMS = ["animalia", "fungi", "plantae", "protista"]
STRUCTURAL = ["bp", "inter", "intra", "bbone"]
ENERGETIC = ["hbond", "stack", "sol"]
TRACKS = STRUCTURAL + ENERGETIC                       # csv name == track name
SITE_ORDER = ["prom", "gs", "ge", "es", "ee", "stac", "stoc"]   # matches TABLE_SITE_ORDER
ELEMENT_NAME = {"prom": "Promoters", "gs": "Gene Start", "ge": "Gene End",
                "es": "Exon Start", "ee": "Exon End", "stac": "Start Codon",
                "stoc": "Stop Codon", "cds": "CDS"}
TABLE_CENTER = 250


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile-root", type=Path, default=REPO)
    ap.add_argument("--out-dir", type=Path, default=REPO / "table_windows_rederived")
    ap.add_argument("--include-cds", action="store_true", help="Also evaluate CDS.")
    ap.add_argument("--grid", type=int, default=5, help="Snap window edges to this step.")
    ap.add_argument("--min-len", type=int, default=30, help="Min window length (nt).")
    ap.add_argument("--max-len", type=int, default=120, help="Max window length (nt).")
    ap.add_argument("--peak-frac", type=float, default=0.5,
                    help="Window extends from the score peak while score >= frac*peak.")
    ap.add_argument("--search", type=int, default=150, help="Search range = landmark +/- this.")
    ap.add_argument("--cds-factor", type=float, default=2.0,
                    help="Assign a window only if amplitude >= cds_amp * this factor "
                         "(per kingdom,track). CDS is the flattest site; validated min "
                         "real/cds ratio ~2.75, so 2.0 keeps all real elements, drops CDS.")
    ap.add_argument("--amp-min", type=float, default=0.005,
                    help="Small absolute amplitude floor (normalized units), in addition to "
                         "the CDS-relative gate. Set 0 + --cds-factor 0 for a diagnostic run "
                         "that windows everything incl CDS.")
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--chunksize", type=int, default=50000)
    ap.add_argument("--max-rows-per-file", type=int, default=0,
                    help="Use only the first N sequences/file for window stats (0=all). "
                         "~15-20k is plenty and is much faster on big files.")
    ap.add_argument("--resume", action="store_true",
                    help="Continue a crashed run: keep rows already in the long CSV and skip "
                         "those (kingdom,track) groups (rows are written incrementally per track).")
    ap.add_argument("--no-edge-guard", action="store_true",
                    help="Disable rejecting windows that touch the search boundary.")
    ap.add_argument("--edge-margin", type=int, default=2,
                    help="Drop a window if it comes within this many positions of the search edge.")
    return ap.parse_args()


def track_csv(elem_dir: Path, track: str) -> Path | None:
    hits = sorted(elem_dir.glob(f"{track}_norm_*.csv"))
    return hits[0] if hits else None


def col_moments(path: Path, chunksize: int, max_rows: int = 0):
    """Per-column n, sum, sumsq. If max_rows>0, use only the first max_rows rows
    (fast; window stats are robust to subsampling at these counts)."""
    if max_rows and max_rows > 0:
        v = pd.read_csv(path, index_col=0, nrows=max_rows).to_numpy(dtype=np.float64)
        return len(v), v.sum(0), (v * v).sum(0)
    n = 0
    s = ss = None
    for chunk in pd.read_csv(path, index_col=0, chunksize=chunksize):
        v = chunk.to_numpy(dtype=np.float64)
        cs, css = v.sum(0), (v * v).sum(0)
        s = cs if s is None else s + cs
        ss = css if ss is None else ss + css
        n += len(v)
    return n, s, ss


def seg_moments(path: Path, lo: int, hi: int, chunksize: int, max_rows: int = 0):
    """n, sum, sumsq of per-row mean over columns [lo, hi] (for the chosen window)."""
    if max_rows and max_rows > 0:
        v = pd.read_csv(path, index_col=0, nrows=max_rows).to_numpy(dtype=np.float64)[:, lo:hi + 1].mean(1)
        return len(v), float(v.sum()), float((v * v).sum())
    n = 0
    s = ss = 0.0
    for chunk in pd.read_csv(path, index_col=0, chunksize=chunksize):
        v = chunk.to_numpy(dtype=np.float64)[:, lo:hi + 1].mean(1)
        s += v.sum()
        ss += (v * v).sum()
        n += len(v)
    return n, s, ss


def anova_from_moments(means, vars, ns):
    """Vectorized one-way ANOVA across groups at each position. means/vars: (k, L)."""
    ns = ns.astype(np.float64)
    N = ns.sum()
    k = len(ns)
    grand = (ns[:, None] * means).sum(0) / N
    ssb = (ns[:, None] * (means - grand) ** 2).sum(0)
    ssw = ((ns[:, None] - 1.0) * vars).sum(0)
    dfb, dfw = k - 1, N - k
    msw = ssw / dfw
    F = (ssb / dfb) / np.maximum(msw, 1e-12)
    p = fdist.sf(F, dfb, dfw)
    return F, p, msw, dfw


def tukey_fdr(seg_means, seg_vars, seg_ns, target_idx, alpha):
    """Tukey-Kramer pairwise + BH-FDR on segment-averaged group stats; return
    (n_sig_pairs_involving_target, total_pairs)."""
    ns = seg_ns.astype(np.float64)
    k = len(ns)
    dfw = ns.sum() - k
    msw = ((ns - 1.0) * seg_vars).sum() / dfw
    pairs, pvals = [], []
    for i in range(k):
        for j in range(i + 1, k):
            se = np.sqrt(max(msw, 1e-12) / 2.0 * (1.0 / ns[i] + 1.0 / ns[j]))
            q = abs(seg_means[i] - seg_means[j]) / max(se, 1e-12)
            pvals.append(tukey_p(q, k, dfw))
            pairs.append((i, j))
    rej = bh_reject(pvals, alpha)
    n_target = sum(1 for (i, j), r in zip(pairs, rej) if r and target_idx in (i, j))
    return n_target, len(pairs)


def localize_window(score, lo, hi, frac, min_len, max_len, grid):
    """Tight window around the dominant score peak: extend from the peak while
    score >= frac * peak (FWHM-like). Snap to grid; clamp to [min_len, max_len]."""
    pk = lo + int(np.argmax(score[lo:hi + 1]))
    if score[pk] <= 0:
        return None
    thr = frac * score[pk]
    ws = pk
    while ws > lo and score[ws - 1] >= thr:
        ws -= 1
    we = pk
    while we < hi and score[we + 1] >= thr:
        we += 1
    if we - ws + 1 < min_len:                       # widen symmetrically
        need = min_len - (we - ws + 1)
        ws = max(lo, ws - need // 2); we = min(hi, ws + min_len - 1)
    if we - ws + 1 > max_len:                        # cap width around the peak
        ws = max(lo, pk - max_len // 2); we = min(hi, ws + max_len - 1)
    ws = (ws // grid) * grid
    we = int(np.ceil((we + 1) / grid) * grid) - 1
    return max(lo, ws), min(hi, we)


def main() -> int:
    args = parse_args()
    root = args.profile_root.resolve()
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    # Always load CDS: it's the per-(kingdom,track) flatness reference for the gate.
    sites = SITE_ORDER + ["cds"]

    # Long-form rows are written incrementally per (kingdom,track) so a crash is resumable.
    csv_path = out / "rederived_windows_long.csv"
    fieldnames = ["kingdom", "parameter", "element", "element_name", "assigned",
                  "table_start", "table_end", "length", "amplitude", "cds_amp_ref",
                  "amp_floor", "max_score", "mean_consistency", "n_organisms",
                  "tukey_fdr_sig_pairs", "tukey_total_pairs"]
    resume_active = bool(args.resume and csv_path.exists() and csv_path.stat().st_size > 0)
    done_tracks: set[tuple[str, str]] = set()
    if resume_active:
        prev = pd.read_csv(csv_path)
        done_tracks = set(map(tuple, prev[["kingdom", "parameter"]].drop_duplicates().to_numpy()))
        print(f"RESUME: {len(done_tracks)} (kingdom,track) groups already in CSV; skipping them")
    csv_fh = csv_path.open("a" if resume_active else "w", newline="")
    writer = _csv.DictWriter(csv_fh, fieldnames=fieldnames)
    if not resume_active:
        writer.writeheader()
        csv_fh.flush()

    for kingdom in KINGDOMS:
        kdir = root / kingdom
        if not kdir.is_dir():
            continue
        organisms = sorted(p.name for p in kdir.iterdir() if p.is_dir())
        for track in TRACKS:
            if (kingdom, track) in done_tracks:
                print(f"  {kingdom}/{track}: (resume: already done, skipped)", flush=True)
                continue
            track_rows = []
            # ---- pass 1: per-position pooled moments + per-organism means ----
            elem_n, elem_mean, elem_var, elem_orgmeans = {}, {}, {}, {}
            L = None
            for el in sites:
                tot_n = 0; tot_s = tot_ss = None; orgmeans = []
                for org in organisms:
                    csv = track_csv(kdir / org / el, track)
                    if csv is None:
                        continue
                    n, s, ss = col_moments(csv, args.chunksize, args.max_rows_per_file)
                    if n == 0:
                        continue
                    L = len(s) if L is None else L
                    if len(s) != L:        # length guard (tri vs tetra differ across tracks, not within)
                        s, ss = s[:L], ss[:L]
                    tot_s = s if tot_s is None else tot_s + s
                    tot_ss = ss if tot_ss is None else tot_ss + ss
                    tot_n += n
                    orgmeans.append(s / n)
                if tot_n == 0 or tot_s is None:
                    continue
                m = tot_s / tot_n
                v = np.maximum((tot_ss - tot_n * m * m) / max(tot_n - 1, 1), 1e-9)
                elem_n[el] = tot_n; elem_mean[el] = m; elem_var[el] = v
                elem_orgmeans[el] = np.array(orgmeans)
            present = [e for e in sites if e in elem_mean]
            if len(present) < 3 or L is None:
                continue

            means = np.array([elem_mean[e] for e in present])      # (k, L)
            vars = np.array([elem_var[e] for e in present])
            ns = np.array([elem_n[e] for e in present])
            _, pvals, msw, dfw = anova_from_moments(means, vars, ns)
            gate = bh_reject(np.clip(pvals, 1e-300, 1), args.fdr).astype(float)

            offset = TABLE_CENTER - L // 2                          # profile idx -> table pos
            lo = max(0, (TABLE_CENTER - args.search) - offset)
            hi = min(L - 1, (TABLE_CENTER + args.search) - offset)

            # CDS is the flattest site per (kingdom,track); use its amplitude as the
            # relative flatness reference (a global absolute threshold can't separate
            # CDS from low-amplitude real promoters across tracks).
            cds_amp_ref = 0.0
            if "cds" in elem_mean:
                cm = elem_mean["cds"][lo:hi + 1]
                cds_amp_ref = float(cm.max() - cm.min())

            for ti, el in enumerate(present):
                if el == "cds" and not args.include_cds:
                    continue
                m_t = means[ti]
                baseline = np.median(m_t[lo:hi + 1])
                amp = float(m_t[lo:hi + 1].max() - m_t[lo:hi + 1].min())
                a = np.abs(m_t - baseline)
                rest = np.delete(means, ti, axis=0)
                rest_n = np.delete(ns, ti)
                m_rest = (rest_n[:, None] * rest).sum(0) / rest_n.sum()
                d = np.abs(m_t - m_rest)
                # organism consistency: sign agreement of (org_mean - org_baseline) vs pooled
                om = elem_orgmeans[el]                              # (n_org, L)
                ob = np.median(om[:, lo:hi + 1], axis=1, keepdims=True)
                pooled_sign = np.sign(m_t - baseline)
                c = (np.sign(om - ob) == pooled_sign).mean(0)
                score = a * d * c * gate

                win = None
                amp_floor = max(cds_amp_ref * args.cds_factor, args.amp_min)
                if amp >= amp_floor and np.any(score[lo:hi + 1] > 0):
                    win = localize_window(score, lo, hi, args.peak_frac,
                                          args.min_len, args.max_len, args.grid)
                    # Edge guard: a window that runs to the search boundary means the score
                    # peak is not localized near the landmark (diffuse/weak signal) -> drop it.
                    if win is not None and not args.no_edge_guard \
                            and (win[0] <= lo + args.edge_margin or win[1] >= hi - args.edge_margin):
                        win = None
                assigned = win is not None
                if assigned:
                    ws, we = win
                    # paper-style significance on the chosen window (pass 2)
                    sm, sv, sn = [], [], []
                    for e2 in present:
                        nn = ssum = sss = 0
                        for org in organisms:
                            csv = track_csv(kdir / org / e2, track)
                            if csv is None:
                                continue
                            n2, s2, ss2 = seg_moments(csv, ws, we, args.chunksize, args.max_rows_per_file)
                            nn += n2; ssum += s2; sss += ss2
                        mm = ssum / nn
                        sm.append(mm); sv.append(max((sss - nn * mm * mm) / max(nn - 1, 1), 1e-9)); sn.append(nn)
                    n_sig, n_pairs = tukey_fdr(np.array(sm), np.array(sv), np.array(sn), ti, args.fdr)
                    t_start, t_end = ws + offset, we + offset       # -> table coords
                else:
                    ws = we = t_start = t_end = -1
                    n_sig = n_pairs = 0

                track_rows.append({
                    "kingdom": kingdom, "parameter": track, "element": el,
                    "element_name": ELEMENT_NAME.get(el, el),
                    "assigned": bool(assigned),
                    "table_start": int(t_start) if assigned else "",
                    "table_end": int(t_end) if assigned else "",
                    "length": int(we - ws + 1) if assigned else 0,
                    "amplitude": round(amp, 4),
                    "cds_amp_ref": round(cds_amp_ref, 4),
                    "amp_floor": round(amp_floor, 4),
                    "max_score": round(float(score[lo:hi + 1].max()), 5),
                    "mean_consistency": round(float(c[lo:hi + 1].mean()), 3),
                    "n_organisms": len(elem_orgmeans[el]),
                    "tukey_fdr_sig_pairs": n_sig, "tukey_total_pairs": n_pairs,
                })
                print(f"  {kingdom}/{track}/{el}: "
                      f"{'window '+str((t_start,t_end)) if assigned else 'NO WINDOW (flat/amp<min)'} "
                      f"amp={amp:.3f} sigpairs={n_sig}/{n_pairs}", flush=True)

            # durably persist this (kingdom,track) before moving on (resume granularity)
            for r in track_rows:
                writer.writerow(r)
            csv_fh.flush()

    csv_fh.close()

    # ---- outputs: rebuild TABLE_WINDOWS from the complete long CSV ----
    df = pd.read_csv(csv_path)
    table_windows: dict[str, dict[str, list]] = {k: {} for k in KINGDOMS}
    for _, r in df.iterrows():
        k, t, el = r["kingdom"], r["parameter"], r["element"]
        table_windows.setdefault(k, {}).setdefault(t, [None] * len(SITE_ORDER))
        if bool(r["assigned"]) and el in SITE_ORDER:
            table_windows[k][t][SITE_ORDER.index(el)] = (int(r["table_start"]), int(r["table_end"]))

    lines = ["TABLE_SITE_ORDER = " + json.dumps(SITE_ORDER).replace('"', '"'),
             "TABLE_WINDOWS = {"]
    for kingdom in KINGDOMS:
        if not table_windows.get(kingdom):
            continue
        lines.append(f'    "{kingdom}": {{')
        for track in TRACKS:
            wins = table_windows[kingdom].get(track)
            if wins is None:
                continue
            wl = ", ".join("None" if w is None else f"({w[0]}, {w[1]})" for w in wins)
            lines.append(f'        "{track}": [{wl}],')
        lines.append("    },")
    lines.append("}")
    (out / "TABLE_WINDOWS_rederived.py").write_text("\n".join(lines) + "\n")

    print(f"\nDone. {len(df)} (kingdom,param,element) evaluated; "
          f"{int(df['assigned'].sum())} windows assigned, "
          f"{int((~df['assigned']).sum())} flat/no-window.")
    print(f"  long table : {out/'rederived_windows_long.csv'}")
    print(f"  dict       : {out/'TABLE_WINDOWS_rederived.py'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
