#!/usr/bin/env python3
"""
Zero-shot euk->prok PROMOTER transfer on the 32-org model (prokaryote prong of
external validation; complements diatoms, which lack prom).

Loads the all-32 LOKO-config biophysics model (profiles-only, kingdom-cond OFF ->
single neutral case) and asks: for each prokaryotic promoter window, does the
element head call it "prom"? Metric = prom-recognition rate (fraction predicted
prom, class 2). Chance for prom = 1/5 = 0.20.

CSV loading replicates the VALIDATED old transfer script exactly
(diagnostics/prok_transfer_test.py): pd.read_csv (NO index_col) -> dropna all-NaN
cols -> nan_to_num -> coerce_width to 475 (truncate if long, zero-pad <=2 short at
3' end, matching the euk build's load_profile_csv). Per-org transductive z-norm.

Usage (genome_reader_tf; see qsub_eval_prok.sh):
  python eval_prok.py --model <loko_baseline_biophys ckpt> \
    --prof-dir prokaryotes/prok_prom_profiles --out prok_eval_out
"""
import argparse, glob, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from eval_diatoms import transductive_norm, get_element_output, output_names

PARAMS = ["bbone", "bp", "hbond", "inter", "intra", "sol", "stack"]  # canonical order
N_POS = 475
PROM_CLASS = 2
CLASS_NAMES = ["exon_boundary", "gene_boundary", "prom", "stac", "stoc"]
SUFFIX = "_norm_promoters_total.csv"


def coerce_width(a, target=N_POS):
    w = a.shape[1]
    if w == target:
        return a
    if w > target:
        return a[:, :target]
    if target - w <= 2:          # zero-pad short channels at 3' end (matches training)
        return np.concatenate([a, np.zeros((a.shape[0], target - w), a.dtype)], axis=1)
    raise ValueError(f"channel too short: {w} -> {target}")


def read_prok_param(path):
    df = pd.read_csv(path)
    df = df.dropna(axis=1, how="all")
    a = np.nan_to_num(df.to_numpy(dtype=np.float32), nan=0.0)
    return coerce_width(a)


def discover_orgs(prof_dir):
    orgs = {}
    for f in glob.glob(str(Path(prof_dir) / f"*{SUFFIX}")):
        stem = Path(f).name[: -len(SUFFIX)]
        for p in PARAMS:
            if stem.endswith("_" + p):
                orgs.setdefault(stem[: -(len(p) + 1)], {})[p] = f
                break
    return orgs


def load_org(paramfiles, cache):
    if cache.exists():
        return np.load(cache)
    chans = [read_prok_param(paramfiles[p]) for p in PARAMS]
    n = chans[0].shape[0]
    if not all(c.shape[0] == n for c in chans):
        raise ValueError(f"row mismatch {[c.shape[0] for c in chans]}")
    x = np.stack(chans, axis=1).astype(np.float32)   # [n,7,475]
    np.save(cache, x)
    return x


def dist_of(yp):
    return {CLASS_NAMES[c]: float(np.mean(yp == c)) for c in range(len(CLASS_NAMES))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prof-dir", default="prokaryotes/prok_prom_profiles")
    ap.add_argument("--cache-dir", default="prok_cache")
    ap.add_argument("--out", default="prok_eval_out")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    cd = Path(args.cache_dir); cd.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    import train_taxonomy_multitask  # noqa: F401  (registers custom layers)
    model = tf.keras.models.load_model(args.model, compile=False)
    print("MODEL loaded. inputs:", [(getattr(i, "name", "?"), tuple(i.shape)) for i in model.inputs])
    print("outputs:", output_names(model), flush=True)

    orgs = discover_orgs(args.prof_dir)
    print(f"discovered {len(orgs)} prok organisms", flush=True)
    results, all_pred = {}, []
    for org, pf in sorted(orgs.items()):
        missing = [p for p in PARAMS if p not in pf]
        if missing:
            print(f"  skip {org}: missing {missing}"); continue
        try:
            x = load_org(pf, cd / f"{org}.npy")
        except Exception as e:
            print(f"  skip {org}: {type(e).__name__} {e}"); continue
        xn = transductive_norm(x)
        pred = model.predict(np.transpose(xn, (0, 2, 1)), batch_size=512, verbose=0)
        yp = np.argmax(get_element_output(pred, model), axis=1).astype(np.int64)
        results[org] = {"n": int(len(yp)), "prom_rate": float(np.mean(yp == PROM_CLASS)),
                        "dist": dist_of(yp)}
        all_pred.append(yp)
        print(f"  {org:8s} n={len(yp):6d}  prom_rate={results[org]['prom_rate']:.4f}", flush=True)

    ap_ = np.concatenate(all_pred)
    results["_combined"] = {"n": int(len(ap_)), "prom_rate": float(np.mean(ap_ == PROM_CLASS)),
                            "dist": dist_of(ap_)}
    results["_meta"] = {"model": args.model, "chance_prom": 0.20, "n_orgs_scored": len(all_pred),
                        "note": "zero-shot euk->prok promoter recognition; neutral (no kingdom cond)"}
    out = Path(args.out) / "prok_zeroshot_eval.json"
    json.dump(results, open(out, "w"), indent=2)
    print("\n==== PROK PROMOTER TRANSFER ====")
    print(f"  COMBINED prom_rate = {results['_combined']['prom_rate']:.4f}  (chance 0.20)  n={len(ap_)}")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
