#!/usr/bin/env python3
"""
Figure 4, panel b — evaluation-only normalization ablation for the WITH-SEQUENCE arm.

Companion to eval_norm_ablation.py (profile-only arm). Same logic: held-out organisms
contribute no training rows, so their normalization statistics affect only test-time
input; we hold the trained model FIXED and score its test set twice.

  identity    : held-out orgs fall back to mean 0 / std 1  -> raw profile scale
  own-profile : held-out orgs standardized from their own profiles (run's npz)

WHY THIS EXISTS. Panel a is eval-only. If panel b came from separately-trained runs,
the headline contrast (+0.44 vs +0.035) would mix two methodologies — and we now know
identity-side numbers differ by ~6 points between them, because in the collapsed regime
the output is arbitrary and two equivalent checkpoints land far apart. Both panels must
be measured the same way.

The extra work versus panel a is the sequence branch: LOKO test rows need their DNA,
which lives behind byte offsets in the sequence manifest (same path kmer_baseline uses).

  python eval_norm_ablation_withseq.py --run-dir <loko withseq run>
  -> norm_ablation_out/norm_ablation_withseq__<rundir>.json
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
from eval_diatoms import get_element_output, output_names
from kmer_baseline import collapse_element, CLASS_ORDER, CLASS_ID, resolve_local

SEQ_LEN = 501


def load_offsets(manifest, idx, t0):
    """global_row -> (seq_path, byte_start, byte_end) for the rows we need."""
    import pyarrow.parquet as pq
    need = set(int(x) for x in idx)
    off = {}
    pf = pq.ParquetFile(manifest)
    cols = ["global_row", "seq_path", "seq_byte_start", "seq_byte_end"]
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=cols)
        gr = t.column("global_row").to_numpy()
        m = np.isin(gr, idx)
        if not m.any():
            continue
        sp = t.column("seq_path").to_pylist()
        bs = t.column("seq_byte_start").to_numpy()
        be = t.column("seq_byte_end").to_numpy()
        for i in np.where(m)[0]:
            off[int(gr[i])] = (sp[i], int(bs[i]), int(be[i]))
        if len(off) >= len(need):
            break
    print(f"[{time.time()-t0:.0f}s] resolved {len(off):,}/{len(need):,} offsets", flush=True)
    return off


def read_seqs(rows, off, seq_root):
    """Sequences for `rows`, in order. Grouped by file so each is opened once."""
    out = [None] * len(rows)
    by_file, pcache = {}, {}
    for i, gr in enumerate(rows):
        if int(gr) not in off:
            continue
        sp, s, e = off[int(gr)]
        by_file.setdefault(sp, []).append((i, s, e))
    for sp, items in by_file.items():
        local = pcache.get(sp) or resolve_local(sp, seq_root)
        pcache[sp] = local
        if local is None:
            continue
        with open(local, "rb") as fh:
            for i, s, e in items:
                fh.seek(s)
                out[i] = fh.read(e - s).strip().decode("ascii", "ignore").upper()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--dataset", default="dataset_reprofiled_32")
    ap.add_argument("--manifest", default="sequence_manifest_reprofiled_32.parquet")
    ap.add_argument("--seq-root", default=None)
    ap.add_argument("--max-eval", type=int, default=0, help="0 = full test set")
    ap.add_argument("--chunk", type=int, default=8000)
    ap.add_argument("--out", default="norm_ablation_out")
    args = ap.parse_args()
    run = Path(args.run_dir)
    model_path = args.model or str(run / "checkpoints" / "best_model.keras")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    te = np.load(run / "provenance_test_idx.npy")
    Ye = np.load(Path(args.dataset) / "Y_element.npy", mmap_mode="r")
    Yo = np.load(Path(args.dataset) / "Y_organism.npy", mmap_mode="r")
    enc = json.load(open(Path(args.dataset) / "encoders.json"))
    inv = {v: k for k, v in enc["element"].items()}
    y = np.array([CLASS_ID.get(collapse_element(inv[int(r)]), -1) for r in np.asarray(Ye[te])],
                 dtype=np.int64)
    keep = y >= 0
    te, y = te[keep], y[keep]
    if args.max_eval and len(te) > args.max_eval:
        rng = np.random.default_rng(0)
        s = np.sort(rng.choice(len(te), args.max_eval, replace=False))
        te, y = te[s], y[s]
    order = np.argsort(te)
    te, y = te[order], y[order]
    org = np.asarray(Yo[te]).astype(np.int64)

    nz = np.load(run / "organism_profile_normalization.npz")
    means, stds = nz["means"].astype(np.float32), nz["stds"].astype(np.float32)
    counts, names = nz["counts"], [str(n) for n in nz["organism_names"]]
    test_orgs = sorted(set(org.tolist()))
    zero = [names[o] for o in test_orgs if counts[o] == 0]
    print(f"[{time.time()-t0:.0f}s] test={len(te):,} rows over {len(test_orgs)} organisms")
    print(f"  {[names[o] for o in test_orgs]}")
    if zero:
        raise SystemExit(f"ABORT: {zero} lack stats -> identity-fallback run, not proper-norm.")

    off = load_offsets(args.manifest, te, t0)

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    import train_taxonomy_multitask as ttm
    model = tf.keras.models.load_model(model_path, compile=False)
    innames = [getattr(i, "name", "?") for i in model.inputs]
    print(f"[{time.time()-t0:.0f}s] loaded {model_path}")
    print(f"  inputs={[(getattr(i,'name','?'), tuple(i.shape)) for i in model.inputs]}")
    print(f"  outputs={output_names(model)}", flush=True)
    if not any("seq" in n.lower() for n in innames):
        raise SystemExit("ABORT: model has no sequence input -> use eval_norm_ablation.py")

    X = np.load(Path(args.dataset) / "X_raw.npy", mmap_mode="r")
    hits = {"identity": np.zeros(len(CLASS_ORDER), np.int64),
            "own_profile": np.zeros(len(CLASS_ORDER), np.int64)}
    tot = np.zeros(len(CLASS_ORDER), np.int64)
    n_missing = 0

    for start in range(0, len(te), args.chunk):
        sl = slice(start, min(start + args.chunk, len(te)))
        rows, yc, oc = te[sl], y[sl], org[sl]
        seqs = read_seqs(rows, off, args.seq_root)
        ok = np.array([s is not None and len(s) >= SEQ_LEN for s in seqs])
        n_missing += int((~ok).sum())
        if not ok.any():
            continue
        rows, yc, oc = rows[ok], yc[ok], oc[ok]
        seqs = [s for s, k in zip(seqs, ok) if k]
        xr = np.asarray(X[rows], dtype=np.float32)                 # [n,7,475] raw
        xseq = np.zeros((len(seqs), SEQ_LEN, 4), dtype=np.float32)
        for i, s in enumerate(seqs):
            b = np.frombuffer(s[:SEQ_LEN].encode("ascii", "ignore"), dtype=np.uint8)
            xseq[i, :len(b), :] = ttm.DNA_ONEHOT_TABLE[b]

        for tag, xin in (("identity", xr), ("own_profile", (xr - means[oc]) / stds[oc])):
            xprof = np.transpose(xin, (0, 2, 1))                   # [n,475,7]
            x = {nm.split(":")[0]: (xseq if "seq" in nm.lower() else xprof) for nm in innames}
            p = model.predict(x, batch_size=256, verbose=0)
            yp = np.argmax(get_element_output(p, model), axis=1)
            for c in range(len(CLASS_ORDER)):
                hits[tag][c] += int(np.sum((yc == c) & (yp == c)))
        for c in range(len(CLASS_ORDER)):
            tot[c] += int(np.sum(yc == c))
        print(f"  [{time.time()-t0:.0f}s] {sl.stop}/{len(te)}", flush=True)

    res = {"run_dir": run.name, "model": model_path, "arm": "withseq",
           "n": int(tot.sum()), "n_missing_seq": n_missing,
           "test_organisms": [names[o] for o in test_orgs], "classes": list(CLASS_ORDER)}
    for tag in ("identity", "own_profile"):
        res[tag] = {"overall": float(hits[tag].sum() / tot.sum()),
                    "per_class": {CLASS_ORDER[c]: float(hits[tag][c] / tot[c])
                                  for c in range(len(CLASS_ORDER)) if tot[c]}}
    res["delta"] = res["own_profile"]["overall"] - res["identity"]["overall"]

    out = Path(args.out) / f"norm_ablation_withseq__{run.name}.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"\n  identity     {res['identity']['overall']:.4f}")
    print(f"  own-profile  {res['own_profile']['overall']:.4f}")
    print(f"  delta        {res['delta']:+.4f}   (missing seq rows: {n_missing})")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
