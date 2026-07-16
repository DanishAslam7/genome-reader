#!/usr/bin/env python3
"""
k-mer sequence floor for the diatom zero-shot test — the sequence-composition
comparator to eval_diatoms.py (biophysics).

Parallel to the biophysics eval: TRAIN a k-mer + LogReg classifier on the 32-org
TRAINING sequences (an all-32 run's provenance_train_idx), then PREDICT the unseen
diatom windows (read directly from pt/tp total_seq files). Same 5-class space, same
"train on 32 / zero-shot on diatoms" setup. Chance ≈ 0.25 (4 classes; no prom).

Usage (genome_reader_tf or any env with sklearn/pyarrow; see qsub_eval_diatoms_kmer.sh):
  python eval_diatoms_kmer.py \
    --train-run-dir taxonomy_multitask_out_run_20260628_143508_loko_baseline_biophys_seed42 \
    --dataset dataset_reprofiled_32 --manifest sequence_manifest_reprofiled_32.parquet \
    --diatoms tp:tp pt:pt --k 4 --max-train-per-class 50000 --out diatom_eval_out
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
from kmer_baseline import (kmer_vector, resolve_local, collapse_element,
                           CLASS_ORDER, CLASS_ID)

# diatom seq files: {org}/{org}_total_seq_{el}
DIATOM_ELEMENTS = ["ee", "es", "ge", "gs", "stac", "stoc"]  # cds dropped (masked); no prom


def featurise_seqs(seq_iter, k, dim, powers):
    rows = [kmer_vector(s, k, dim, powers) for s in seq_iter]
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, dim), np.float32)


def load_train(run_dir, dataset, manifest, k, dim, powers, cap, seq_root, t0):
    import pyarrow.parquet as pq
    tr_idx = np.load(Path(run_dir) / "provenance_train_idx.npy")
    Ye = np.load(Path(dataset) / "Y_element.npy", mmap_mode="r")
    enc = json.load(open(Path(dataset) / "encoders.json"))
    inv = {v: k_ for k_, v in enc["element"].items()}
    y = np.array([CLASS_ID.get(collapse_element(inv[int(r)]), -1) for r in np.asarray(Ye[tr_idx])],
                 dtype=np.int64)
    keep = y >= 0
    tr_idx, y = tr_idx[keep], y[keep]
    if cap and cap > 0:
        rng = np.random.default_rng(0); sel = []
        for c in range(len(CLASS_ORDER)):
            ci = np.where(y == c)[0]
            if len(ci) > cap:
                ci = rng.choice(ci, cap, replace=False)
            sel.append(ci)
        sel = np.sort(np.concatenate(sel)); tr_idx, y = tr_idx[sel], y[sel]
    print(f"[{time.time()-t0:.0f}s] train rows={len(tr_idx):,}", flush=True)
    # byte offsets for the needed rows
    need = set(int(x) for x in tr_idx)
    off = {}
    pf = pq.ParquetFile(manifest)
    cols = ["global_row", "seq_path", "seq_byte_start", "seq_byte_end"]
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=cols)
        gr = t.column("global_row").to_numpy()
        m = np.isin(gr, tr_idx)
        if not m.any():
            continue
        sp = t.column("seq_path").to_pylist()
        bs = t.column("seq_byte_start").to_numpy(); be = t.column("seq_byte_end").to_numpy()
        for i in np.where(m)[0]:
            off[int(gr[i])] = (sp[i], int(bs[i]), int(be[i]))
        if len(off) >= len(need):
            break
    print(f"[{time.time()-t0:.0f}s] resolved {len(off):,} train offsets", flush=True)
    # read sequences grouped by file
    X = np.zeros((len(tr_idx), dim), np.float32)
    by_file, pcache = {}, {}
    for row, gr in enumerate(tr_idx):
        sp, s, e = off[int(gr)]; by_file.setdefault(sp, []).append((row, s, e))
    for sp, items in by_file.items():
        local = pcache.get(sp) or resolve_local(sp, seq_root); pcache[sp] = local
        if local is None:
            continue
        with open(local, "rb") as fh:
            for row, s, e in items:
                fh.seek(s); X[row] = kmer_vector(fh.read(e - s).strip(), k, dim, powers)
    print(f"[{time.time()-t0:.0f}s] featurised train", flush=True)
    return X, y


def load_diatom(org_dir, org, k, dim, powers):
    Xs, ys = [], []
    for el in DIATOM_ELEMENTS:
        cls = CLASS_ID.get(collapse_element(el))
        if cls is None:
            continue
        path = Path(org_dir) / f"{org}_total_seq_{el}"
        if not path.exists():
            print(f"  skip {el}: {path} missing"); continue
        with open(path, "rb") as fh:
            seqs = [ln.strip() for ln in fh if ln.strip()]
        Xs.append(featurise_seqs(seqs, k, dim, powers))
        ys.append(np.full(len(seqs), cls, dtype=np.int64))
        print(f"  {el:5s} n={len(seqs)}")
    return np.concatenate(Xs, 0), np.concatenate(ys, 0)


def score(yt, yp):
    per_class = {}
    for c in sorted(set(yt.tolist())):
        m = yt == c
        per_class[CLASS_ORDER[c]] = {"support": int(m.sum()), "recall": float(np.mean(yp[m] == c))}
    return {"n": int(len(yt)), "accuracy": float(np.mean(yp == yt)), "per_class": per_class}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-run-dir", required=True)
    ap.add_argument("--dataset", default="dataset_reprofiled_32")
    ap.add_argument("--manifest", default="sequence_manifest_reprofiled_32.parquet")
    ap.add_argument("--seq-root", default=None)
    ap.add_argument("--diatoms", nargs="+", required=True, help="org:dir e.g. tp:tp pt:pt")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-train-per-class", type=int, default=50000)
    ap.add_argument("--out", default="diatom_eval_out")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dim = 4 ** args.k
    powers = (4 ** np.arange(args.k - 1, -1, -1)).astype(np.int64)

    Xtr, ytr = load_train(args.train_run_dir, args.dataset, args.manifest, args.k, dim,
                          powers, args.max_train_per_class, args.seq_root, t0)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=200, n_jobs=-1, multi_class="multinomial", solver="saga")
    clf.fit(Xtr, ytr)
    print(f"[{time.time()-t0:.0f}s] trained LogReg", flush=True)

    results, all_t, all_p = {}, [], []
    for spec in args.diatoms:
        org, d = spec.split(":", 1)
        print(f"\n=== {org} ({d}) ===")
        Xte, yte = load_diatom(d, org, args.k, dim, powers)
        yp = clf.predict(Xte).astype(np.int64)
        results[org] = score(yte, yp)
        print(f"  -> kmer accuracy={results[org]['accuracy']:.4f} on n={results[org]['n']}")
        all_t.append(yte); all_p.append(yp)
    at, apr = np.concatenate(all_t), np.concatenate(all_p)
    results["_combined"] = {"n": int(len(at)), "accuracy": float(np.mean(apr == at))}
    results["_meta"] = {"train_run_dir": args.train_run_dir, "k": args.k,
                        "chance_4class": 0.25, "note": "train on 32 orgs, zero-shot on diatoms"}
    out = Path(args.out) / f"diatom_kmer_eval.k{args.k}.json"
    json.dump(results, open(out, "w"), indent=2)
    print("\n==== SUMMARY (k-mer floor) ====")
    for spec in args.diatoms:
        org = spec.split(":", 1)[0]
        print(f"  {org}: acc={results[org]['accuracy']:.4f} (n={results[org]['n']})")
    print(f"  COMBINED: acc={results['_combined']['accuracy']:.4f}  chance≈0.25")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
