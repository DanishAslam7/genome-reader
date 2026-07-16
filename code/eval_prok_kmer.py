#!/usr/bin/env python3
"""
k-mer sequence floor on prokaryote promoters — the composition baseline for the
prok·prom column. Train k-mer+LogReg on the 32-org sequences (same as the diatom
k-mer), predict prok promoter sequences -> prom-recognition rate (fraction called
"prom", class 2). Chance = 0.20.

  python eval_prok_kmer.py --k 4 --out diatom_eval_out
"""
import argparse, glob, json, time
from pathlib import Path
import numpy as np
from eval_diatoms_kmer import load_train, featurise_seqs
from kmer_baseline import CLASS_ORDER, CLASS_ID

PROM_CLASS = 2


def read_prok(seqdir):
    seqs = []
    for f in sorted(glob.glob(str(Path(seqdir) / "*_total_seq"))):
        with open(f, encoding="utf-8", errors="replace") as fh:
            seqs += [ln.strip().upper().encode() for ln in fh if ln.strip()]
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-run-dir",
                    default="taxonomy_multitask_out_run_20260628_143508_loko_baseline_biophys_seed42")
    ap.add_argument("--dataset", default="dataset_reprofiled_32")
    ap.add_argument("--manifest", default="sequence_manifest_reprofiled_32.parquet")
    ap.add_argument("--seq-root", default=None)
    ap.add_argument("--prok-dir", default="prokaryotes/prok_prom_seq")
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
    print(f"[{time.time()-t0:.0f}s] trained", flush=True)

    seqs = read_prok(args.prok_dir)
    Xte = featurise_seqs(seqs, args.k, dim, powers)
    yp = clf.predict(Xte).astype(np.int64)
    prom_rate = float(np.mean(yp == PROM_CLASS))
    dist = {CLASS_ORDER[c]: float(np.mean(yp == c)) for c in range(len(CLASS_ORDER))}
    result = {"n": int(len(yp)), "k": args.k, "prom_rate": prom_rate, "dist": dist,
              "chance": 0.20, "note": "train on 32 orgs, predict prok promoters"}
    out = Path(args.out) / f"prok_kmer_eval.k{args.k}.json"
    json.dump(result, open(out, "w"), indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nPROK k-mer prom_rate={prom_rate:.4f}  n={len(seqs)}  (chance 0.20)")


if __name__ == "__main__":
    main()
