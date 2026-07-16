#!/usr/bin/env python3
"""
k-mer + linear classifier baseline for element recognition.

Scored on the EXACT test rows a given multitask run used (its
provenance_test_idx.npy), trained on that run's provenance_train_idx.npy.
This makes it a fair sequence-only benchmark: does a classical k-mer model
match the biophysics model on the *identical* split?

Works for both in-distribution and LOKO -- the provenance indices already
encode whichever split the run used (LOKO holdout is baked in).

Element space matches the multitask element head exactly: 5 classes
{exon_boundary(es+ee), gene_boundary(gs+ge), prom, stac, stoc}; cds masked out.

Usage:
  python kmer_baseline.py \
      --run-dir <multitask run dir with provenance_*_idx.npy> \
      --dataset dataset_reprofiled_32 \
      --manifest sequence_manifest_reprofiled_32.parquet \
      --k 4 --classifier logreg --max-train-per-class 50000 \
      --out kmer_baseline_out
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

# --- 5-class element space (must match the multitask element head) ------------
BOUNDARY_COLLAPSE = {"es": "exon_boundary", "ee": "exon_boundary",
                     "gs": "gene_boundary", "ge": "gene_boundary"}
KEEP_AS_IS = {"prom", "stac", "stoc"}
CLASS_ORDER = ["exon_boundary", "gene_boundary", "prom", "stac", "stoc"]
CLASS_ID = {c: i for i, c in enumerate(CLASS_ORDER)}

def collapse_element(name):
    if name in BOUNDARY_COLLAPSE:
        return BOUNDARY_COLLAPSE[name]
    if name in KEEP_AS_IS:
        return name
    return None  # cds and everything else -> masked / dropped

# --- base encoding for k-mers -------------------------------------------------
BASE = np.full(256, -1, dtype=np.int64)
for i, ch in enumerate(b"ACGT"):
    BASE[ch] = i
for i, ch in enumerate(b"acgt"):
    BASE[ch] = i

def kmer_vector(seq_bytes, k, dim, powers):
    """L1-normalised k-mer frequency vector; windows containing N are skipped."""
    v = np.zeros(dim, dtype=np.float32)
    codes = BASE[np.frombuffer(seq_bytes, dtype=np.uint8)]
    n = len(codes) - k + 1
    if n <= 0:
        return v
    ids = np.zeros(n, dtype=np.int64)
    ok = np.ones(n, dtype=bool)
    for j in range(k):
        cj = codes[j:j + n]
        ids += cj * powers[j]
        ok &= (cj >= 0)
    ids = ids[ok]
    if len(ids):
        bc = np.bincount(ids, minlength=dim)
        v[:len(bc)] = bc
        s = v.sum()
        if s > 0:
            v /= s
    return v

# --- sequence path resolution (scratch abs path -> local mirror) --------------
def resolve_local(seq_path, seq_root):
    marker = "genome_reader/"
    rel = seq_path.split(marker, 1)[1] if marker in seq_path else None
    cands = []
    if seq_root and rel:
        cands.append(os.path.join(seq_root, rel))
    if rel:
        cands.append(rel)
    cands.append(seq_path)
    if rel:
        cands.append(os.path.join("ncbi_inspect/extracted_new",
                                  os.path.basename(os.path.dirname(rel)),
                                  os.path.basename(rel)))
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="multitask run dir with provenance_{train,test}_idx.npy")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--seq-root", default=None,
                    help="local root for extracted FASTAs (else auto-resolve)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--classifier", choices=["logreg", "sgd"], default="logreg")
    ap.add_argument("--max-train-per-class", type=int, default=50000,
                    help="subsample train for speed; k-mer LR saturates fast. "
                         "Set 0 to use the full matched provenance train set.")
    ap.add_argument("--out", type=Path, default=Path("kmer_baseline_out"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # --- load matched split + labels -----------------------------------------
    tr_idx = np.load(args.run_dir / "provenance_train_idx.npy")
    te_idx = np.load(args.run_dir / "provenance_test_idx.npy")
    Ye = np.load(args.dataset / "Y_element.npy", mmap_mode="r")
    enc = json.load(open(args.dataset / "encoders.json"))
    inv_el = {v: k for k, v in enc["element"].items()}

    def to_class(idx):
        raw = np.asarray(Ye[np.asarray(idx)])
        names = [inv_el[int(r)] for r in raw]
        cls = np.array([CLASS_ID.get(collapse_element(n), -1) for n in names],
                       dtype=np.int64)
        return cls

    y_tr_full = to_class(tr_idx)
    y_te = to_class(te_idx)
    tr_keep = y_tr_full >= 0
    te_keep = y_te >= 0
    tr_idx, y_tr = tr_idx[tr_keep], y_tr_full[tr_keep]
    te_idx, y_te = te_idx[te_keep], y_te[te_keep]

    # optional per-class train cap
    if args.max_train_per_class and args.max_train_per_class > 0:
        rng = np.random.default_rng(0)
        keep = []
        for c in range(len(CLASS_ORDER)):
            ci = np.where(y_tr == c)[0]
            if len(ci) > args.max_train_per_class:
                ci = rng.choice(ci, args.max_train_per_class, replace=False)
            keep.append(ci)
        keep = np.sort(np.concatenate(keep))
        tr_idx, y_tr = tr_idx[keep], y_tr[keep]
    print(f"[{time.time()-t0:.0f}s] train={len(tr_idx):,} test={len(te_idx):,} "
          f"classes={CLASS_ORDER}", flush=True)

    # --- build byte-offset table for the needed global_rows ------------------
    import pyarrow.parquet as pq
    need = np.unique(np.concatenate([tr_idx, te_idx]))
    need_set = set(int(x) for x in need)
    off = {}  # global_row -> (seq_path, start, end)
    pf = pq.ParquetFile(args.manifest)
    cols = ["global_row", "seq_path", "seq_byte_start", "seq_byte_end"]
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=cols)
        gr = t.column("global_row").to_numpy()
        m = np.isin(gr, need)
        if not m.any():
            continue
        sp = t.column("seq_path").to_pylist()
        bs = t.column("seq_byte_start").to_numpy()
        be = t.column("seq_byte_end").to_numpy()
        for i in np.where(m)[0]:
            off[int(gr[i])] = (sp[i], int(bs[i]), int(be[i]))
        if len(off) >= len(need_set):
            break
    print(f"[{time.time()-t0:.0f}s] resolved byte offsets for {len(off):,} rows", flush=True)

    # --- featurise: group reads by file -------------------------------------
    dim = 4 ** args.k
    powers = (4 ** np.arange(args.k - 1, -1, -1)).astype(np.int64)
    path_cache = {}

    def featurise(idx_arr):
        X = np.zeros((len(idx_arr), dim), dtype=np.float32)
        # group by file for sequential reads
        by_file = {}
        for row, gr in enumerate(idx_arr):
            gr = int(gr)
            sp, s, e = off[gr]
            by_file.setdefault(sp, []).append((row, s, e))
        miss = 0
        for sp, items in by_file.items():
            local = path_cache.get(sp)
            if local is None:
                local = resolve_local(sp, args.seq_root)
                path_cache[sp] = local
            if local is None:
                miss += len(items)
                continue
            with open(local, "rb") as fh:
                for row, s, e in items:
                    fh.seek(s)
                    raw = fh.read(e - s).strip()
                    X[row] = kmer_vector(raw, args.k, dim, powers)
        if miss:
            print(f"  WARNING: {miss} rows had unresolved seq_path", flush=True)
        return X

    Xtr = featurise(tr_idx)
    print(f"[{time.time()-t0:.0f}s] featurised train", flush=True)
    Xte = featurise(te_idx)
    print(f"[{time.time()-t0:.0f}s] featurised test", flush=True)

    # --- train + predict -----------------------------------------------------
    if args.classifier == "logreg":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=200, C=1.0, n_jobs=-1,
                                 multi_class="multinomial", solver="saga")
    else:
        from sklearn.linear_model import SGDClassifier
        clf = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=30, n_jobs=-1)
    clf.fit(Xtr, y_tr)
    pred = clf.predict(Xte)
    print(f"[{time.time()-t0:.0f}s] trained {args.classifier}", flush=True)

    acc = float(np.mean(pred == y_te))
    per_class = {}
    for c, name in enumerate(CLASS_ORDER):
        m = y_te == c
        per_class[name] = {"support": int(m.sum()),
                           "accuracy": float(np.mean(pred[m] == c)) if m.any() else None}

    # --- target from the multitask run for direct comparison -----------------
    target = None
    pk = args.run_dir / "per_kingdom_element_accuracy.json"
    if pk.exists():
        d = json.load(open(pk))
        kd = d.get("per_kingdom", {})
        if len(kd) == 1:
            target = list(kd.values())[0].get("element_masked_accuracy")
    if target is None:  # in-dist (multi-kingdom): fall back to overall test metric
        mj = args.run_dir / "metrics.json"
        if mj.exists():
            m = json.load(open(mj))
            target = m.get("test_results", {}).get("element_masked_accuracy")

    result = {
        "run_dir": str(args.run_dir),
        "k": args.k, "classifier": args.classifier,
        "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
        "kmer_accuracy": acc,
        "per_class": per_class,
        "multitask_target": target,
        "delta_vs_multitask": (acc - target) if target is not None else None,
    }
    out = args.out / (args.run_dir.name + f".k{args.k}.{args.classifier}.json")
    json.dump(result, open(out, "w"), indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nWROTE {out}  ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
