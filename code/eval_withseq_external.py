#!/usr/bin/env python3
"""
External eval for the profiles+sequence model (loko_baseline_withseq) on the unseen
diatoms + prokaryote promoters — completes the table row "Sequence CNN" (really
biophysics + sequence branch). Dual input: profile [N,475,7] + sequence one-hot
[N,501,4]. Sequences are filtered to exactly 501 bp so they align 1:1 with the
profiler's kept windows.

  python eval_withseq_external.py --model <loko_baseline_withseq ckpt> \
    --run-dir <loko_baseline_withseq run> --out withseq_external_out
"""
import argparse, glob, json, os, time
from pathlib import Path
import numpy as np
from eval_diatoms import transductive_norm, get_element_output, output_names, CANONICAL, ELEM_LAYOUT, read_param_csv
from eval_prok import coerce_width, read_prok_param, discover_orgs
from kmer_baseline import CLASS_ORDER, CLASS_ID, collapse_element

SEQ_LEN = 501
PROM_CLASS = 2
DIATOM_ELEMS = ["ee", "es", "ge", "gs", "stac", "stoc", "cds"]   # cds -> masked (-1)
ELEM_TO_CLASS = {"ee": 0, "es": 0, "ge": 1, "gs": 1, "stac": 3, "stoc": 4, "cds": -1}


def onehot(seqs, table):
    out = np.zeros((len(seqs), SEQ_LEN, 4), dtype=np.float32)
    for i, s in enumerate(seqs):
        b = np.frombuffer(s[:SEQ_LEN].encode("ascii", "ignore"), dtype=np.uint8)
        out[i, :len(b), :] = table[b]
    return out


def valid_seqs(path):
    """Replicate the profiler's read_sequences filter EXACTLY (strip+upper, len==501,
    all bases in ATGC) so the kept set matches the profile CSVs 1:1 and in order."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            s = ln.strip().upper()
            if len(s) == SEQ_LEN and not any(b not in "ATGC" for b in s):
                out.append(s)
    return out


def load_diatom(prof_root, org_prof, seq_dir, org_seq, cache):
    Xp, seqs, ys = [], [], []
    for el in DIATOM_ELEMS:
        folder, prefix = ELEM_LAYOUT[el]
        tri = Path(prof_root) / "tri" / folder / f"{prefix}_{org_prof}"
        tet = Path(prof_root) / "tetra" / folder / f"{prefix}_{org_prof}"
        sp = Path(seq_dir) / f"{org_seq}_total_seq_{el}"
        if not tri.exists() or not sp.exists():
            print(f"  skip {el}"); continue
        params = {p: read_param_csv(tri / f"{p}.csv") for p in CANONICAL if p != "inter"}
        params["inter"] = read_param_csv(tet / "inter.csv")
        x = np.stack([params[p] for p in CANONICAL], axis=1).astype(np.float32)  # [n,7,475]
        s = valid_seqs(sp)
        m = min(x.shape[0], len(s))
        if x.shape[0] != len(s):
            print(f"  align {el}: prof {x.shape[0]} seq {len(s)} -> {m}")
        Xp.append(x[:m]); seqs.extend(s[:m]); ys.extend([ELEM_TO_CLASS[el]] * m)
        print(f"  {el:5s} n={m}")
    return np.concatenate(Xp, 0), seqs, np.array(ys, np.int64)


def load_prok(prof_dir, seq_dir):
    orgs = discover_orgs(prof_dir)
    Xp, seqs = [], []
    for org, pf in sorted(orgs.items()):
        if any(p not in pf for p in CANONICAL):
            continue
        sp = Path(seq_dir) / f"{org}_total_seq"
        if not sp.exists():
            continue
        try:
            chans = [read_prok_param(pf[p]) for p in CANONICAL]
        except Exception as e:
            print(f"  skip {org}: {e}"); continue
        n = chans[0].shape[0]
        if not all(c.shape[0] == n for c in chans):
            continue
        x = np.stack(chans, axis=1).astype(np.float32)
        s = valid_seqs(sp)
        m = min(n, len(s))
        Xp.append(x[:m]); seqs.extend(s[:m])
    return np.concatenate(Xp, 0), seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="withseq_external_out")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    import train_taxonomy_multitask as ttm
    model = tf.keras.models.load_model(args.model, compile=False)
    innames = [getattr(i, "name", "?") for i in model.inputs]
    print("inputs:", [(getattr(i, "name", "?"), tuple(i.shape)) for i in model.inputs])
    print("outputs:", output_names(model), flush=True)

    def predict(Xp, seqs):
        xn = transductive_norm(Xp)
        xprof = np.transpose(xn, (0, 2, 1))                 # [N,475,7]
        xseq = onehot(seqs, ttm.DNA_ONEHOT_TABLE)           # [N,501,4]
        x = {}
        for nm in innames:
            key = nm.split(":")[0]
            x[key] = xseq if "seq" in key.lower() else xprof
        p = model.predict(x, batch_size=256, verbose=0)
        return np.argmax(get_element_output(p, model), axis=1).astype(np.int64)

    results = {}
    # ---- diatoms ----
    Xp, seqs, y = load_diatom("tp_profiles", "tp_total", "tp", "tp", None)
    Xp2, seqs2, y2 = load_diatom("pt_profiles", "pt_total", "pt", "pt", None)
    Xp = np.concatenate([Xp, Xp2], 0); seqs = seqs + seqs2; y = np.concatenate([y, y2])
    yp = predict(Xp, seqs)
    valid = y >= 0
    results["diatoms"] = {"n": int(valid.sum()),
                          "accuracy": float(np.mean(yp[valid] == y[valid]))}
    print(f"[{time.time()-t0:.0f}s] diatoms acc={results['diatoms']['accuracy']:.4f} n={int(valid.sum())}", flush=True)
    # ---- prok ----
    Xp, seqs = load_prok("prokaryotes/prok_prom_profiles", "prokaryotes/prok_prom_seq")
    yp = predict(Xp, seqs)
    results["prokaryotes"] = {"n": int(len(yp)), "prom_rate": float(np.mean(yp == PROM_CLASS))}
    print(f"[{time.time()-t0:.0f}s] prok prom_rate={results['prokaryotes']['prom_rate']:.4f} n={len(yp)}", flush=True)

    results["_meta"] = {"model": args.model, "note": "profiles+sequence (withseq) on external sets"}
    json.dump(results, open(Path(args.out) / "withseq_external.json", "w"), indent=2)
    print(f"\nWITHSEQ EXTERNAL: diatoms={results['diatoms']['accuracy']:.4f} "
          f"prok_prom={results['prokaryotes']['prom_rate']:.4f}", flush=True)


if __name__ == "__main__":
    main()
