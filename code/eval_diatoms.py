#!/usr/bin/env python3
"""
Zero-shot element recognition on unseen diatoms (external validation).

Loads the all-32 LOKO-config biophysics model (profiles-only, leak-safe:
sequence/KG/table-window/kingdom-conditioning all OFF; complexity computed inside
the model) and evaluates its element head on diatom windows the model has never
seen. Normalization is reproduced per-diatom transductively (each diatom's own
profiles → mean/std; uses profiles only, never labels → still zero-shot).

Model input (this config) = a single tensor [N, 475, 7] (positions x params,
canonical param order). Model output "element" = 5-class softmax
[exon_boundary, gene_boundary, prom, stac, stoc].

Usage (run in genome_reader_tf; see qsub_eval_diatoms.sh):
  python eval_diatoms.py \
    --model taxonomy_multitask_out_run_20260628_143508_loko_baseline_biophys_seed42/checkpoints/best_model.keras \
    --diatoms tp:tp_profiles pt:pt_profiles \
    --out diatom_eval_out
"""
import argparse, json, os
from pathlib import Path
import numpy as np

# canonical param order the model was trained on (6 from tri, inter from tetra)
CANONICAL = ["bbone", "bp", "hbond", "inter", "intra", "sol", "stack"]
TRI_PARAMS = {"bbone", "bp", "hbond", "intra", "sol", "stack"}
N_POS = 475

# element -> (profiler folder, output-dir prefix)
ELEM_LAYOUT = {
    "ee":   ("exon_end",    "total_ee"),
    "es":   ("exon_start",  "total_es"),
    "ge":   ("gene_end",    "total_gene_end"),
    "gs":   ("gene_start",  "total_gene_start"),
    "stac": ("start_codon", "total_start_codon"),
    "stoc": ("stop_codon",  "total_stop_codon"),
    "cds":  ("cds",         "total_cds"),
}
# diatom element -> model element-head class id (5-class order); cds masked (-1)
ELEM_TO_CLASS = {"ee": 0, "es": 0, "ge": 1, "gs": 1, "stac": 3, "stoc": 4, "cds": -1}
CLASS_NAMES = ["exon_boundary", "gene_boundary", "prom", "stac", "stoc"]


def read_param_csv(path, n_pos=N_POS):
    import pandas as pd
    arr = pd.read_csv(path, index_col=0, dtype=np.float32).to_numpy(np.float32, copy=False)
    if arr.shape[1] < n_pos:
        arr = np.concatenate([arr, np.zeros((arr.shape[0], n_pos - arr.shape[1]), np.float32)], axis=1)
    elif arr.shape[1] > n_pos:
        arr = arr[:, :n_pos]
    return arr


def load_diatom(profiles_root, org, elements, cache_dir="diatom_cache"):
    """Return X [N,7,475] and y_class [N] (model class ids; -1 = cds/masked).

    Per-element .npy caching: each element is read from the big CSVs once, then
    saved compactly. Retries skip already-cached elements → the load is resumable
    across the flaky mount (a drop mid-read only costs the current element)."""
    root = Path(profiles_root)
    cd = Path(cache_dir); cd.mkdir(parents=True, exist_ok=True)
    Xs, ys = [], []
    for el in elements:
        cache = cd / f"{org}_{el}.npy"
        if cache.exists():
            x = np.load(cache)
            print(f"  {el:5s} n={x.shape[0]} (cached)", flush=True)
        else:
            folder, prefix = ELEM_LAYOUT[el]
            tri_dir = root / "tri" / folder / f"{prefix}_{org}"
            tetra_dir = root / "tetra" / folder / f"{prefix}_{org}"
            if not tri_dir.exists():
                print(f"  skip {el}: {tri_dir} missing"); continue
            params = {p: read_param_csv(tri_dir / f"{p}.csv") for p in TRI_PARAMS}
            params["inter"] = read_param_csv(tetra_dir / "inter.csv")
            n = params["bbone"].shape[0]
            for p in CANONICAL:
                if params[p].shape[0] != n:
                    raise ValueError(f"{org}/{el}: {p} rows {params[p].shape[0]} != {n}")
            x = np.stack([params[p] for p in CANONICAL], axis=1).astype(np.float32)
            np.save(cache, x)                       # persist for resumable retries
            print(f"  {el:5s} n={n} (built + cached)", flush=True)
        Xs.append(x)
        ys.append(np.full(x.shape[0], ELEM_TO_CLASS[el], dtype=np.int64))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


def transductive_norm(X, eps=1e-6):
    """Per-diatom z-norm over all its windows, per [param,position] — mirrors the
    training organism_profile_normalizer (uses profiles only, no labels)."""
    mean = X.mean(axis=0, keepdims=True)            # [1,7,475]
    std = np.sqrt(np.maximum(X.var(axis=0, keepdims=True), eps))
    return (X - mean) / std


def output_names(model):
    n = getattr(model, "output_names", None)
    if n:
        return list(n)
    return [getattr(o, "name", str(o)).split("/")[0] for o in model.outputs]


def get_element_output(pred, model):
    if isinstance(pred, dict):
        # exact "element" head (not element_coarse / aux_element_*)
        if "element" in pred:
            return pred["element"]
        for k in pred:
            if k == "element" or k.endswith("/element"):
                return pred[k]
        raise KeyError(f"no 'element' output; have {list(pred)}")
    names = output_names(model)
    idx = names.index("element") if "element" in names else \
        next(i for i, nm in enumerate(names) if nm == "element" or nm.endswith("element"))
    return pred[idx]


def score(y_true, y_pred):
    valid = y_true >= 0                             # drop cds (masked)
    yt, yp = y_true[valid], y_pred[valid]
    acc = float(np.mean(yp == yt)) if len(yt) else None
    per_class = {}
    for c in sorted(set(yt.tolist())):
        m = yt == c
        per_class[CLASS_NAMES[c]] = {"support": int(m.sum()),
                                     "recall": float(np.mean(yp[m] == c))}
    # what the model predicts on diatom windows (incl. prom it can never be right on)
    pred_dist = {CLASS_NAMES[c]: int((yp == c).sum()) for c in range(len(CLASS_NAMES))}
    return {"n": int(len(yt)), "accuracy": acc, "per_class": per_class,
            "pred_distribution": pred_dist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--diatoms", nargs="+", required=True,
                    help="org:profiles_root, e.g. tp:tp_profiles pt:pt_profiles "
                         "(org must match the <org> token in total_*_<org> dirs)")
    ap.add_argument("--elements", nargs="+",
                    default=["ee", "es", "ge", "gs", "stac", "stoc", "cds"])
    ap.add_argument("--out", default="diatom_eval_out")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    import train_taxonomy_multitask  # noqa: F401  (registers custom layers/metrics)
    model = tf.keras.models.load_model(args.model, compile=False)
    print("MODEL loaded.")
    print("  inputs :", [(getattr(i, "name", "?"), tuple(i.shape)) for i in model.inputs])
    print("  outputs:", output_names(model))

    results, all_true, all_pred = {}, [], []
    for spec in args.diatoms:
        org, proot = spec.split(":", 1)
        print(f"\n=== {org}  ({proot}) ===")
        X, y = load_diatom(proot, org, args.elements)
        Xn = transductive_norm(X)
        x_in = np.transpose(Xn, (0, 2, 1))          # [N,475,7]
        pred = model.predict(x_in, batch_size=512, verbose=0)
        elem = get_element_output(pred, model)
        yp = np.argmax(elem, axis=1).astype(np.int64)
        results[org] = score(y, yp)
        print(f"  -> accuracy={results[org]['accuracy']:.4f} on n={results[org]['n']}")
        valid = y >= 0
        all_true.append(y[valid]); all_pred.append(yp[valid])

    at, ap_ = np.concatenate(all_true), np.concatenate(all_pred)
    combined = {"n": int(len(at)), "accuracy": float(np.mean(ap_ == at))}
    results["_combined"] = combined
    results["_meta"] = {"model": args.model, "chance_4class": 0.25, "chance_5class": 0.20,
                        "note": "zero-shot; diatoms have no prom so 4 classes present"}
    out = Path(args.out) / "diatom_zeroshot_eval.json"
    json.dump(results, open(out, "w"), indent=2)
    print("\n==== SUMMARY ====")
    for org in [s.split(':', 1)[0] for s in args.diatoms]:
        print(f"  {org}: acc={results[org]['accuracy']:.4f} (n={results[org]['n']})")
    print(f"  COMBINED: acc={combined['accuracy']:.4f} (n={combined['n']})  chance≈0.25")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
