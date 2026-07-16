#!/usr/bin/env python3
"""
Per-parameter mechanistic ablation — which of the 7 biophysical parameters carry
which element signal. Inference-time channel ablation on the trained all-32
biophysics model (no retraining):
  - DROP-p : zero parameter p's channel -> accuracy loss = p's marginal importance
  - KEEP-p : zero all channels except p  -> p's standalone informativeness
measured overall AND per element class. Ablation is done in normalized space, where
zero == the per-position mean, so it removes p's information without distorting scale.

  python eval_param_ablation.py --model <loko_baseline_biophys ckpt> \
    --run-dir <loko_baseline_biophys run> --out param_ablation_out
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
from eval_diatoms import get_element_output, output_names, CANONICAL
from kmer_baseline import collapse_element, CLASS_ORDER, CLASS_ID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-dir", required=True,
                    help="supplies provenance_test_idx.npy + organism_profile_normalization.npz")
    ap.add_argument("--dataset", default="dataset_reprofiled_32")
    ap.add_argument("--max-eval", type=int, default=40000)
    ap.add_argument("--out", default="param_ablation_out")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # --- test rows + 5-class labels (drop cds / non-universal) -----------------
    te = np.load(Path(args.run_dir) / "provenance_test_idx.npy")
    Ye = np.load(Path(args.dataset) / "Y_element.npy", mmap_mode="r")
    Yo = np.load(Path(args.dataset) / "Y_organism.npy", mmap_mode="r")
    enc = json.load(open(Path(args.dataset) / "encoders.json"))
    inv = {v: k for k, v in enc["element"].items()}
    yraw = np.asarray(Ye[te])
    y = np.array([CLASS_ID.get(collapse_element(inv[int(r)]), -1) for r in yraw], dtype=np.int64)
    keep = y >= 0
    te, y = te[keep], y[keep]
    if args.max_eval and len(te) > args.max_eval:
        rng = np.random.default_rng(0)
        s = np.sort(rng.choice(len(te), args.max_eval, replace=False))
        te, y = te[s], y[s]
    # sort for efficient memmap gather
    order = np.argsort(te)
    te, y = te[order], y[order]
    org = np.asarray(Yo[te]).astype(np.int64)
    print(f"[{time.time()-t0:.0f}s] test={len(te)}  classes={CLASS_ORDER}", flush=True)

    # --- normalize per-organism with the training-split normalizer -------------
    nz = np.load(Path(args.run_dir) / "organism_profile_normalization.npz")
    means = nz["means"].astype(np.float32)   # [n_org, 7, 475]
    stds = nz["stds"].astype(np.float32)
    X = np.load(Path(args.dataset) / "X_raw.npy", mmap_mode="r")
    xr = np.asarray(X[te], dtype=np.float32)                       # [N,7,475]
    xn = (xr - means[org]) / stds[org]
    del xr
    print(f"[{time.time()-t0:.0f}s] normalized profiles {xn.shape}", flush=True)

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    import train_taxonomy_multitask  # noqa: F401  (registers custom layers)
    model = tf.keras.models.load_model(args.model, compile=False)
    print("outputs:", output_names(model), flush=True)

    def acc(xin):
        p = model.predict(np.transpose(xin, (0, 2, 1)), batch_size=1024, verbose=0)
        yp = np.argmax(get_element_output(p, model), axis=1)
        overall = float(np.mean(yp == y))
        per = {CLASS_ORDER[c]: float(np.mean(yp[y == c] == c))
               for c in range(len(CLASS_ORDER)) if (y == c).any()}
        return overall, per

    base_o, base_c = acc(xn)
    print(f"[{time.time()-t0:.0f}s] BASELINE overall={base_o:.4f}", flush=True)

    drop, keeponly = {}, {}
    for i, p in enumerate(CANONICAL):
        xa = xn.copy(); xa[:, i, :] = 0.0                          # drop p
        do, dc = acc(xa)
        drop[p] = {"overall": do, "overall_loss": base_o - do,
                   "per_class_loss": {k: base_c[k] - dc.get(k, 0.0) for k in base_c}}
        xk = np.zeros_like(xn); xk[:, i, :] = xn[:, i, :]          # keep only p
        ko, kc = acc(xk)
        keeponly[p] = {"overall": ko, "per_class": kc}
        print(f"  {p:6s}  drop_loss={base_o-do:+.4f}   keep_only={ko:.4f}", flush=True)

    out = {"model": args.model, "n": int(len(te)), "params": list(CANONICAL),
           "classes": list(CLASS_ORDER), "baseline_overall": base_o,
           "baseline_per_class": base_c, "drop_one": drop, "keep_only": keeponly}
    json.dump(out, open(Path(args.out) / f"param_ablation__{Path(args.run_dir).name}.json", "w"), indent=2)
    print("\n== marginal importance (drop-one overall loss), ranked ==")
    for p, d in sorted(drop.items(), key=lambda kv: -kv[1]["overall_loss"]):
        print(f"  {p:6s}  -{d['overall_loss']:.4f}")
    print("WROTE param_ablation.json", flush=True)


if __name__ == "__main__":
    main()
