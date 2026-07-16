#!/usr/bin/env python3
"""
Figure 4 — evaluation-only normalization ablation (profile-only arm).

WHY THIS IS EVAL-ONLY. Held-out organisms contribute NO training rows, so their
per-organism normalization statistics can only affect TEST-time input; the trained
weights are unaffected by the flag. (Confirmed empirically: the identity and proper
animalia runs have training curves matching to ~3 decimals — GPU nondeterminism.)
We can therefore hold the model FIXED and vary only the test normalization, which
isolates the confound with no second training run and no checkpoint seam:

  identity    : held-out orgs fall back to mean 0 / std 1  -> raw profile scale
  own-profile : held-out orgs standardized from their own profiles (run's npz)

Because a LOKO test set is 100% held-out kingdom, "identity" is exactly the raw
profiles. This also fills the protista fold, whose identity-fallback run was never
launched (it was 4th in a 3-job-per-node queue, and by the time a slot opened the
fix was already in).

  python eval_norm_ablation.py --run-dir taxonomy_multitask_out_run_..._loko_biophys_protista_seed42
  -> norm_ablation_out/norm_ablation__<rundir>.json
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
from eval_diatoms import get_element_output, output_names
from kmer_baseline import collapse_element, CLASS_ORDER, CLASS_ID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", default=None, help="default <run-dir>/checkpoints/best_model.keras")
    ap.add_argument("--dataset", default="dataset_reprofiled_32")
    ap.add_argument("--max-eval", type=int, default=0, help="0 = full test set")
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--out", default="norm_ablation_out")
    args = ap.parse_args()
    run = Path(args.run_dir)
    model_path = args.model or str(run / "checkpoints" / "best_model.keras")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # --- test rows + 5-class labels (drop cds / non-universal) -----------------
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

    # --- normalizer from THIS run (held-out orgs carry own-profile stats) ------
    nz = np.load(run / "organism_profile_normalization.npz")
    means, stds = nz["means"].astype(np.float32), nz["stds"].astype(np.float32)
    counts = nz["counts"]
    names = [str(n) for n in nz["organism_names"]]

    # sanity: every organism in the test set must have own-profile stats here,
    # i.e. this is a proper-norm run and the fold is a true hold-out.
    test_orgs = sorted(set(org.tolist()))
    zero_stat = [names[o] for o in test_orgs if counts[o] == 0]
    print(f"[{time.time()-t0:.0f}s] test={len(te)} rows over {len(test_orgs)} organisms")
    print(f"  test organisms: {[names[o] for o in test_orgs]}")
    if zero_stat:
        raise SystemExit(f"ABORT: {zero_stat} have no stats in this run's npz -> "
                         f"this is an identity-fallback run, not a proper-norm one.")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    import train_taxonomy_multitask  # noqa: F401  (registers custom layers)
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"[{time.time()-t0:.0f}s] loaded {model_path}; outputs={output_names(model)}", flush=True)

    X = np.load(Path(args.dataset) / "X_raw.npy", mmap_mode="r")
    hits = {"identity": np.zeros(len(CLASS_ORDER), np.int64),
            "own_profile": np.zeros(len(CLASS_ORDER), np.int64)}
    tot = np.zeros(len(CLASS_ORDER), np.int64)

    for start in range(0, len(te), args.chunk):
        sl = slice(start, min(start + args.chunk, len(te)))
        xr = np.asarray(X[te[sl]], dtype=np.float32)          # [n,7,475] raw
        yc, oc = y[sl], org[sl]
        variants = {"identity": xr,                            # mean0/std1 fallback
                    "own_profile": (xr - means[oc]) / stds[oc]}
        for tag, xin in variants.items():
            p = model.predict(np.transpose(xin, (0, 2, 1)), batch_size=1024, verbose=0)
            yp = np.argmax(get_element_output(p, model), axis=1)
            for c in range(len(CLASS_ORDER)):
                hits[tag][c] += int(np.sum((yc == c) & (yp == c)))
        for c in range(len(CLASS_ORDER)):
            tot[c] += int(np.sum(yc == c))
        del xr, variants
        print(f"  [{time.time()-t0:.0f}s] {sl.stop}/{len(te)}", flush=True)

    res = {"run_dir": run.name, "model": model_path, "n": int(len(te)),
           "test_organisms": [names[o] for o in test_orgs], "classes": list(CLASS_ORDER)}
    for tag in ("identity", "own_profile"):
        overall = float(hits[tag].sum() / tot.sum())
        res[tag] = {"overall": overall,
                    "per_class": {CLASS_ORDER[c]: float(hits[tag][c] / tot[c])
                                  for c in range(len(CLASS_ORDER)) if tot[c]}}
    res["delta"] = res["own_profile"]["overall"] - res["identity"]["overall"]

    out = Path(args.out) / f"norm_ablation__{run.name}.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"\n  identity     {res['identity']['overall']:.4f}")
    print(f"  own-profile  {res['own_profile']['overall']:.4f}")
    print(f"  delta        {res['delta']:+.4f}")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
