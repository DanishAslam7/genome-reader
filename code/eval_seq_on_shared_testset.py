#!/usr/bin/env python3
"""Evaluate the sequence-only model on the SHARED multitask test split, on a
COMMON label space, so its element accuracy is comparable to the biophys/both arms.

Two reconciliations are required (verified against dataset_meta + per-class CSVs):

1. Join key. dataset_meta has BOTH a coarse `element` column (short: cds, ee, es,
   ge, gs, prom, stac, stoc -- matches the seq model) and a fine `element_label`
   column (long: start_codon, stop_codon, gene_start, gene_end, ...). We join on
   the coarse `element` column + local index.

2. Label space. The multitask main element head is 5-way COLLAPSED:
       exon_boundary = ee+es,  gene_boundary = ge+gs,  prom,  stac,  stoc
   and EXCLUDES cds (handled by a separate cds_detector_head). The seq model is
   8-way. To compare on the multitask's task, we drop cds rows, collapse both the
   seq model's predictions AND the truth into the 5-class scheme, and report
   accuracy over those rows -- directly comparable to multitask element_masked_accuracy.

Run on a GPU node (genome_reader_tf). See qsub_eval_seq_shared.sh.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# seq short element -> multitask 5-class collapsed label. cds is excluded (the
# multitask element head does not classify it). Anything mapping to None is dropped.
COLLAPSE = {
    "ee": "exon_boundary", "es": "exon_boundary",
    "ge": "gene_boundary", "gs": "gene_boundary",
    "prom": "prom", "stac": "stac", "stoc": "stoc",
    "cds": None,
}
TARGET_ELEMENTS = ["ee", "es", "ge", "gs", "prom", "stac", "stoc"]  # rows kept (short `element`)
COLLAPSED_CLASSES = ["exon_boundary", "gene_boundary", "prom", "stac", "stoc"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq-x", type=Path, default=Path("sequence_inputs_all/X_sequence.npy"))
    ap.add_argument("--seq-meta", type=Path,
                    default=Path("sequence_inputs_all/sequence_metadata.parquet"))
    ap.add_argument("--run-dir", type=Path, default=Path("ablation_seq_only_run"))
    ap.add_argument("--model", type=Path, default=None,
                    help="Model file; default <run-dir>/best.keras (falls back to final.keras).")
    ap.add_argument("--mt-dataset", type=Path, default=Path("dataset_reprofiled_32"))
    ap.add_argument("--provenance-test-idx", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=1024)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    encoders = json.loads((args.run_dir / "encoders.json").read_text())
    uni = list(encoders["element_universal"])          # 8-way class order the model predicts
    print(f"seq model universal classes ({len(uni)}): {uni}")
    print(f"collapsed target classes ({len(COLLAPSED_CLASSES)}): {COLLAPSED_CLASSES}")

    # ----- the exact multitask test rows, on the coarse `element` column -----
    prov = np.load(args.provenance_test_idx)
    dm = pd.read_parquet(args.mt_dataset / "dataset_meta.parquet",
                         columns=["organism", "element", "local_seq_idx"])
    if prov.max() >= len(dm):
        raise SystemExit(f"provenance idx max {prov.max()} >= dataset_meta rows {len(dm)}")
    test_meta = dm.iloc[prov].reset_index(drop=True)
    n_test_all = len(test_meta)

    # keep only the rows the multitask element head was scored on (5-class universe; no cds)
    test_meta = test_meta[test_meta["element"].isin(TARGET_ELEMENTS)].copy()
    n_keep = len(test_meta)
    print(f"shared test rows: {n_test_all:,} total | {n_keep:,} in 5-class element universe")
    print("  breakdown:", test_meta["element"].value_counts().to_dict())

    # ----- map into the seq dataset via (organism, short-element, local idx) -----
    # dataset_meta's short labels live in `element`; the seq dataset's short labels
    # live in `element_label` (its `element` column holds long descriptive names).
    seqm = pd.read_parquet(args.seq_meta,
                           columns=["global_row", "organism", "element_label", "local_idx"])
    merged = test_meta.merge(
        seqm,
        left_on=["organism", "element", "local_seq_idx"],
        right_on=["organism", "element_label", "local_idx"],
        how="inner", validate="one_to_one",
    )
    coverage = len(merged) / n_keep if n_keep else 0.0
    print(f"matched into seq dataset: {len(merged):,} / {n_keep:,} ({coverage:.4%})")
    if coverage < 0.99:
        print("WARNING: <99% matched; check key alignment.")

    merged = merged.sort_values("global_row")
    rows = merged["global_row"].to_numpy()
    truth_short = merged["element"].to_numpy()
    truth_collapsed = np.array([COLLAPSE[e] for e in truth_short])

    # ----- gather sequences + run model -----
    X = np.load(args.seq_x, mmap_mode="r")
    Xg = np.asarray(X[rows])
    print(f"gathered X: {Xg.shape} {Xg.dtype} ({Xg.nbytes/1e9:.1f} GB)")

    import tensorflow as tf
    model_path = args.model or (args.run_dir / "best.keras")
    if not model_path.exists():
        model_path = args.run_dir / "final.keras"
    print(f"loading model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    out_names = list(model.output_names)
    print("model outputs:", out_names)

    preds = []
    bs = args.batch_size
    for i in range(0, len(rows), bs):
        xb = Xg[i:i + bs].astype(np.float32)
        out = model.predict(xb, verbose=0)
        if isinstance(out, dict):
            logits = out["element_universal"]
        elif isinstance(out, (list, tuple)):
            logits = out[out_names.index("element_universal")]
        else:
            logits = out
        preds.append(np.asarray(logits).argmax(axis=1))
        if (i // bs) % 50 == 0:
            print(f"  {i + len(xb):,}/{len(rows):,}", flush=True)
    pred_short = np.array(uni)[np.concatenate(preds)]
    # collapse predictions; a 'cds' prediction (not a valid target) maps to "cds" -> always wrong here
    pred_collapsed = np.array([COLLAPSE.get(p) or "cds" for p in pred_short])

    # ----- metrics on the collapsed 5-class space (comparable to multitask) -----
    correct = (pred_collapsed == truth_collapsed)
    acc = float(correct.mean())
    # native 8-way accuracy on the same rows, for reference
    acc_native = float((pred_short == truth_short).mean())

    per_class = {}
    for c in COLLAPSED_CLASSES:
        m = truth_collapsed == c
        per_class[c] = {"n": int(m.sum()),
                        "accuracy": float(correct[m].mean()) if m.any() else None}

    result = {
        "collapsed_element_accuracy": acc,
        "native_8way_accuracy": acc_native,
        "n_evaluated": int(len(rows)),
        "coverage": coverage,
        "collapsed_classes": COLLAPSED_CLASSES,
        "per_class_accuracy": per_class,
        "collapse_map": COLLAPSE,
        "model": str(model_path),
        "provenance_test_idx": str(args.provenance_test_idx),
    }
    out_path = args.out or (args.run_dir / "shared_testset_metrics.json")
    out_path.write_text(json.dumps(result, indent=2) + "\n")

    print("\n==================== RESULT ====================")
    print(f"seq-only COLLAPSED element accuracy (shared test, multitask 5-class space): {acc:.4f}")
    print(f"  (native 8-way accuracy on same rows: {acc_native:.4f})")
    print(f"  evaluated {len(rows):,} rows  (coverage {coverage:.2%})")
    print("  per collapsed class:")
    for c in COLLAPSED_CLASSES:
        pc = per_class[c]
        a = f"{pc['accuracy']:.4f}" if pc["accuracy"] is not None else "  n/a"
        print(f"    {c:14s} n={pc['n']:>8,}  acc={a}")
    print(f"\nCompare to multitask element_masked_accuracy: biophys 0.683 | both 0.708")
    print(f"saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
