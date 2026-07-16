#!/usr/bin/env python3
"""
DNA-LM on the EXTERNAL sets — the sequence-model counterpart to eval_diatoms.py /
eval_prok.py (biophysics). Finetune the LM on the all-32 in-distribution training
set, then predict UNSEEN organisms:
  - diatoms (pt, tp): 4-class element accuracy   (vs biophysics 0.569, k-mer 0.283)
  - prokaryote promoters: prom-recognition rate  (vs biophysics 0.386)

The prokaryote test is the fair one: NT/DNABERT multi-species pretraining is largely
eukaryotic, so bacteria are (likely) genuinely novel — unlike the LOKO folds where the
held-out kingdom was in the LM's pretraining corpus.
"""
import argparse, glob, json, os, time
from pathlib import Path
import numpy as np
from finetune_dnalm import load_split_labels, read_seqs, cap_per_class, build_classifier
from kmer_baseline import CLASS_ORDER, CLASS_ID, collapse_element

PROM_CLASS = 2
DIATOM_ELEMS = ["ee", "es", "ge", "gs", "stac", "stoc"]   # no prom; cds dropped
CANON_TRAIN = ("taxonomy_multitask_out_run_20260621_074207_"
               "canonical_final_d256_cap50k_masked_metric_seed42")


def read_diatom(org_dirs):
    seqs, ys = [], []
    for org, d in org_dirs:
        for el in DIATOM_ELEMS:
            cls = CLASS_ID.get(collapse_element(el))
            if cls is None:
                continue
            p = Path(d) / f"{org}_total_seq_{el}"
            if not p.exists():
                print(f"  diatom skip {org}/{el}: {p} missing"); continue
            with open(p) as fh:
                s = [ln.strip() for ln in fh if ln.strip()]
            seqs += s; ys += [cls] * len(s)
    return seqs, np.array(ys, dtype=np.int64)


def read_prok(seqdir):
    seqs = []
    for f in sorted(glob.glob(str(Path(seqdir) / "*_total_seq"))):
        with open(f) as fh:
            seqs += [ln.strip() for ln in fh if ln.strip()]
    return seqs, np.full(len(seqs), PROM_CLASS, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-run-dir", default=CANON_TRAIN)
    ap.add_argument("--dataset", default="dataset_reprofiled_32")
    ap.add_argument("--manifest", default="sequence_manifest_reprofiled_32.parquet")
    ap.add_argument("--out", default="dnalm_external_out")
    ap.add_argument("--max-train-per-class", type=int, default=30000)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--from-scratch", action="store_true")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    (tr_idx, ytr), _ = load_split_labels(args.train_run_dir, args.dataset)
    tr_idx, ytr = cap_per_class(tr_idx, ytr, args.max_train_per_class)
    tr_seqs = read_seqs(tr_idx, args.manifest)
    print(f"[{time.time()-t0:.0f}s] train={len(tr_idx)}", flush=True)

    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer, DataCollatorWithPadding)
    from datasets import Dataset
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    def tokfn(b):
        return tok(b["seq"], truncation=True, max_length=args.max_length)
    dtr = Dataset.from_dict({"seq": tr_seqs, "labels": ytr.tolist()}).map(
        tokfn, batched=True, remove_columns=["seq"])
    model = build_classifier(args.model, len(CLASS_ORDER), from_scratch=args.from_scratch)
    if args.from_scratch:
        model.init_weights()
    coll = DataCollatorWithPadding(tok)
    targs = TrainingArguments(
        output_dir=str(Path(args.out) / "hf"), num_train_epochs=args.epochs,
        per_device_train_batch_size=32, per_device_eval_batch_size=64,
        learning_rate=2e-5, save_strategy="no", logging_steps=200, report_to=[],
        fp16=torch.cuda.is_available())
    trainer = Trainer(model=model, args=targs, train_dataset=dtr, data_collator=coll)
    trainer.train()
    print(f"[{time.time()-t0:.0f}s] trained", flush=True)

    def predict(seqs):
        ds = Dataset.from_dict({"seq": seqs}).map(tokfn, batched=True, remove_columns=["seq"])
        out = trainer.predict(ds)
        logits = out.predictions[0] if isinstance(out.predictions, tuple) else out.predictions
        return np.argmax(logits, axis=1).astype(np.int64)

    results = {}
    # ---- diatoms (unseen euk lineage) ----
    dseqs, dy = read_diatom([("tp", "tp"), ("pt", "pt")])
    dp = predict(dseqs)
    results["diatoms"] = {"n": int(len(dy)), "accuracy": float((dp == dy).mean()),
        "per_class": {CLASS_ORDER[c]: {"support": int((dy == c).sum()),
                      "recall": float((dp[dy == c] == c).mean())}
                      for c in sorted(set(dy.tolist()))}}
    print(f"[{time.time()-t0:.0f}s] diatoms acc={results['diatoms']['accuracy']:.4f} n={len(dy)}", flush=True)

    # ---- prokaryotes (different domain — the fair test) ----
    pseqs, py = read_prok("prokaryotes/prok_prom_seq")
    pp = predict(pseqs)
    results["prokaryotes"] = {"n": int(len(py)), "prom_rate": float((pp == PROM_CLASS).mean()),
        "dist": {CLASS_ORDER[c]: float((pp == c).mean()) for c in range(len(CLASS_ORDER))}}
    print(f"[{time.time()-t0:.0f}s] prok prom_rate={results['prokaryotes']['prom_rate']:.4f} n={len(py)}", flush=True)

    results["_meta"] = {"model": args.model, "train": "all-32 in-dist",
                        "diatom_chance": 0.25, "prok_chance": 0.20}
    mtag = ("scratch_" if args.from_scratch else "") + args.model.split("/")[-1]
    json.dump(results, open(Path(args.out) / f"{mtag}_external.json", "w"), indent=2)
    print(f"\nEXTERNAL {mtag}: diatoms acc={results['diatoms']['accuracy']:.4f} | "
          f"prok prom_rate={results['prokaryotes']['prom_rate']:.4f}", flush=True)


if __name__ == "__main__":
    main()
