#!/usr/bin/env python3
"""
Finetune a DNA language model (NT-50M / DNABERT-2) for 5-class element recognition
on the SAME provenance split as the biophysics runs, scored on the identical test
rows. The strong-sequence comparator to biophysics.

  in-distribution: --run-dir <canonical all-32 run>   -> vs biophys 0.685
  LOKO fold:       --run-dir <loko_biophys_<k> run>   -> vs biophys ~0.54 (transfer)

Split + labels come from the run dir's provenance_{train,test}_idx.npy (LOKO holdout
already baked in). 5 classes {exon_boundary, gene_boundary, prom, stac, stoc}; cds
dropped (matches the biophysics element head). Runs on GPU compute node; HF offline.
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
from kmer_baseline import collapse_element, CLASS_ORDER, CLASS_ID, resolve_local


def build_classifier(model_name, n_labels, from_scratch=False):
    """AutoModelForSequenceClassification for NT; a manual mean-pool + linear head on
    the AutoModel encoder for DNABERT-2 (which ships no ForSequenceClassification and
    whose remote BertConfig clashes with the built-in one).
    from_scratch=True -> random-init the architecture (controlled pretraining ablation)."""
    import torch.nn as nn
    from transformers import (AutoModelForSequenceClassification, AutoModel, BertConfig, AutoConfig)
    if "dnabert" not in model_name.lower():
        if from_scratch:
            cfg = AutoConfig.from_pretrained(model_name, num_labels=n_labels, trust_remote_code=True)
            return AutoModelForSequenceClassification.from_config(cfg, trust_remote_code=True)
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=n_labels, trust_remote_code=True)
    # DNABERT-2's remote BertModel declares config_class = built-in BertConfig, so load
    # the config with the built-in class (not the remote one) to avoid the 4.44 mismatch.
    cfg = BertConfig.from_pretrained(model_name)
    for k in ("use_flash_attn", "flash_attn"):        # avoid the Triton flash-attn path
        if hasattr(cfg, k):
            setattr(cfg, k, False)
    encoder = AutoModel.from_pretrained(model_name, config=cfg, trust_remote_code=True)
    hidden = int(getattr(cfg, "hidden_size", 768))

    class DNAClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(hidden, n_labels)
            self.config = cfg
        def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            h = out[0] if isinstance(out, (tuple, list)) else getattr(out, "last_hidden_state", out)
            if attention_mask is not None:
                m = attention_mask.unsqueeze(-1).to(h.dtype)
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            else:
                pooled = h.mean(1)
            logits = self.classifier(self.dropout(pooled))
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(logits, labels)
            from transformers.modeling_outputs import SequenceClassifierOutput
            return SequenceClassifierOutput(loss=loss, logits=logits)
    return DNAClassifier()


def load_split_labels(run_dir, dataset):
    tr = np.load(Path(run_dir) / "provenance_train_idx.npy")
    te = np.load(Path(run_dir) / "provenance_test_idx.npy")
    Ye = np.load(Path(dataset) / "Y_element.npy", mmap_mode="r")
    enc = json.load(open(Path(dataset) / "encoders.json"))
    inv = {v: k for k, v in enc["element"].items()}
    def to5(idx):
        y = np.array([CLASS_ID.get(collapse_element(inv[int(r)]), -1)
                      for r in np.asarray(Ye[idx])], dtype=np.int64)
        keep = y >= 0
        return np.asarray(idx)[keep], y[keep]
    return to5(tr), to5(te)


def read_seqs(idx, manifest, seq_root=None):
    import pyarrow.parquet as pq
    idx = np.asarray(idx)
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
        bs = t.column("seq_byte_start").to_numpy(); be = t.column("seq_byte_end").to_numpy()
        for i in np.where(m)[0]:
            off[int(gr[i])] = (sp[i], int(bs[i]), int(be[i]))
        if len(off) >= len(need):
            break
    seqs = [""] * len(idx)
    by_file, pcache = {}, {}
    for row, gr in enumerate(idx):
        sp, s, e = off[int(gr)]; by_file.setdefault(sp, []).append((row, s, e))
    for sp, items in by_file.items():
        local = pcache.get(sp) or resolve_local(sp, seq_root); pcache[sp] = local
        with open(local, "rb") as fh:
            for row, s, e in items:
                fh.seek(s); seqs[row] = fh.read(e - s).strip().decode("ascii", "ignore")
    return seqs


def cap_per_class(idx, y, cap, seed=0):
    if not cap or cap <= 0:
        return idx, y
    rng = np.random.default_rng(seed); sel = []
    for c in range(len(CLASS_ORDER)):
        ci = np.where(y == c)[0]
        if len(ci) > cap:
            ci = rng.choice(ci, cap, replace=False)
        sel.append(ci)
    sel = np.sort(np.concatenate(sel))
    return idx[sel], y[sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", default="dataset_reprofiled_32")
    ap.add_argument("--manifest", default="sequence_manifest_reprofiled_32.parquet")
    ap.add_argument("--seq-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-train-per-class", type=int, default=30000)
    ap.add_argument("--max-eval", type=int, default=40000)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--from-scratch", action="store_true",
                    help="random-init the architecture (controlled no-pretraining baseline)")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    (tr_idx, ytr), (te_idx, yte) = load_split_labels(args.run_dir, args.dataset)
    tr_idx, ytr = cap_per_class(tr_idx, ytr, args.max_train_per_class)
    if args.max_eval and len(te_idx) > args.max_eval:
        rng = np.random.default_rng(1)
        s = np.sort(rng.choice(len(te_idx), args.max_eval, replace=False))
        te_idx, yte = te_idx[s], yte[s]
    print(f"[{time.time()-t0:.0f}s] train={len(tr_idx)} test={len(te_idx)} classes={CLASS_ORDER}", flush=True)

    tr_seqs = read_seqs(tr_idx, args.manifest, args.seq_root)
    te_seqs = read_seqs(te_idx, args.manifest, args.seq_root)
    print(f"[{time.time()-t0:.0f}s] read sequences", flush=True)

    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer, DataCollatorWithPadding)
    from datasets import Dataset
    # Both NT-v2 and DNABERT-2 ship custom modeling code and require trust_remote_code;
    # AutoModelForSequenceClassification + trust_remote_code gives the classification head.
    trc = True
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=trc)
    def tokfn(b):
        return tok(b["seq"], truncation=True, max_length=args.max_length)
    dtr = Dataset.from_dict({"seq": tr_seqs, "labels": ytr.tolist()}).map(
        tokfn, batched=True, remove_columns=["seq"])
    dte = Dataset.from_dict({"seq": te_seqs, "labels": yte.tolist()}).map(
        tokfn, batched=True, remove_columns=["seq"])
    model = build_classifier(args.model, len(CLASS_ORDER), from_scratch=args.from_scratch)
    if args.from_scratch:
        model.init_weights()   # ensure random init (no cached pretrained tensors)
    coll = DataCollatorWithPadding(tok)
    targs = TrainingArguments(
        output_dir=str(Path(args.out) / "hf"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=64,
        learning_rate=args.lr, save_strategy="no", logging_steps=200, report_to=[],
        fp16=torch.cuda.is_available())
    trainer = Trainer(model=model, args=targs, train_dataset=dtr, data_collator=coll)
    trainer.train()
    print(f"[{time.time()-t0:.0f}s] trained", flush=True)

    out = trainer.predict(dte)
    logits = out.predictions[0] if isinstance(out.predictions, tuple) else out.predictions
    pred = np.argmax(logits, axis=1)
    acc = float((pred == yte).mean())
    per_class = {}
    for c, name in enumerate(CLASS_ORDER):
        m = yte == c
        per_class[name] = {"support": int(m.sum()),
                           "recall": float((pred[m] == c).mean()) if m.any() else None}
    result = {"model": args.model, "run_dir": args.run_dir,
              "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
              "accuracy": acc, "per_class": per_class}
    mtag = ("scratch_" if args.from_scratch else "") + args.model.split("/")[-1]
    rtag = Path(args.run_dir).name
    json.dump(result, open(Path(args.out) / f"{mtag}__{rtag}.json", "w"), indent=2)
    print(f"\nACCURACY={acc:.4f} (model={mtag}, split={rtag})  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
