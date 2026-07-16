#!/usr/bin/env python3
"""
Dump the trained model's real layer graph, to cross-check Supplementary Figure S1.

FigS_architecture.py is built from train_taxonomy_multitask.py + run_config.json —
the sources the model was built FROM, so it is accurate by construction. This script
closes the loop by reading the graph back out of a saved checkpoint: layer names,
types, output shapes and parameter counts, plus the per-head wiring.

Run wherever the trained-model environment (matching Keras/TensorFlow versions)
is available: an older head-node Keras cannot deserialize checkpoints saved by a
newer Keras ("No module named 'keras.src'").

  python dump_model_summary.py --run-dir <run> --out model_summary_out
"""
import argparse, json, os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",
                    default="taxonomy_multitask_out_run_20260628_143508_loko_baseline_biophys_seed42")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="model_summary_out")
    args = ap.parse_args()
    run = Path(args.run_dir)
    model_path = args.model or str(run / "checkpoints" / "best_model.keras")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    import train_taxonomy_multitask  # registers custom layers (ParameterScale, GRL, ...)

    m = tf.keras.models.load_model(model_path, compile=False)

    rows = []
    for l in m.layers:
        try:
            shp = str(tuple(l.output.shape))
        except Exception:
            shp = "?"
        rows.append({"name": l.name, "type": type(l).__name__,
                     "params": int(l.count_params()), "output_shape": shp})

    summary = {
        "run_dir": run.name,
        "model": model_path,
        "total_params": int(m.count_params()),
        "n_layers": len(m.layers),
        "inputs": [{"name": i.name, "shape": str(tuple(i.shape))} for i in m.inputs],
        "outputs": [o.name.split("/")[0] for o in m.outputs],
        "layers": rows,
    }
    json.dump(summary, open(out / f"model_summary__{run.name}.json", "w"), indent=2)

    # human-readable dump too
    with open(out / f"model_summary__{run.name}.txt", "w") as fh:
        m.summary(print_fn=lambda s: fh.write(s + "\n"), line_length=130)

    # ---- the numbers Supplementary Figure S1 claims ----
    print(f"\ntotal params : {summary['total_params']:,}")
    print(f"layers       : {summary['n_layers']}")
    print(f"inputs       : {[(i['name'], i['shape']) for i in summary['inputs']]}")
    print(f"outputs      : {summary['outputs']}")

    def find(sub):
        return [r for r in rows if sub in r["name"].lower() or sub in r["type"].lower()]

    print("\n--- FigS1 cross-check ---")
    for label, sub in (("stem", "stem"), ("depthwise/grouped conv", "conv1d"),
                       ("layernorm", "layernormalization"), ("residual add", "add"),
                       ("global pool", "globalaverage"), ("gradient reversal", "reversal"),
                       ("shared dense", "shared_dense")):
        hits = find(sub)
        print(f"  {label:24s} {len(hits):3d}  e.g. {[h['name'] for h in hits[:3]]}")
    print("\n--- output shapes along the trunk (expect 475 -> 238 -> 119) ---")
    for r in rows:
        if r["type"] in ("Conv1D",) and "475" in r["output_shape"] or \
           r["type"] in ("Conv1D",) and ("238" in r["output_shape"] or "119" in r["output_shape"]):
            print(f"  {r['name']:34s} {r['output_shape']:>20s}  {r['params']:>10,}")
    print(f"\nWROTE {out}/model_summary__{run.name}.json  (+ .txt)")


if __name__ == "__main__":
    main()
