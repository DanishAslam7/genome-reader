#!/usr/bin/env python3
"""Rule out artifacts in the biophys-LOKO near-chance transfer result.

For a held-out kingdom (default animalia), checks:
  1. Held-out organisms have VALID profile-normalization stats in the fold
     (finite, non-zero std, counts>0) -> not garbage input.
  2. The biophys fold's normalization npz is IDENTICAL to the withseq fold's
     (same kingdom) -> normalization is not the differentiator (withseq transfers
     fine with the same stats, so a biophys-only failure is real).
  3. In-distribution learnability: biophys BASELINE per-kingdom accuracy (trained
     WITH the kingdom) vs the fold's zero-shot accuracy -> a big drop = transfer
     failure, not a data problem.
  4. The fold's test set really is the held-out kingdom only.
"""
import json, os, sys, glob
import numpy as np
import pandas as pd

KING = sys.argv[1] if len(sys.argv) > 1 else "animalia"

def newest(pat):
    xs = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
    return xs[0] if xs else None

bio  = newest(f"taxonomy_multitask_out_run_*_loko_biophys_{KING}_seed42")
ws   = newest(f"taxonomy_multitask_out_run_*_loko_withseq_{KING}_seed42")
base = newest("taxonomy_multitask_out_run_*_loko_baseline_biophys_seed42")
print(f"held-out kingdom: {KING}")
print(f"  biophys fold : {bio}")
print(f"  withseq fold : {ws}")
print(f"  biophys base : {base}\n")

# organisms in the held-out kingdom
dm = pd.read_parquet("dataset_reprofiled_32/dataset_meta.parquet", columns=["organism", "kingdom"])
king_orgs = sorted(dm.loc[dm.kingdom == KING, "organism"].unique().tolist())
print(f"[orgs] {KING} organisms ({len(king_orgs)}): {king_orgs}\n")

# ---- 1 & 2: normalization stats for held-out orgs, biophys vs withseq ----
nb = np.load(os.path.join(bio, "organism_profile_normalization.npz"), allow_pickle=True)
names = list(nb["organism_names"])
idx = [names.index(o) for o in king_orgs if o in names]
means, stds, counts = nb["means"][idx], nb["stds"][idx], nb["counts"][idx]
print("[1] held-out org normalization in the biophys fold:")
print(f"    counts: {counts.tolist()}")
print(f"    stds finite & >0 for all held-out orgs? {np.all(np.isfinite(stds)) and np.all(stds > 0)}")
print(f"    means finite? {bool(np.all(np.isfinite(means)))}  | std range [{stds.min():.3g}, {stds.max():.3g}]")
if ws and os.path.exists(os.path.join(ws, "organism_profile_normalization.npz")):
    nw = np.load(os.path.join(ws, "organism_profile_normalization.npz"), allow_pickle=True)
    same = (np.allclose(nb["means"], nw["means"], equal_nan=True)
            and np.allclose(nb["stds"], nw["stds"], equal_nan=True))
    print(f"[2] biophys-fold npz IDENTICAL to withseq-fold npz? {same}")
    print("    -> if True, normalization is the same for both; withseq transfers fine,")
    print("       so biophys's collapse is NOT a normalization artifact.\n")

# ---- 3: in-dist learnable vs zero-shot ----
pk = json.load(open(os.path.join(base, "per_kingdom_element_accuracy.json")))["per_kingdom"]
indist = pk.get(KING, {}).get("element_masked_accuracy")
zshot = json.load(open(os.path.join(bio, "metrics.json")))["test_results"]["element_masked_accuracy"]
print(f"[3] biophys {KING}: in-distribution (baseline) = {indist:.4f}  vs  zero-shot (fold) = {zshot:.4f}")
print(f"    drop = {(indist - zshot)*100:.1f} pts  -> {'TRANSFER FAILURE (learnable in-dist, collapses zero-shot)' if indist-zshot>0.2 else 'small gap'}\n")

# ---- 4: fold test set is the held-out kingdom only ----
prov = np.load(os.path.join(bio, "provenance_test_idx.npy"))
ks = dm.iloc[prov]["kingdom"].value_counts(normalize=True)
print(f"[4] fold test-set kingdom composition: {ks.round(3).to_dict()}")
print(f"    -> should be ~100% {KING}")

print("\nVERDICT: artifact ruled out if [1] stds valid, [2] npz identical to withseq,")
print("         [3] big in-dist->zero-shot drop, [4] test set is the held-out kingdom.")
