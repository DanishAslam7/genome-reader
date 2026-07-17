# Genome Reader - biophysical profiles as a universal, interpretable genomic code

Code for *"[paper title], Aslam et al."* (Nucleic Acids Research, submitted).

A multitask convolutional network reads five genomic element classes — promoters, gene and exon
boundaries, start and stop codons — from **DNA mechanical profiles alone** (seven biophysical
parameters over 475 positions, no sequence), across 32 organisms spanning four eukaryotic kingdoms.
The representation transfers across kingdoms as well as sequence does, and *better* than a
matched-training sequence model on organisms as distant as prokaryotes; channel ablation resolves
which physical property encodes which element.

> **Repository:** https://github.com/DanishAslam7/genome-reader
> **Archived code:** https://doi.org/10.5281/zenodo.21400171 (Zenodo concept DOI — always resolves to
> the latest version; this GitHub repository is the working mirror, snapshotted at each release).
> **Trained model weights and the exact split indices needed to reproduce every comparison** are
> deposited as a separate Zenodo record — https://doi.org/10.5281/zenodo.21403873 (CC-BY-4.0) —
> because they are too large for the code repository.

## What is (and isn't) here

This repository contains **code only**. The large artefacts are deposited on Zenodo or are
regenerable from public data (see *Data* below), and are excluded via `.gitignore`.

```
code/         model, baselines, and all evaluation scripts (flat layout — they import each other)
  train_taxonomy_multitask.py     the multitask model (training + eval entry point)
  kmer_baseline.py                classical k-mer + logistic-regression floor
  finetune_dnalm.py               Nucleotide Transformer finetuning (pretrained & from-scratch)
  eval_diatoms.py / eval_prok.py  zero-shot external validation (biophysics)
  eval_*_kmer.py                  k-mer floor on the external sets
  eval_withseq_external.py        with-sequence arm, external
  eval_dnalm_external.py          DNA-LM arms, external
  eval_norm_ablation*.py          the normalization-confound experiment (Figure 4)
  eval_param_ablation.py          per-parameter mechanism ablation (Figure 6)
  dump_model_summary.py           read the trained graph back out of a checkpoint
  artifact_check_loko_biophys.py  rules out the leave-one-kingdom-out artefact
  derive_table_windows.py         cohort-derived segment windows (CESL)
  build_*.py                      dataset / manifest / knowledge-graph construction
profiling/    biophysical profiling pipeline (run_profile_batch.py + asyncPython*.py)
figures/      one script per figure; run to regenerate the PDFs/PNGs
  phylopic/   public-domain organism silhouettes for Figure 1 (see CREDITS.json)
```

## Data

- **Genome assemblies** (32 cohort organisms + 2 diatoms): public, NCBI, accessions in the paper's
  Supplementary Table S1.
- **Biophysical parameter tables** (tri- and tetranucleotide structural/energetic values): from the
  molecular-dynamics framework of the predecessor study (Sharma *et al.*, *Int J Biol Macromol* 329,
  2025, 147488); used by `profiling/asyncPython*.py`.
- **Prokaryotic promoter set** (external validation): experimentally validated compilation from
  *Nucleic Acids Research* 53(21), gkaf1310, Supplementary Data S1.
- **Trained weights + split indices** (`provenance_{train,val,test}_idx.npy`): Zenodo,
  https://doi.org/10.5281/zenodo.21403873 (38 runs; see its `MANIFEST.csv`). These indices are the
  reproducibility crux — they let you score any baseline on the *identical* rows the model used.
- **The full profile tensor (~184 GB)** is not deposited; regenerate it with `profiling/` from the
  public assemblies, or request it from the corresponding author.

## Reproducing a result

1. Obtain the genome assemblies (Table S1) and profile them with `profiling/run_profile_batch.py`
   (`--mode both --length 501`), **or** download the profiled dataset from Zenodo.
2. Build the dataset and manifests (`code/build_*.py`).
3. Train, or download the trained weights from Zenodo.
4. Evaluate on the matched rows using the deposited `provenance_*_idx.npy`; every comparison in the
   paper is scored this way.
5. Regenerate figures with the scripts in `figures/` (each is self-contained and reads the numbers
   from the evaluation outputs / paper).

## The DNA language-model baseline (pretrained vs from-scratch)

Both NT-50M arms come from the *same* scripts; the only difference is the `--from-scratch` flag,
which random-initializes the architecture instead of loading pretrained weights. This is what lets
the paper claim "identical architecture, differing only in initialization" — the two arms are one
code path. To finetune and evaluate each arm on the matched splits:

```bash
# pretrained arm
python code/finetune_dnalm.py --model InstaDeepAI/nucleotide-transformer-v2-50m-multi-species \
    --run-dir <RUN_DIR>
# from-scratch control (no pretraining exposure)
python code/finetune_dnalm.py --model InstaDeepAI/nucleotide-transformer-v2-50m-multi-species \
    --run-dir <RUN_DIR> --from-scratch

# external evaluation (diatoms / prokaryotes) — add --from-scratch for the control arm
python code/eval_dnalm_external.py --model <FINETUNED_DIR> [--from-scratch]
```

DNABERT-2 was attempted but is not included: its custom flash-attention kernel is incompatible with
the available Triton runtime (documented in the paper, Section 2.9).

## Environment

Training and evaluation: TensorFlow 2 (CUDA 12.4), NumPy, pandas, scikit-learn, pyarrow. DNA-LM
finetuning: PyTorch 2.2 + Transformers 4.44 (`nucleotide-transformer-v2-50m-multi-species`). Figures:
matplotlib + Pillow. Exact versions in `requirements.txt`.

## Citation

See `CITATION.cff`. Please cite both this work and the predecessor descriptive study (Sharma *et
al.*, 2025), from which the profiling framework and parameter tables are taken.

## License

Code under the MIT License (`LICENSE`). Organism silhouettes in `figures/phylopic/` are public
domain (PhyloPic; per-image credit in `CREDITS.json`).
