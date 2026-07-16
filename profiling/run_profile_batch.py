#!/usr/bin/env python3
"""Batch-run tri/tetra biophysical profiling over sequence files.

This runner is intentionally conservative:
- lowercase atgc is accepted by converting every sequence to uppercase;
- only non-ATGC characters and wrong lengths are rejected;
- inputs are processed one file at a time to keep memory bounded;
- obvious duplicate aggregate/revision files are skipped by default.

The actual profile calculation is delegated to the existing modules:
`asyncPython.normalize_params` for trimer and
`asyncPython_tetra.normalize_params` for tetramer.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas should exist in the conda env.
    pd = None


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = CODE_DIR.parent
DEFAULT_OUT_ROOT = DEFAULT_DATASET_ROOT / "reprofiled_profiles"

SKIP_DIR_NAMES = {
    "code",
    "profiling_logs",
    "reprofiled_profiles",
    "reprofiled_inputs_uppercase_501",
    "local_scratch",
}
SKIP_FILE_NAMES = {
    "revision_human_ee",
    "revision_human_es",
    "total_stop_codon_Animal",
    "total_stop_codon_Fungi",
    "total_stop_codon_Plant",
}

ORGANISM_TO_KINGDOM = {
    "fly": "animalia",
    "human": "animalia",
    "mouse": "animalia",
    "rat": "animalia",
    "worm": "animalia",
    "klactis": "fungi",
    "scerevisiae": "fungi",
    "ylipolytica": "fungi",
    "arabidopsis": "plantae",
    "zeamays": "plantae",
    "pfalciparum": "protista",
    # New NCBI RefSeq organisms (extracted_new/); kingdoms per download_plan.tsv.
    "toxoplasma_gondii": "protista",
    "tetrahymena_thermophila": "protista",
    "chlamydomonas_reinhardtii": "protista",
    "trypanosoma_brucei": "protista",
    "aspergillus_fumigatus": "fungi",
    "aspergillus_niger": "fungi",
    "schizosaccharomyces_pombe": "fungi",
    "cryptococcus_neoformans": "fungi",
    "neurospora_crassa": "fungi",
    "populus_trichocarpa": "plantae",
    "triticum_aestivum": "plantae",
    "physcomitrium_patens": "plantae",
    "glycine_max": "plantae",
    "solanum_lycopersicum": "plantae",
    "oryza_sativa": "plantae",
    # Balancing additions (animalia + protista -> 8 each).
    "danio_rerio": "animalia",
    "xenopus_tropicalis": "animalia",
    "gallus_gallus": "animalia",
    "dictyostelium_discoideum": "protista",
    "giardia_intestinalis": "protista",
    "leishmania_major": "protista",
}

INPUT_LAYOUT = {
    ("cds",): ("total_cds_", "cds", "cds"),
    ("exon_end",): ("total_ee_", "ee", "ee"),
    ("exon_start",): ("total_es_", "es", "es"),
    ("gene_end",): ("total_gene_end_", "ge", "gene_end"),
    ("gene_start",): ("total_gene_start_", "gs", "gene_start"),
    ("promoters",): ("total_prom_", "prom", "prom"),
    ("start_codon",): ("total_start_codon_", "stac", "start_codon"),
    ("stop_codon",): ("total_stop_codon_", "stoc", "stop_codon"),
    ("expanded_human_analysis", "3utr_end"): ("total_3utr_end_", "3utre", "utr_end"),
    ("expanded_human_analysis", "3utr_start"): ("total_3utr_start_", "3utrs", "utr_start"),
    ("expanded_human_analysis", "5utr_end"): ("total_5utr_end_", "5utre", "5utr_end"),
    ("expanded_human_analysis", "5utr_start"): ("total_5utr_start_", "5utrs", "5utr_start"),
    ("expanded_human_analysis", "enhancer_end"): ("total_end_enhancers_", "ene", "enhancers_end"),
    ("expanded_human_analysis", "enhancer_start"): ("total_start_enhancers_", "ens", "enhancers_start"),
}

TRI_CANONICAL_GROUPS = {
    "intra": ("f", "g", "h", "i", "j", "k"),
    "bp": ("a", "b", "c", "d"),
    "bbone": ("t", "u", "v", "w", "x", "y", "z", "aa", "ab"),
}
TRI_DIRECT_PARAMS = {
    "hbond": "ac",
    "stack": "ad",
    "sol": "ae",
}
TETRA_CANONICAL_GROUPS = {
    "inter": ("l", "ma", "n", "o", "p", "q"),
}
CANONICAL_ORDER = ("bbone", "bp", "hbond", "inter", "intra", "sol", "stack")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--mode", choices=["tri", "tetra", "both"], default="both")
    parser.add_argument("--length", type=int, default=501)
    parser.add_argument("--include-duplicates", action="store_true",
                        help="Include revision_human_* and stop_codon kingdom aggregate files.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a mode/file if its result manifest already exists.")
    parser.add_argument("--only-file", type=Path, default=None,
                        help="Profile one input file only; path may be absolute or relative to dataset root.")
    parser.add_argument("--limit-files", type=int, default=0,
                        help="Debug/smoke-test limit after filtering; 0 means no limit.")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="CSV summary path. Defaults to <out-root>/profile_manifest.csv.")
    parser.add_argument("--progress-log", type=Path, default=None,
                        help="Append timestamped progress events here. Defaults to <out-root>/profile_progress.log.")
    parser.add_argument("--legacy-layout-root", type=Path, default=None,
                        help="Write profile CSVs as kingdom/organism/element/*_norm_* files under this root.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "value"


def import_profile_modules(mode: str) -> dict[str, Any]:
    sys.path.insert(0, str(CODE_DIR))
    modules: dict[str, Any] = {}
    if mode in {"tri", "both"}:
        import asyncPython

        modules["tri"] = asyncPython
    if mode in {"tetra", "both"}:
        import asyncPython_tetra

        modules["tetra"] = asyncPython_tetra
    missing = [name for name, module in modules.items() if not hasattr(module, "normalize_params")]
    if missing:
        raise RuntimeError(
            "Missing normalize_params in "
            + ", ".join(missing)
            + ". Copy the real profiling implementation into "
            + str(CODE_DIR)
            + " before submitting the PBS job."
        )
    return modules


def should_skip_file(path: Path, dataset_root: Path, include_duplicates: bool) -> bool:
    rel = path.relative_to(dataset_root)
    if any(part in SKIP_DIR_NAMES for part in rel.parts[:-1]):
        return True
    if path.name.startswith("."):
        return True
    if (not include_duplicates) and path.name in SKIP_FILE_NAMES:
        return True
    return False


def iter_input_files(args: argparse.Namespace) -> list[Path]:
    dataset_root = args.dataset_root.resolve()
    if args.only_file is not None:
        only = args.only_file
        if not only.is_absolute():
            only = dataset_root / only
        return [only.resolve()]

    files = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file():
            continue
        if should_skip_file(path, dataset_root, args.include_duplicates):
            continue
        files.append(path)
    if args.limit_files and args.limit_files > 0:
        files = files[: args.limit_files]
    return files


def read_sequences(path: Path, length: int) -> tuple[list[str], dict[str, int]]:
    sequences: list[str] = []
    stats = {
        "total_lines": 0,
        "valid_sequences": 0,
        "invalid_length": 0,
        "invalid_alphabet": 0,
        "blank_lines": 0,
    }
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stats["total_lines"] += 1
            seq = line.strip().upper()
            if not seq:
                stats["blank_lines"] += 1
                continue
            if len(seq) != length:
                stats["invalid_length"] += 1
                continue
            if any(base not in "ATGC" for base in seq):
                stats["invalid_alphabet"] += 1
                continue
            sequences.append(seq)
    stats["valid_sequences"] = len(sequences)
    return sequences, stats


def save_value(value: Any, base: Path) -> list[str]:
    saved: list[str] = []
    if pd is not None and isinstance(value, pd.DataFrame):
        out = base.with_suffix(".csv")
        value.to_csv(out)
        saved.append(str(out))
    elif isinstance(value, np.ndarray):
        out = base.with_suffix(".npy")
        np.save(out, value)
        saved.append(str(out))
    elif isinstance(value, (list, tuple)) and value and all(np.isscalar(x) for x in value):
        out = base.with_suffix(".npy")
        np.save(out, np.asarray(value))
        saved.append(str(out))
    elif isinstance(value, (str, int, float, bool)) or value is None:
        out = base.with_suffix(".json")
        out.write_text(json.dumps(value, indent=2) + "\n")
        saved.append(str(out))
    else:
        out = base.with_suffix(".pkl")
        with out.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        saved.append(str(out))
    return saved


def legacy_target(dataset_root: Path, input_file: Path, legacy_root: Path) -> tuple[Path, str]:
    rel = input_file.relative_to(dataset_root)
    parts = tuple(rel.parts[:-1])
    if parts not in INPUT_LAYOUT:
        raise ValueError(f"Cannot map input path to legacy output layout: {rel}")
    prefix, element, element_label = INPUT_LAYOUT[parts]
    if not input_file.name.startswith(prefix):
        raise ValueError(f"Input filename does not match expected prefix {prefix!r}: {rel}")
    organism = input_file.name[len(prefix):]
    if organism not in ORGANISM_TO_KINGDOM:
        raise ValueError(f"Unknown organism {organism!r} parsed from {rel}")
    kingdom = ORGANISM_TO_KINGDOM[organism]
    out_dir = legacy_root / kingdom / organism / element
    return out_dir, f"{element_label}_{organism}"


def canonical_profiles_from_result(result: Any, mode: str) -> dict[str, list[list[float]]]:
    if not (
        isinstance(result, list)
        and result
        and all(isinstance(record, dict) for record in result if record is not None)
    ):
        return {}

    canonical: dict[str, list[list[float]]] = {}
    if mode == "tri":
        for name, keys in TRI_CANONICAL_GROUPS.items():
            rows = []
            for record in result:
                if record is None:
                    rows.append([])
                    continue
                arr = None
                for key in keys:
                    values = np.asarray(record[key], dtype=float)
                    arr = values.copy() if arr is None else arr + values
                rows.append((arr / len(keys)).tolist())
            canonical[name] = rows
        for name, key in TRI_DIRECT_PARAMS.items():
            canonical[name] = [
                ([] if record is None else list(np.asarray(record[key], dtype=float)))
                for record in result
            ]
    elif mode == "tetra":
        for name, keys in TETRA_CANONICAL_GROUPS.items():
            rows = []
            for record in result:
                if record is None:
                    rows.append([])
                    continue
                arr = None
                for key in keys:
                    values = np.asarray(record[key], dtype=float)
                    arr = values.copy() if arr is None else arr + values
                rows.append((arr / len(keys)).tolist())
            canonical[name] = rows
    return canonical


def save_profile_result(
    result: Any,
    out_dir: Path,
    *,
    mode: str = "result",
    filename_suffix: str | None = None,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    canonical = canonical_profiles_from_result(result, mode)
    if canonical:
        for key in CANONICAL_ORDER:
            if key not in canonical:
                continue
            rows = canonical[key]
            if filename_suffix:
                out = out_dir / f"{sanitize_name(key)}_norm_{filename_suffix}.csv"
            else:
                out = out_dir / f"{sanitize_name(key)}.csv"
            if pd is not None:
                pd.DataFrame(rows).to_csv(out)
            else:
                np.savetxt(out, np.asarray(rows, dtype=float), delimiter=",")
            saved.append(str(out))
        (out_dir / f"{sanitize_name(mode)}_profile_keys.json").write_text(
            json.dumps([key for key in CANONICAL_ORDER if key in canonical], indent=2) + "\n"
        )
    elif isinstance(result, dict):
        for key, value in result.items():
            saved.extend(save_value(value, out_dir / sanitize_name(key)))
    else:
        saved.extend(save_value(result, out_dir / "profile_result"))
    (out_dir / f"{sanitize_name(mode)}_result_manifest.json").write_text(
        json.dumps({"mode": mode, "saved_files": saved}, indent=2) + "\n"
    )
    return saved


def output_dir_for(out_root: Path, mode: str, dataset_root: Path, input_file: Path) -> Path:
    rel = input_file.relative_to(dataset_root)
    return out_root / mode / rel.parent / input_file.name


def output_target_for(
    args: argparse.Namespace,
    out_root: Path,
    mode: str,
    dataset_root: Path,
    input_file: Path,
) -> tuple[Path, str | None]:
    if args.legacy_layout_root is not None:
        return legacy_target(dataset_root, input_file, args.legacy_layout_root.resolve())
    return output_dir_for(out_root, mode, dataset_root, input_file), None


def append_manifest(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_progress(path: Path, event: str, relative_input: str = "", mode: str = "", message: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a") as handle:
        handle.write(f"{timestamp}\t{event}\t{mode}\t{relative_input}\t{message}\n")
        handle.flush()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    out_root = args.out_root.resolve()
    manifest = args.manifest.resolve() if args.manifest else out_root / "profile_manifest.csv"
    progress_log = args.progress_log.resolve() if args.progress_log else out_root / "profile_progress.log"
    modes = ["tri", "tetra"] if args.mode == "both" else [args.mode]
    manifest_fields = [
        "mode",
        "input_file",
        "relative_input",
        "status",
        "total_lines",
        "valid_sequences",
        "invalid_length",
        "invalid_alphabet",
        "blank_lines",
        "output_dir",
        "seconds",
        "error",
    ]

    print(f"Dataset root: {dataset_root}", flush=True)
    print(f"Output root:  {out_root}", flush=True)
    print(f"Modes:        {','.join(modes)}", flush=True)
    print(f"Progress log: {progress_log}", flush=True)
    if args.legacy_layout_root is not None:
        print(f"Legacy layout root: {args.legacy_layout_root.resolve()}", flush=True)
    append_progress(
        progress_log,
        "start",
        message=(
            f"dataset_root={dataset_root} out_root={out_root} modes={','.join(modes)} "
            f"legacy_layout_root={args.legacy_layout_root.resolve() if args.legacy_layout_root else ''}"
        ),
    )
    modules = import_profile_modules(args.mode)
    print("Preflight passed: profiling modules expose normalize_params.", flush=True)
    append_progress(progress_log, "preflight_ok", message="profiling modules expose normalize_params")
    if args.preflight_only:
        return 0

    input_files = iter_input_files(args)
    if not input_files:
        raise RuntimeError(f"No input files found under {dataset_root}")
    print(f"Input files:  {len(input_files)}", flush=True)
    append_progress(progress_log, "inputs_found", message=f"n_files={len(input_files)}")

    failures = 0
    for file_index, input_file in enumerate(input_files, start=1):
        rel = input_file.relative_to(dataset_root)
        print(f"\n=== {rel} ===", flush=True)
        append_progress(progress_log, "file_start", str(rel), message=f"file={file_index}/{len(input_files)}")
        sequences, seq_stats = read_sequences(input_file, length=args.length)
        print(
            "Loaded "
            f"{seq_stats['valid_sequences']:,}/{seq_stats['total_lines']:,} valid sequences "
            f"(bad length={seq_stats['invalid_length']:,}, bad alphabet={seq_stats['invalid_alphabet']:,}, "
            f"blank={seq_stats['blank_lines']:,})",
            flush=True,
        )
        append_progress(
            progress_log,
            "file_loaded",
            str(rel),
            message=(
                f"file={file_index}/{len(input_files)} valid={seq_stats['valid_sequences']} "
                f"total={seq_stats['total_lines']} invalid_length={seq_stats['invalid_length']} "
                f"invalid_alphabet={seq_stats['invalid_alphabet']} blank={seq_stats['blank_lines']}"
            ),
        )
        if not sequences:
            failures += 1
            for mode in modes:
                row = {
                    "mode": mode,
                    "input_file": str(input_file),
                    "relative_input": str(rel),
                    "status": "no_valid_sequences",
                    **seq_stats,
                    "output_dir": str(output_target_for(args, out_root, mode, dataset_root, input_file)[0]),
                    "seconds": 0.0,
                    "error": "",
                }
                append_manifest(manifest, row, manifest_fields)
                append_progress(progress_log, "file_no_valid_sequences", str(rel), mode)
            if args.fail_fast:
                return 1
            continue

        for mode in modes:
            mode_out, filename_suffix = output_target_for(args, out_root, mode, dataset_root, input_file)
            result_manifest = mode_out / f"{sanitize_name(mode)}_result_manifest.json"
            if args.skip_existing and result_manifest.exists():
                print(f"[{mode}] skip existing {mode_out}", flush=True)
                append_progress(progress_log, "mode_skip_existing", str(rel), mode, str(mode_out))
                row = {
                    "mode": mode,
                    "input_file": str(input_file),
                    "relative_input": str(rel),
                    "status": "skipped_existing",
                    **seq_stats,
                    "output_dir": str(mode_out),
                    "seconds": 0.0,
                    "error": "",
                }
                append_manifest(manifest, row, manifest_fields)
                continue
            started = time.time()
            status = "ok"
            error = ""
            try:
                print(f"[{mode}] profiling {len(sequences):,} sequences", flush=True)
                append_progress(progress_log, "mode_start", str(rel), mode, f"n_sequences={len(sequences)}")
                result = modules[mode].normalize_params(sequences)
                append_progress(progress_log, "mode_normalized", str(rel), mode, "saving outputs")
                save_profile_result(result, mode_out, mode=mode, filename_suffix=filename_suffix)
                print(f"[{mode}] wrote {mode_out}", flush=True)
                append_progress(progress_log, "mode_done", str(rel), mode, f"seconds={round(time.time() - started, 3)} output={mode_out}")
            except Exception as exc:
                failures += 1
                status = "failed"
                error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                (mode_out / "error.txt").parent.mkdir(parents=True, exist_ok=True)
                (mode_out / "error.txt").write_text(traceback.format_exc())
                print(f"[{mode}] FAILED: {error}", flush=True)
                append_progress(progress_log, "mode_failed", str(rel), mode, error)
                if args.fail_fast:
                    row = {
                        "mode": mode,
                        "input_file": str(input_file),
                        "relative_input": str(rel),
                        "status": status,
                        **seq_stats,
                        "output_dir": str(mode_out),
                        "seconds": round(time.time() - started, 3),
                        "error": error,
                    }
                    append_manifest(manifest, row, manifest_fields)
                    return 1
            row = {
                "mode": mode,
                "input_file": str(input_file),
                "relative_input": str(rel),
                "status": status,
                **seq_stats,
                "output_dir": str(mode_out),
                "seconds": round(time.time() - started, 3),
                "error": error,
            }
            append_manifest(manifest, row, manifest_fields)

    print(f"\nDone. failures={failures}. Manifest: {manifest}", flush=True)
    append_progress(progress_log, "done", message=f"failures={failures} manifest={manifest}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
