#!/usr/bin/env python3
"""Build a Genome Reader dataset from the refreshed canonical profile files.

This builder is safer for the full dataset than the original in-memory builder:
it streams profile CSVs into a NumPy .npy memmap, writes labels directly, and
optionally writes a synchronized sequence manifest using the same valid-row
filter used during profiling. That keeps profile rows and 501 bp sequences
aligned for sequence-branch training.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PARAMETERS = ["bbone", "bp", "hbond", "inter", "intra", "sol", "stack"]
DNA_BYTES = set(b"ACGT")

DATASET_META_SCHEMA = pa.schema(
    [
        ("global_row", pa.int64()),
        ("kingdom", pa.string()),
        ("organism", pa.string()),
        ("element", pa.string()),
        ("element_label", pa.string()),
        ("local_seq_idx", pa.int64()),
    ]
)

SEQUENCE_SCHEMA = pa.schema(
    [
        ("global_row", pa.int64()),
        ("kingdom", pa.string()),
        ("organism", pa.string()),
        ("element", pa.string()),
        ("element_label", pa.string()),
        ("local_seq_idx", pa.int64()),
        ("seq_path", pa.string()),
        ("seq_line_number", pa.int64()),
        ("seq_byte_start", pa.int64()),
        ("seq_byte_end", pa.int64()),
        ("seq_length", pa.int64()),
        ("seq_start", pa.int64()),
        ("seq_end", pa.int64()),
        ("strand", pa.string()),
        ("external_location", pa.string()),
    ]
)


@dataclass(frozen=True)
class ProfileFile:
    parameter: str
    file_path: Path
    n_positions: int


@dataclass(frozen=True)
class ProfileGroup:
    kingdom: str
    organism: str
    element: str
    element_label: str
    output_dir: Path
    input_file: Path | None
    relative_input: str | None
    n_rows: int
    files: dict[str, ProfileFile]


def parse_profile_filename(path: Path, organism: str) -> tuple[str, str]:
    parts = path.stem.split("_norm_", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse profile filename: {path.name}")
    parameter, remainder = parts
    suffix = f"_{organism}"
    if remainder.endswith(suffix):
        element_label = remainder[: -len(suffix)]
    else:
        element_label = "_".join(remainder.split("_")[:-1])
    return parameter, element_label


def count_header_positions(path: Path) -> int:
    with path.open("rb") as fh:
        header = fh.readline().decode("utf-8", errors="replace").rstrip("\r\n")
    return max(len(header.split(",")) - 1, 0)


def resolve_manifest_path(_path_value: str, fallback: Path) -> Path:
    # Prefer the manifest's absolute input_file when present (covers sequences that
    # live in different roots, e.g. Biophysical_Profiling_Datasets/ vs
    # ncbi_inspect/extracted_new/); otherwise fall back to sequence_root/relative.
    if _path_value:
        p = Path(_path_value)
        if p.is_absolute() and p.exists():
            return p
    return fallback


def load_profile_manifest(
    manifest_path: Path,
    profile_root: Path,
    sequence_root: Path,
) -> dict[Path, dict[str, Any]]:
    by_output_dir: dict[Path, dict[str, Any]] = {}
    with manifest_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("mode") != "tri":
                continue
            if row.get("status") != "ok":
                raise ValueError(f"Non-ok profiling row in manifest: {row}")

            manifest_output = Path(row["output_dir"])
            output_dir = profile_root.joinpath(*manifest_output.parts[-3:])

            relative_input = row.get("relative_input") or None
            input_fallback = sequence_root / relative_input if relative_input else Path(row["input_file"])
            input_file = resolve_manifest_path(row["input_file"], input_fallback)

            by_output_dir[output_dir] = {
                "input_file": input_file,
                "relative_input": relative_input,
                "n_rows": int(row["valid_sequences"]),
                "total_lines": int(row["total_lines"]),
                "invalid_length": int(row["invalid_length"]),
                "invalid_alphabet": int(row["invalid_alphabet"]),
                "blank_lines": int(row["blank_lines"]),
            }
    if not by_output_dir:
        raise ValueError(f"No successful tri rows found in {manifest_path}")
    return by_output_dir


def discover_groups(
    profile_root: Path,
    manifest_info: dict[Path, dict[str, Any]],
    parameters: list[str],
    profile_length: int,
    limit_groups: int = 0,
) -> list[ProfileGroup]:
    groups: list[ProfileGroup] = []
    for output_dir, info in sorted(manifest_info.items(), key=lambda item: str(item[0])):
        if limit_groups > 0 and len(groups) >= limit_groups:
            break
        if not output_dir.exists():
            raise FileNotFoundError(f"Profile output directory not found: {output_dir}")
        kingdom, organism, element = output_dir.parts[-3:]
        files: dict[str, ProfileFile] = {}
        element_labels: list[str] = []
        for csv_file in sorted(output_dir.glob("*_norm_*.csv")):
            parameter, element_label = parse_profile_filename(csv_file, organism)
            if parameter not in parameters:
                continue
            files[parameter] = ProfileFile(
                parameter=parameter,
                file_path=csv_file.resolve(),
                n_positions=profile_length,
            )
            element_labels.append(element_label)

        missing = [p for p in parameters if p not in files]
        if missing:
            raise FileNotFoundError(f"{output_dir} is missing profile files for: {missing}")
        element_label = sorted(set(element_labels))[0] if element_labels else element
        groups.append(
            ProfileGroup(
                kingdom=kingdom,
                organism=organism,
                element=element,
                element_label=element_label,
                output_dir=output_dir,
                input_file=info.get("input_file"),
                relative_input=info.get("relative_input"),
                n_rows=int(info["n_rows"]),
                files=files,
            )
        )

    groups.sort(key=lambda g: (g.kingdom, g.organism, g.element))
    if not groups:
        raise ValueError(f"No profile groups discovered under {profile_root}")
    return groups


def make_encoder(values: list[str]) -> dict[str, int]:
    return {value: i for i, value in enumerate(sorted(set(values)))}


def load_profile_csv(path: Path, expected_rows: int, n_positions: int) -> np.ndarray:
    df = pd.read_csv(path, index_col=0, dtype=np.float32)
    arr = df.to_numpy(dtype=np.float32, copy=False)
    if arr.shape[0] != expected_rows:
        raise ValueError(f"Row mismatch in {path}: expected {expected_rows}, got {arr.shape[0]}")
    if arr.shape[1] < n_positions:
        pad = np.zeros((arr.shape[0], n_positions - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    elif arr.shape[1] > n_positions:
        arr = arr[:, :n_positions]
    return np.asarray(arr, dtype=np.float32)


def append_arrow_rows(
    writer: pq.ParquetWriter,
    rows: dict[str, list],
    schema: pa.Schema,
) -> None:
    if not rows:
        return
    first_key = next(iter(rows))
    if not rows[first_key]:
        return
    writer.write_table(pa.Table.from_pydict(rows, schema=schema))
    for values in rows.values():
        values.clear()


def write_dataset_meta_group(
    writer: pq.ParquetWriter,
    group: ProfileGroup,
    start_row: int,
    chunk_size: int,
) -> None:
    rows = {name: [] for name in DATASET_META_SCHEMA.names}
    end_row = start_row + group.n_rows
    for chunk_start in range(start_row, end_row, chunk_size):
        chunk_end = min(chunk_start + chunk_size, end_row)
        size = chunk_end - chunk_start
        local_start = chunk_start - start_row
        rows["global_row"].extend(range(chunk_start, chunk_end))
        rows["kingdom"].extend([group.kingdom] * size)
        rows["organism"].extend([group.organism] * size)
        rows["element"].extend([group.element] * size)
        rows["element_label"].extend([group.element_label] * size)
        rows["local_seq_idx"].extend(range(local_start, local_start + size))
        append_arrow_rows(writer, rows, DATASET_META_SCHEMA)


def valid_sequence_offsets(
    input_file: Path,
    sequence_length: int,
) -> tuple[list[tuple[int, int, int, int]], dict[str, int]]:
    offsets: list[tuple[int, int, int, int]] = []
    stats = {
        "total_lines": 0,
        "valid": 0,
        "invalid_length": 0,
        "invalid_alphabet": 0,
        "blank": 0,
    }
    byte_start = 0
    with input_file.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            byte_end = byte_start + len(line)
            sequence = line.rstrip(b"\r\n")
            stats["total_lines"] += 1
            if not sequence:
                stats["blank"] += 1
            else:
                seq_upper = sequence.upper()
                if sequence_length > 0 and len(seq_upper) != sequence_length:
                    stats["invalid_length"] += 1
                elif any(base not in DNA_BYTES for base in seq_upper):
                    stats["invalid_alphabet"] += 1
                else:
                    offsets.append((line_number, byte_start, byte_end, len(sequence)))
                    stats["valid"] += 1
            byte_start = byte_end
    return offsets, stats


def write_sequence_manifest_group(
    writer: pq.ParquetWriter,
    group: ProfileGroup,
    start_row: int,
    sequence_length: int,
    chunk_size: int,
) -> dict[str, int]:
    if group.input_file is None:
        raise ValueError(f"No input sequence file recorded for {group.output_dir}")
    if not group.input_file.exists():
        raise FileNotFoundError(f"Sequence input file not found: {group.input_file}")

    offsets, stats = valid_sequence_offsets(group.input_file, sequence_length)
    if len(offsets) != group.n_rows:
        raise ValueError(
            f"Valid sequence count mismatch for {group.input_file}: "
            f"manifest/profile expects {group.n_rows}, sequence scan found {len(offsets)}"
        )

    rows = {name: [] for name in SEQUENCE_SCHEMA.names}
    seq_path = str(group.input_file.resolve())
    external_location = group.relative_input or seq_path
    for local_idx, (line_number, byte_start, byte_end, seq_len) in enumerate(offsets):
        rows["global_row"].append(start_row + local_idx)
        rows["kingdom"].append(group.kingdom)
        rows["organism"].append(group.organism)
        rows["element"].append(group.element)
        rows["element_label"].append(group.element_label)
        rows["local_seq_idx"].append(local_idx)
        rows["seq_path"].append(seq_path)
        rows["seq_line_number"].append(line_number)
        rows["seq_byte_start"].append(byte_start)
        rows["seq_byte_end"].append(byte_end)
        rows["seq_length"].append(seq_len)
        rows["seq_start"].append(None)
        rows["seq_end"].append(None)
        rows["strand"].append(None)
        rows["external_location"].append(external_location)
        if len(rows["global_row"]) >= chunk_size:
            append_arrow_rows(writer, rows, SEQUENCE_SCHEMA)
    append_arrow_rows(writer, rows, SEQUENCE_SCHEMA)
    return stats


def write_metadata_tables(
    out_dir: Path,
    groups: list[ProfileGroup],
    parameters: list[str],
    param_n_positions: dict[str, int],
) -> None:
    records = []
    sample_id = 0
    for group in groups:
        for parameter in parameters:
            profile_file = group.files[parameter]
            records.append(
                {
                    "sample_id": sample_id,
                    "kingdom": group.kingdom,
                    "organism": group.organism,
                    "element": group.element,
                    "element_label": group.element_label,
                    "parameter": parameter,
                    "file_path": str(profile_file.file_path),
                    "n_rows": int(group.n_rows),
                    "n_positions": int(profile_file.n_positions),
                }
            )
            sample_id += 1
    metadata = pd.DataFrame(records)
    metadata.to_parquet(out_dir / "metadata.parquet", index=False)
    metadata.to_csv(out_dir / "metadata.csv", index=False)

    summary = {
        "n_profile_files": int(len(metadata)),
        "n_groups": int(len(groups)),
        "parameters": parameters,
        "param_n_positions": param_n_positions,
        "total_sequences": int(sum(g.n_rows for g in groups)),
        "groups": [
            {
                "kingdom": g.kingdom,
                "organism": g.organism,
                "element": g.element,
                "element_label": g.element_label,
                "n_rows": int(g.n_rows),
                "output_dir": str(g.output_dir),
                "input_file": str(g.input_file) if g.input_file is not None else None,
            }
            for g in groups
        ],
    }
    with (out_dir / "profile_metadata_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)


def ensure_output_paths(out_dir: Path, overwrite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        "X_raw.npy",
        "Y_element.npy",
        "Y_organism.npy",
        "Y_kingdom.npy",
        "train_idx.npy",
        "val_idx.npy",
        "test_idx.npy",
        "dataset_meta.parquet",
        "metadata.parquet",
        "metadata.csv",
        "encoders.json",
        "build_summary.json",
    ]
    existing = [out_dir / name for name in outputs if (out_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output files already exist. Use --overwrite to replace: "
            + ", ".join(str(p) for p in existing[:8])
        )
    if overwrite:
        for path in existing:
            path.unlink()


def build_dataset(
    profile_root: Path,
    sequence_root: Path,
    profile_manifest: Path,
    out_dir: Path,
    sequence_manifest_out: Path | None,
    split_ratios: tuple[float, float, float],
    seed: int,
    sequence_length: int,
    write_sequence_manifest: bool,
    chunk_size: int,
    overwrite: bool,
    limit_groups: int,
    dry_run: bool,
    profile_length: int,
    resume: bool = False,
) -> None:
    # Resume: keep an already-written X_raw.npy/Y_*.npy and continue from the last
    # checkpointed group (deterministic group order makes this safe). Rebuild only the
    # cheap deterministic outputs (parquets, metadata, splits, sequence manifest).
    checkpoint_path = out_dir / "_build_checkpoint.json"
    resume_active = bool(
        resume and not dry_run
        and (out_dir / "X_raw.npy").exists() and checkpoint_path.exists()
    )
    if not dry_run:
        if resume_active:
            for name in ["train_idx.npy", "val_idx.npy", "test_idx.npy",
                         "dataset_meta.parquet", "metadata.parquet", "metadata.csv",
                         "encoders.json", "build_summary.json"]:
                p = out_dir / name
                if p.exists():
                    p.unlink()
            if sequence_manifest_out is not None and sequence_manifest_out.exists():
                sequence_manifest_out.unlink()
        else:
            ensure_output_paths(out_dir, overwrite=overwrite)
            if sequence_manifest_out is not None and sequence_manifest_out.exists() and not overwrite:
                raise FileExistsError(f"Sequence manifest exists: {sequence_manifest_out}")
            if sequence_manifest_out is not None and sequence_manifest_out.exists() and overwrite:
                sequence_manifest_out.unlink()

    manifest_info = load_profile_manifest(profile_manifest, profile_root, sequence_root)
    groups = discover_groups(
        profile_root,
        manifest_info,
        PARAMETERS,
        profile_length=profile_length,
        limit_groups=limit_groups,
    )

    param_n_positions = {
        parameter: max(group.files[parameter].n_positions for group in groups)
        for parameter in PARAMETERS
    }
    n_positions = max(param_n_positions.values())
    total_sequences = int(sum(group.n_rows for group in groups))
    size_gb = total_sequences * len(PARAMETERS) * n_positions * 4 / 1e9

    kingdom_encoder = make_encoder([group.kingdom for group in groups])
    organism_encoder = make_encoder([group.organism for group in groups])
    element_encoder = make_encoder([group.element for group in groups])
    parameter_encoder = {parameter: i for i, parameter in enumerate(PARAMETERS)}

    encoders = {
        "kingdom": kingdom_encoder,
        "organism": organism_encoder,
        "element": element_encoder,
        "parameter": parameter_encoder,
        "n_positions": int(n_positions),
        "param_n_positions": {k: int(v) for k, v in param_n_positions.items()},
    }

    print(f"Profile groups:    {len(groups)}")
    print(f"Total sequences:   {total_sequences:,}")
    print(f"X_raw shape:       ({total_sequences:,}, {len(PARAMETERS)}, {n_positions})")
    print(f"Estimated X size:  {size_gb:.2f} GB")
    print(f"Output directory:  {out_dir}")
    if sequence_manifest_out is not None and write_sequence_manifest:
        print(f"Sequence manifest: {sequence_manifest_out}")

    if dry_run:
        print("\nDry run only. No dataset files were written.")
        print("Discovered groups:")
        for group in groups:
            print(
                f"  {group.kingdom}/{group.organism}/{group.element}: "
                f"{group.n_rows:,} rows"
            )
        return

    write_metadata_tables(out_dir, groups, PARAMETERS, param_n_positions)
    with (out_dir / "encoders.json").open("w") as fh:
        json.dump(encoders, fh, indent=2)

    resume_until = 0
    if resume_active:
        with checkpoint_path.open() as fh:
            resume_until = int(json.load(fh).get("completed_groups", 0))
        print(f"RESUME: {resume_until} groups already in X_raw; continuing from group {resume_until + 1}")
    mmap_mode = "r+" if resume_active else "w+"

    x_raw = np.lib.format.open_memmap(
        out_dir / "X_raw.npy",
        mode=mmap_mode,
        dtype=np.float32,
        shape=(total_sequences, len(PARAMETERS), n_positions),
    )
    y_element = np.lib.format.open_memmap(
        out_dir / "Y_element.npy",
        mode=mmap_mode,
        dtype=np.int32,
        shape=(total_sequences,),
    )
    y_organism = np.lib.format.open_memmap(
        out_dir / "Y_organism.npy",
        mode=mmap_mode,
        dtype=np.int32,
        shape=(total_sequences,),
    )
    y_kingdom = np.lib.format.open_memmap(
        out_dir / "Y_kingdom.npy",
        mode=mmap_mode,
        dtype=np.int32,
        shape=(total_sequences,),
    )

    organism_ranges: dict[str, list[tuple[int, int]]] = {org: [] for org in organism_encoder}
    sequence_stats: dict[str, dict[str, int]] = {}

    dataset_meta_writer = pq.ParquetWriter(
        out_dir / "dataset_meta.parquet",
        DATASET_META_SCHEMA,
        compression="zstd",
    )
    sequence_writer = None
    if write_sequence_manifest:
        if sequence_manifest_out is None:
            raise ValueError("--sequence-manifest-out is required when writing sequence manifest")
        sequence_manifest_out.parent.mkdir(parents=True, exist_ok=True)
        sequence_writer = pq.ParquetWriter(sequence_manifest_out, SEQUENCE_SCHEMA, compression="zstd")

    global_row = 0
    try:
        for group_id, group in enumerate(groups, start=1):
            start = global_row
            end = start + group.n_rows
            skip = group_id <= resume_until
            print(
                f"[{group_id:03d}/{len(groups):03d}] "
                f"{group.kingdom}/{group.organism}/{group.element} "
                f"rows={group.n_rows:,}" + ("  (resume: already in X_raw)" if skip else "")
            )

            if not skip:
                for parameter in PARAMETERS:
                    pidx = parameter_encoder[parameter]
                    profile_file = group.files[parameter]
                    profiles = load_profile_csv(profile_file.file_path, group.n_rows, n_positions)
                    x_raw[start:end, pidx, :] = profiles
                    del profiles

                y_element[start:end] = element_encoder[group.element]
                y_organism[start:end] = organism_encoder[group.organism]
                y_kingdom[start:end] = kingdom_encoder[group.kingdom]

            organism_ranges[group.organism].append((start, end))

            write_dataset_meta_group(dataset_meta_writer, group, start, chunk_size)
            if sequence_writer is not None:
                stats = write_sequence_manifest_group(
                    sequence_writer,
                    group,
                    start,
                    sequence_length=sequence_length,
                    chunk_size=chunk_size,
                )
                sequence_stats[f"{group.kingdom}/{group.organism}/{group.element}"] = stats

            global_row = end
            if not skip:
                x_raw.flush()
                y_element.flush()
                y_organism.flush()
                y_kingdom.flush()
                with checkpoint_path.open("w") as fh:
                    json.dump({"completed_groups": group_id, "global_row": global_row}, fh)
    finally:
        dataset_meta_writer.close()
        if sequence_writer is not None:
            sequence_writer.close()

    del x_raw, y_element, y_organism, y_kingdom

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    train_ratio, val_ratio, test_ratio = split_ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0: {split_ratios}")

    for organism, ranges in sorted(organism_ranges.items()):
        indices = np.concatenate([np.arange(start, end, dtype=np.int64) for start, end in ranges])
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_parts.append(indices[:n_train])
        val_parts.append(indices[n_train:n_train + n_val])
        test_parts.append(indices[n_train + n_val:])
        print(
            f"Split {organism:<12} total={n:,} "
            f"train={n_train:,} val={n_val:,} test={n - n_train - n_val:,}"
        )

    train_idx = np.concatenate(train_parts).astype(np.int64)
    val_idx = np.concatenate(val_parts).astype(np.int64)
    test_idx = np.concatenate(test_parts).astype(np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    np.save(out_dir / "train_idx.npy", train_idx)
    np.save(out_dir / "val_idx.npy", val_idx)
    np.save(out_dir / "test_idx.npy", test_idx)

    final_summary = {
        "dataset": str(out_dir),
        "sequence_manifest": str(sequence_manifest_out) if sequence_manifest_out is not None else None,
        "total_sequences": int(total_sequences),
        "x_shape": [int(total_sequences), len(PARAMETERS), int(n_positions)],
        "x_size_gb": float(size_gb),
        "split_counts": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "encoders": encoders,
        "sequence_stats": sequence_stats,
    }
    with (out_dir / "build_summary.json").open("w") as fh:
        json.dump(final_summary, fh, indent=2)

    print("\nDone.")
    print(f"Dataset:           {out_dir}")
    print(f"X_raw.npy:         {out_dir / 'X_raw.npy'}")
    print(f"Encoders:          {out_dir / 'encoders.json'}")
    print(f"Splits:            train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")
    if sequence_manifest_out is not None:
        print(f"Sequence manifest: {sequence_manifest_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, default=Path.cwd())
    parser.add_argument("--sequence-root", type=Path, default=Path("Biophysical_Profiling_Datasets"))
    parser.add_argument(
        "--profile-manifest",
        type=Path,
        default=Path("Biophysical_Profiling_Datasets/reprofiled_profiles/profile_manifest.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("dataset_reprofiled"))
    parser.add_argument(
        "--sequence-manifest-out",
        type=Path,
        default=Path("sequence_manifest_reprofiled.parquet"),
    )
    parser.add_argument("--no-sequence-manifest", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=501)
    parser.add_argument("--split-ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--profile-length", type=int, default=475)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Continue a crashed build: keep existing X_raw/Y, skip groups "
                             "recorded in _build_checkpoint.json, rebuild the cheap outputs.")
    parser.add_argument("--limit-groups", type=int, default=0, help="Debug option; 0 means all groups.")
    parser.add_argument("--dry-run", action="store_true", help="Discover inputs and report size without writing files.")
    args = parser.parse_args()

    build_dataset(
        profile_root=args.profile_root.resolve(),
        sequence_root=args.sequence_root.resolve(),
        profile_manifest=args.profile_manifest.resolve(),
        out_dir=args.out.resolve(),
        sequence_manifest_out=args.sequence_manifest_out.resolve() if not args.no_sequence_manifest else None,
        split_ratios=tuple(float(v) for v in args.split_ratios),
        seed=int(args.seed),
        sequence_length=int(args.sequence_length),
        write_sequence_manifest=not args.no_sequence_manifest,
        chunk_size=int(args.chunk_size),
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
        limit_groups=int(args.limit_groups),
        dry_run=bool(args.dry_run),
        profile_length=int(args.profile_length),
    )


if __name__ == "__main__":
    main()
