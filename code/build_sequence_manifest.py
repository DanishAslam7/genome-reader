#!/usr/bin/env python3
"""Build a row-level sequence manifest from total_* sequence files.

The source files in Biophysical_Profiling_Datasets store one DNA sequence per
line. This script records stable row keys and byte offsets so training code can
join profile tensors to sequence data without copying sequence bodies locally.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


ELEMENT_DIR_TO_LABEL = {
    "gene_start": "gs",
    "gene_end": "ge",
    "exon_start": "es",
    "exon_end": "ee",
    "cds": "cds",
    "promoters": "prom",
    "start_codon": "stac",
    "stop_codon": "stoc",
    "5utr_start": "5utrs",
    "5utr_end": "5utre",
    "3utr_start": "3utrs",
    "3utr_end": "3utre",
    "enhancer_start": "ens",
    "enhancer_end": "ene",
}

ELEMENT_DIR_TO_PREFIXES = {
    "enhancer_start": ["total_start_enhancers_"],
    "enhancer_end": ["total_end_enhancers_"],
}

ORGANISM_TO_KINGDOM = {
    "human": "Animal",
    "mouse": "Animal",
    "rat": "Animal",
    "fly": "Animal",
    "worm": "Animal",
    "animal": "Animal",
    "arabidopsis": "Plant",
    "zeamays": "Plant",
    "plant": "Plant",
    "scerevisiae": "Fungi",
    "klactis": "Fungi",
    "ylipolytica": "Fungi",
    "fungi": "Fungi",
    "pfalciparum": "Protist",
    "protist": "Protist",
}

SCHEMA = pa.schema(
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


def parse_total_file(path: Path) -> tuple[str, str, str]:
    element = path.parent.name
    if element not in ELEMENT_DIR_TO_LABEL:
        raise ValueError(f"Unsupported element directory: {path.parent}")

    element_label = ELEMENT_DIR_TO_LABEL[element]
    expected_prefixes = ELEMENT_DIR_TO_PREFIXES.get(element, []) + [
        f"total_{element_label}_",
        f"total_{element}_",
    ]

    expected_prefix = next(
        (prefix for prefix in expected_prefixes if path.name.startswith(prefix)),
        None,
    )
    if expected_prefix is None:
        raise ValueError(f"Cannot parse organism from {path}")

    organism = path.name.removeprefix(expected_prefix).lower()
    kingdom = ORGANISM_TO_KINGDOM.get(organism, "Unknown")
    return kingdom, organism, element


def find_total_files(sequence_root: Path) -> list[Path]:
    return sorted(p for p in sequence_root.rglob("total_*") if p.is_file())


def flush_rows(writer: pq.ParquetWriter, rows: dict[str, list]) -> None:
    if not rows["global_row"]:
        return
    table = pa.Table.from_pydict(rows, schema=SCHEMA)
    writer.write_table(table)
    for values in rows.values():
        values.clear()


def add_row(rows: dict[str, list], **values: object) -> None:
    for column in SCHEMA.names:
        rows[column].append(values[column])


def build_manifest(
    sequence_root: Path,
    output: Path,
    external_root: str | None,
    chunk_size: int,
) -> tuple[int, int]:
    files = find_total_files(sequence_root)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: dict[str, list] = {name: [] for name in SCHEMA.names}
    global_row = 0

    with pq.ParquetWriter(output, SCHEMA, compression="zstd") as writer:
        for seq_file in files:
            kingdom, organism, element = parse_total_file(seq_file)
            element_label = ELEMENT_DIR_TO_LABEL[element]
            rel_path = seq_file.relative_to(sequence_root.parent).as_posix()
            external_location = (
                f"{external_root.rstrip('/')}/{rel_path}" if external_root else rel_path
            )

            byte_start = 0
            local_seq_idx = 0
            with seq_file.open("rb") as handle:
                for line in handle:
                    byte_end = byte_start + len(line)
                    sequence = line.rstrip(b"\r\n")
                    if sequence:
                        add_row(
                            rows,
                            global_row=global_row,
                            kingdom=kingdom,
                            organism=organism,
                            element=element,
                            element_label=element_label,
                            local_seq_idx=local_seq_idx,
                            seq_path=rel_path,
                            seq_line_number=local_seq_idx + 1,
                            seq_byte_start=byte_start,
                            seq_byte_end=byte_end,
                            seq_length=len(sequence),
                            seq_start=None,
                            seq_end=None,
                            strand=None,
                            external_location=external_location,
                        )
                        global_row += 1
                        local_seq_idx += 1

                        if len(rows["global_row"]) >= chunk_size:
                            flush_rows(writer, rows)

                    byte_start = byte_end

        flush_rows(writer, rows)

    return len(files), global_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-root",
        type=Path,
        default=Path("Biophysical_Profiling_Datasets"),
        help="Directory containing element subdirectories with total_* files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sequence_manifest.parquet"),
        help="Output Parquet manifest path.",
    )
    parser.add_argument(
        "--external-root",
        default=None,
        help="Optional external root/URI prepended to each relative sequence path.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Rows per Parquet write batch.",
    )
    args = parser.parse_args()

    file_count, row_count = build_manifest(
        sequence_root=args.sequence_root,
        output=args.output,
        external_root=args.external_root,
        chunk_size=args.chunk_size,
    )
    print(f"Wrote {row_count:,} sequence rows from {file_count} files to {args.output}")


if __name__ == "__main__":
    main()
