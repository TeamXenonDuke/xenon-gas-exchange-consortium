"""Merge one ``*_stats.csv`` from each batch subject's ``gx_batch`` folder.

Example (WSL):
    python merge_batch_csvs.py

By default, subjects are read from ``data/batch`` and the merged file is
written to ``data/batch/all_stats.csv``.  Subject folders are processed in
natural/numeric order, matching ``merge_batch_pdfs.py``.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


DEFAULT_BATCH_DIR = Path("/mnt/d/xenon-gas-exchange-consortium/data/batch")
DEFAULT_OUTPUT = DEFAULT_BATCH_DIR / "all_stats.csv"


def natural_sort_key(path: Path) -> list[object]:
    """Sort names containing numbers in human/numeric order."""
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    ]


def find_stats_csvs(batch_dir: Path, output_path: Path) -> list[Path]:
    """Find one stats CSV in each direct child subject folder.

    The preferred file is ``<subject>/gx_batch/<subject>_stats.csv``.  A
    differently prefixed ``*_stats.csv`` in the same ``gx_batch`` folder is
    accepted as a fallback.  For compatibility with already-collected batch
    results, ``<subject>/<subject>_stats.csv`` is also supported.
    """
    csv_paths: list[Path] = []
    resolved_output = output_path.resolve()

    child_dirs = sorted(
        (path for path in batch_dir.iterdir() if path.is_dir()),
        key=natural_sort_key,
    )
    for child_dir in child_dirs:
        gx_dir = child_dir / "gx_batch"
        search_dir = gx_dir if gx_dir.is_dir() else child_dir
        expected = search_dir / f"{child_dir.name}_stats.csv"

        if expected.is_file() and expected.resolve() != resolved_output:
            csv_paths.append(expected)
            continue

        candidates = sorted(
            (
                path
                for path in search_dir.glob("*_stats.csv")
                if path.resolve() != resolved_output
            ),
            key=natural_sort_key,
        )
        if candidates:
            csv_paths.append(candidates[0])
            print(
                f"Warning: using {candidates[0].name} for {child_dir.name}",
                file=sys.stderr,
            )
        else:
            print(
                f"Warning: no *_stats.csv in {search_dir}",
                file=sys.stderr,
            )

    return csv_paths


def merge_csvs(csv_paths: list[Path], output_path: Path) -> int:
    """Append CSV records with one shared header and return the row count."""
    header: list[str] | None = None
    rows: list[dict[str, str]] = []

    for csv_path in csv_paths:
        print(f"Adding: {csv_path}")
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {csv_path}")
            if header is None:
                header = reader.fieldnames
            elif reader.fieldnames != header:
                raise ValueError(
                    "CSV headers do not match:\n"
                    f"  expected: {header}\n"
                    f"  found in {csv_path}: {reader.fieldnames}"
                )
            rows.extend(reader)

    if header is None:
        raise ValueError("No CSV data to merge")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge one *_stats.csv from each batch subject's gx_batch folder "
            "in natural folder order."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_BATCH_DIR,
        help=f"Batch directory (default: {DEFAULT_BATCH_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Merged CSV path (default: <input-dir>/all_stats.csv)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batch_dir = args.input_dir.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else batch_dir / DEFAULT_OUTPUT.name
    )

    if not batch_dir.is_dir():
        print(f"Error: input directory does not exist: {batch_dir}", file=sys.stderr)
        return 1

    csv_paths = find_stats_csvs(batch_dir, output_path)
    if not csv_paths:
        print(f"Error: no stats CSV files found in {batch_dir}", file=sys.stderr)
        return 1

    try:
        row_count = merge_csvs(csv_paths, output_path)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(
        f"Merged {len(csv_paths)} CSV file(s), {row_count} data row(s), "
        f"into: {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
