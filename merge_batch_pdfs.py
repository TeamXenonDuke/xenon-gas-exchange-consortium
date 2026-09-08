"""Merge one ``*_combined_report.pdf`` from each batch subfolder.

Example (WSL):
    python merge_batch_pdfs.py

The default input directory is the WSL path used by the batch pipeline.  A
different directory or output path can be supplied with command-line options.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from pypdf import PdfWriter


DEFAULT_BATCH_DIR = Path(
    "/mnt/d/xenon-gas-exchange-consortium/data/batch"
)
DEFAULT_OUTPUT = DEFAULT_BATCH_DIR / "all_combined_reports.pdf"


def natural_sort_key(path: Path) -> list[object]:
    """Sort names containing numbers in human/numeric order."""
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    ]


def find_reports(batch_dir: Path) -> list[Path]:
    """Find one combined report in each direct child folder.

    The preferred filename is ``<folder-name>_combined_report.pdf``.  If a
    folder uses a slightly different prefix, its first ``*_combined_report.pdf``
    file is accepted as a fallback.
    """
    reports: list[Path] = []

    child_dirs = sorted(
        (path for path in batch_dir.iterdir() if path.is_dir()),
        key=natural_sort_key,
    )

    for child_dir in child_dirs:
        expected = child_dir / f"{child_dir.name}_combined_report.pdf"
        if expected.is_file():
            reports.append(expected)
            continue

        candidates = sorted(
            child_dir.glob("*_combined_report.pdf"),
            key=natural_sort_key,
        )
        if candidates:
            reports.append(candidates[0])
            print(
                f"Warning: using {candidates[0].name} in {child_dir.name}",
                file=sys.stderr,
            )
        else:
            print(
                f"Warning: no *_combined_report.pdf in {child_dir}",
                file=sys.stderr,
            )

    return reports


def merge_reports(reports: list[Path], output_path: Path) -> None:
    """Append reports in the supplied order and write one PDF."""
    writer = PdfWriter()
    for report in reports:
        print(f"Adding: {report}")
        writer.append(str(report))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_file:
        writer.write(output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge *_combined_report.pdf files from batch subfolders "
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
        default=DEFAULT_OUTPUT,
        help=f"Merged PDF path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batch_dir = args.input_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not batch_dir.is_dir():
        print(f"Error: input directory does not exist: {batch_dir}", file=sys.stderr)
        return 1

    reports = find_reports(batch_dir)
    if not reports:
        print(f"Error: no combined reports found in {batch_dir}", file=sys.stderr)
        return 1

    merge_reports(reports, output_path)
    print(f"Merged {len(reports)} PDF(s) into: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
