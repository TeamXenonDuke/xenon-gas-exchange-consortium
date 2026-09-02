"""Create one ordered cohort PDF from subject-level combined reports."""

import logging
import re
from pathlib import Path
from typing import List

from absl import app, flags
from PyPDF2 import PdfMerger

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_root",
    "data",
    "Directory that contains one subdirectory per subject.",
)
flags.DEFINE_string("output", None, "Optional path for the final cohort-level PDF.")


def natural_sort_key(path: Path) -> list[object]:
    """Sort subject IDs naturally, so 006-2 comes before 006-10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def get_pdfs(data_root: Path) -> List[Path]:
    """Return one combined PDF per subject, ordered by subject directory name."""
    pdfs = []
    for subject_dir in sorted(
        (path for path in data_root.iterdir() if path.is_dir()),
        key=natural_sort_key,
    ):
        pdf = subject_dir / f"{subject_dir.name}_combined_report.pdf"
        if pdf.is_file():
            pdfs.append(pdf)
        else:
            logging.warning("Combined report not found; skipping %s", pdf)
    return pdfs


def main(argv):
    """Merge each subject's combined report into one cohort PDF."""
    del argv
    data_root = Path(FLAGS.data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    output_path = (
        Path(FLAGS.output).resolve()
        if FLAGS.output
        else data_root / "combined_subject_reports.pdf"
    )
    pdfs = get_pdfs(data_root)
    if not pdfs:
        raise ValueError(
            f"No subject combined reports found under {data_root}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merger = PdfMerger()
    try:
        for pdf in pdfs:
            merger.append(str(pdf))
        merger.write(str(output_path))
    finally:
        merger.close()


if __name__ == "__main__":
    app.run(main)
