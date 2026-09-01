"""Combine reports from the completed subjects in the latest batch manifest."""

import csv
import logging
from pathlib import Path
from typing import List

from absl import app, flags
from PyPDF2 import PdfMerger

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "output",
    "tmp/combined.pdf",
    "Path for the final cohort-level PDF.",
)


def get_pdfs() -> List[Path]:
    """Return reports for the completed subjects in manifest batch order."""
    status_path = Path(__file__).with_name("batch_status.csv")
    if not status_path.is_file():
        raise FileNotFoundError(
            f"Batch status not found: {status_path}. Run the manifest batch first."
        )

    output_path = Path(FLAGS.output).resolve()
    pdfs = []
    with status_path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if row.get("status") != "completed":
                continue

            subject_id = row.get("subject_id", "").strip()
            subject_dir = Path(row.get("data_dir", ""))
            pdf = subject_dir / f"{subject_id}_combined_report.pdf"
            if pdf.resolve() == output_path:
                continue
            if pdf.is_file():
                pdfs.append(pdf)
            else:
                logging.warning("Combined report not found; skipping %s", pdf)
    return pdfs


def main(argv):
    """Merge each completed subject's combined report into one PDF."""
    del argv
    pdfs = get_pdfs()
    if not pdfs:
        raise ValueError(
            "No subject combined reports found. Enable combine_reports for each "
            "subject before running this script."
        )

    output_path = Path(FLAGS.output)
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
