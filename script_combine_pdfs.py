"""Script to combine PDFs in a folder into a single PDF."""
import glob
import os
from typing import List

from absl import app, flags
from PyPDF2 import PdfMerger

FLAGS = flags.FLAGS

flags.DEFINE_string("cohort", "all", "cohort folder name in data folder")


def sort_pdfs(pdfs: List[str]) -> List[str]:
    """Sort pdfs by patient id.

    A subject id is a tag unique to the patient. It could end in s1 or s2 to indicate
    the first scan of the session or the second scan of the session. The subject id
    may also contain letters "A-Z" to indicate which session the scan belongs to.
    The algorithm will sort the pdfs by subject id, then by scan, then by session.

    Examples:
        "000-001A_s1" will be sorted before "000-001A_s2".
        "000-001A" will be sorted before "000-001B_s1".
        "000-001" will be sorted before "000-002_s1".
    Args:
        pdfs: list of pdf paths
    Returns:
        sorted pdfs
    """
    # Get subject ids
    subject_ids = [os.path.basename(pdf).split("_")[0] for pdf in pdfs]
    # Get scan numbers
    scan_numbers = [os.path.basename(pdf).split("_")[1] for pdf in pdfs]
    # Get session numbers
    session_numbers = [os.path.basename(pdf).split("_")[2] for pdf in pdfs]
    # Sort pdfs
    return [
        pdf
        for _, pdf in sorted(
            zip(
                zip(
                    zip(subject_ids, scan_numbers, session_numbers),
                    [os.path.basename(pdf) for pdf in pdfs],
                ),
                pdfs,
            )
        )
    ]


def get_pdfs() -> List[str]:
    """Get all pdfs in the data/ folder."""
    if FLAGS.cohort == "healthy":
        pdfs = glob.glob(
            os.path.join("data", "healthy", "**/report_clinical**.pdf"), recursive=True
        )
    elif FLAGS.cohort == "cteph":
        pdfs = glob.glob(
            os.path.join("data", "cteph", "**/report_clinical**.pdf"), recursive=True
        )
    elif FLAGS.cohort == "ild":
        pdfs = glob.glob(
            os.path.join("data", "ild", "**/report_clinical**.pdf"), recursive=True
        )
    elif FLAGS.cohort == "tyvaso":
        pdfs = glob.glob(
            os.path.join("data", "tyvaso", "**/report_clinical**.pdf"), recursive=True
        )
    elif FLAGS.cohort == "jupiter":
        pdfs = glob.glob(
            os.path.join("data", "jupiter", "**/report_clinical**.pdf"), recursive=True
        )
    elif FLAGS.cohort == "all":
        pdfs = glob.glob("data/**/report_clinical.pdf", recursive=True)
    else:
        raise ValueError("Invalid cohort name")
    return sort_pdfs(pdfs)


def main(argv):
    """Combine PDFs in a folder into a single PDF.

    Combines all pdfs in the data/ folder into a single pdf. Save it to tmp folder.
    This may take a while to run.
    """
    pdfs = get_pdfs()
    merger = PdfMerger()
    for file in pdfs:
        merger.append(file)
    merger.write(os.path.join("tmp", "combined.pdf"))


if __name__ == "__main__":
    """Run the script to combine PDFs in a folder into a single PDF."""
    app.run(main)
