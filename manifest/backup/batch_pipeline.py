"""Build and run a subject batch for the xenon gas exchange pipeline.

This first version reads one CSV manifest row per subject, builds the same
Config object used by main.py, saves a JSON configuration snapshot, and
optionally runs subjects serially.

Serial execution is intentional: the current project shares one tmp/
directory, so parallel subjects could overwrite intermediate files.
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# Existing pipeline modules use project-relative paths for assets and tmp/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import base_config  # noqa: E402
from utils import constants  # noqa: E402


STATUS_FIELDS = [
    "subject_id",
    "status",
    "process_mode",
    "data_dir",
    "input_type",
    "config_path",
    "output_dir",
    "started_at",
    "finished_at",
    "message",
]


def parse_args() -> argparse.Namespace:
    """Parse arguments for dry-run validation or pipeline execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_ROOT / "manifest_example.csv",
        help="CSV manifest containing one subject per row.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root containing subject folders when data_dir is blank.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run validated subjects serially. Default behavior is dry-run.",
    )
    return parser.parse_args()


def is_true(value: str) -> bool:
    """Parse simple CSV boolean values."""
    return value.strip().lower() in {"1", "true", "yes"}


def optional_float(value: str) -> Optional[float]:
    """Return None for blank CSV cells and float for populated cells."""
    return float(value) if value.strip() else None


def get_data_dir(row: dict[str, str], data_root: Path) -> Path:
    """Resolve explicit data_dir or default to data/<subject_id>."""
    data_dir = row.get("data_dir", "").strip()
    return Path(data_dir) if data_dir else data_root / row["subject_id"]


def find_input_type(data_dir: Path, process_mode: str) -> tuple[str, str]:
    """Perform a lightweight input-data preflight check.

    Reconstruction mode accepts Twix .dat or ISMRMRD .h5 files. Read-in mode
    requires a previously exported .mat file. Detailed sequence classification
    remains the responsibility of the project's existing IO utilities.
    """
    if not data_dir.is_dir():
        return "", "subject data directory does not exist"

    suffixes = {
        path.suffix.lower()
        for path in data_dir.rglob("*")
        if path.is_file()
    }

    if process_mode == "readin":
        if ".mat" in suffixes:
            return "mat", ""
        return "", "readin mode requires a .mat file"

    if ".dat" in suffixes:
        return "twix", ""

    if ".h5" in suffixes:
        return "mrd", ""

    return "", "reconstruction mode requires .dat or .h5 input"


def build_config(
    row: dict[str, str],
    data_dir: Path,
) -> base_config.Config:
    """Build a project-native Config object from one manifest row.

    Blank manifest values retain the defaults defined in base_config.Config.
    This avoids dynamic imports of one Python config file per subject while
    preserving the existing pipeline's configuration model.
    """
    config = base_config.Config()

    # Required subject-level settings.
    config.subject_id = row["subject_id"]
    config.data_dir = str(data_dir)
    config.output_folder = row.get("output_folder", "").strip() or "gx_batch"
    config.trachea_plus_lung_mask_output_dir = str(data_dir)

    # Optional RBC:M override. Blank leaves the existing spectroscopy workflow.
    rbc_m_ratio = optional_float(row.get("rbc_m_ratio", ""))
    if rbc_m_ratio is not None:
        config.rbc_m_ratio = rbc_m_ratio

    # Supplying hemoglobin activates the project-standard Hb correction.
    hb = optional_float(row.get("hb", ""))
    if hb is not None:
        config.hb = hb
        config.hb_correction_key = (
            constants.HbCorrectionKey.RBC_AND_MEMBRANE.value
        )

    # Supplying lung volume activates the project-standard volume correction.
    lung_volume = optional_float(row.get("corrected_lung_volume", ""))
    if lung_volume is not None:
        config.corrected_lung_volume = lung_volume
        config.vol_correction_key = (
            constants.VolCorrectionKey.RBC_AND_MEMBRANE.value
        )

    # Reconstruction choices. Blank values retain base-config defaults.
    config.recon.recon_proton = is_true(row.get("recon_proton", "true"))
    config.recon.recon_key = (
        row.get("recon_key", "").strip() or config.recon.recon_key
    )
    config.recon.scan_type = row.get("scan_type", "").strip()

    # Gradient delays are optional per-subject overrides.
    for field in ("del_x", "del_y", "del_z"):
        value = optional_float(row.get(field, ""))
        if value is not None:
            setattr(config.recon, field, value)

    config.osc_recon.oscillation_analysis = is_true(
        row.get("oscillation_analysis", "false")
    )

    # Select exactly one existing pipeline entry point.
    config.processes.gx_mapping_recon = row["process_mode"] == "recon"
    config.processes.gx_mapping_readin = row["process_mode"] == "readin"

    return config


def config_snapshot(
    config: base_config.Config,
    input_type: str,
) -> dict[str, Any]:
    """Return the relevant generated settings as a portable JSON record."""
    return {
        "subject_id": config.subject_id,
        "data_dir": config.data_dir,
        "output_folder": config.output_folder,
        "input_type": input_type,
        "rbc_m_ratio": config.rbc_m_ratio,
        "hb": config.hb,
        "hb_correction_key": config.hb_correction_key,
        "corrected_lung_volume": config.corrected_lung_volume,
        "vol_correction_key": config.vol_correction_key,
        "processes": {
            "gx_mapping_recon": config.processes.gx_mapping_recon,
            "gx_mapping_readin": config.processes.gx_mapping_readin,
        },
        "recon": {
            "recon_key": config.recon.recon_key,
            "recon_proton": config.recon.recon_proton,
            "scan_type": config.recon.scan_type,
            "del_x": config.recon.del_x,
            "del_y": config.recon.del_y,
            "del_z": config.recon.del_z,
        },
        "osc_recon": {
            "oscillation_analysis": config.osc_recon.oscillation_analysis,
        },
    }


def write_json_config(
    config: base_config.Config,
    input_type: str,
) -> Path:
    """Save a frozen generated configuration before processing."""
    config_dir = MANIFEST_ROOT / "generated_configs"
    config_dir.mkdir(exist_ok=True)

    config_path = config_dir / f"{config.subject_id}_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config_snapshot(config, input_type), file, indent=2)

    return config_path


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Read manifest rows after checking its minimum required schema."""
    with manifest_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required_fields = {"subject_id", "process_mode"}

        if not reader.fieldnames:
            raise ValueError("manifest is missing a header row")

        if not required_fields.issubset(reader.fieldnames):
            raise ValueError(
                "manifest must contain subject_id and process_mode columns"
            )

        return list(reader)


def make_status(
    row: dict[str, str],
    data_dir: Path,
    **updates: str,
) -> dict[str, str]:
    """Create one stable-format record for batch_status.csv."""
    status = {field: "" for field in STATUS_FIELDS}
    status.update(
        subject_id=row.get("subject_id", ""),
        process_mode=row.get("process_mode", ""),
        data_dir=str(data_dir),
    )
    status.update(updates)
    return status


def write_status(status_rows: list[dict[str, str]]) -> None:
    """Persist status after each subject for monitoring and recovery."""
    status_path = MANIFEST_ROOT / "batch_status.csv"

    with status_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        writer.writerows(status_rows)


def process_subject(
    row: dict[str, str],
    data_root: Path,
    run: bool,
) -> dict[str, str]:
    """Validate, snapshot, and optionally run one subject."""
    subject_id = row.get("subject_id", "").strip()
    process_mode = row.get("process_mode", "").strip().lower()
    data_dir = get_data_dir(row, data_root)

    if not subject_id:
        return make_status(
            row,
            data_dir,
            status="needs_review",
            message="subject_id is blank",
        )

    if process_mode not in {"recon", "readin"}:
        return make_status(
            row,
            data_dir,
            status="needs_review",
            message="process_mode must be recon or readin",
        )

    input_type, message = find_input_type(data_dir, process_mode)
    if message:
        return make_status(
            row,
            data_dir,
            status="needs_review",
            message=message,
        )

    try:
        normalized_row = {**row, "process_mode": process_mode}
        config = build_config(normalized_row, data_dir)
        config_path = write_json_config(config, input_type)
    except (TypeError, ValueError) as error:
        return make_status(
            row,
            data_dir,
            status="needs_review",
            message=str(error),
        )

    status = make_status(
        row,
        data_dir,
        status="planned",
        input_type=input_type,
        config_path=str(config_path),
        output_dir=str(data_dir / config.output_folder),
    )

    # Dry-run mode stops after validation and config generation.
    if not run:
        return status

    status["status"] = "running"
    status["started_at"] = datetime.now().isoformat(timespec="seconds")

    try:
        # Existing project code expects to run from the repository root.
        os.chdir(PROJECT_ROOT)

        if process_mode == "recon":
            # Import the full pipeline only for an actual processing run.
            # This keeps dry-run validation independent of report and model
            # dependencies that are not needed to inspect the manifest.
            from main import gx_mapping_reconstruction

            gx_mapping_reconstruction(config)
        else:
            from main import gx_mapping_readin

            gx_mapping_readin(config)

        status["status"] = "completed"

    except Exception as error:
        # Continue the remainder of the batch after a single-subject failure.
        logging.exception("Subject %s failed", subject_id)
        status["status"] = "failed"
        status["message"] = f"{type(error).__name__}: {error}"

    finally:
        status["finished_at"] = datetime.now().isoformat(timespec="seconds")

    return status


def main() -> None:
    """Execute a batch and write an auditable status report."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    manifest_path = args.manifest.resolve()
    data_root = args.data_root.resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    status_rows = []

    for row in read_manifest(manifest_path):
        status_rows.append(process_subject(row, data_root, args.run))
        write_status(status_rows)

    summary = {
        state: sum(row["status"] == state for row in status_rows)
        for state in ("completed", "planned", "failed", "needs_review")
    }

    logging.info("Batch complete: %s", summary)


if __name__ == "__main__":
    main()
