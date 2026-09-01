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
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# Existing pipeline modules use project-relative paths for assets and tmp/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DISCOVERED_MANIFEST_PATH = MANIFEST_ROOT / "manifest_example.csv"
DISCOVERED_MANIFEST_FIELDS = [
    "subject_id",
    "data_dir",
    "process_mode",
    "rbc_m_ratio",
    "hb",
    "corrected_lung_volume",
    "recon_proton",
    "recon_key",
    "recon_size",
    "scan_type",
    "del_x",
    "del_y",
    "del_z",
    "ramp_time",
    "oscillation_analysis",
    "key_radius_pct",
    "output_folder",
    "combine_reports",
    "vc_correction",
    "segmentation_key",
    "manual_seg_filepath",
    "registration_key",
    "manual_reg_filepath",
    "bias_key",
    "reference_data_key",
    "bag_volume",
    "vent_normalization_method",
    "n_skip_start",
    "n_skip_end",
    "traj_type",
    "traj_scaling_factor",
    "dicom_proton_dir",
    "multi_echo",
    "age",
    "sex",
    "height_cm",
    "weight_kg",
]
DEFAULT_RAMP_TIME = 90.0

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

DEFAULT_DEMOGRAPHICS = {
    "age": 50,
    "sex": "M",
    "height_cm": 170.0,
    "weight_kg": 70.0,
}


def parse_args() -> argparse.Namespace:
    """Parse arguments for dry-run validation or pipeline execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    input_source = parser.add_mutually_exclusive_group()
    input_source.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_ROOT / "manifest_example.csv",
        help="CSV manifest containing one subject per row.",
    )
    input_source.add_argument(
        "--discover",
        action="store_true",
        help=(
            "Discover subject folders under --data-root instead of reading a CSV. "
            "Raw .dat/.h5 inputs are planned as recon; folders with only .mat "
            "inputs are planned as readin."
        ),
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


def optional_int(value: str) -> Optional[int]:
    """Return None for blank CSV cells and int for populated cells."""
    return int(value) if value.strip() else None


def json_value(value: Any) -> Any:
    """Convert non-finite numeric defaults to strict-JSON null values."""
    return None if isinstance(value, float) and math.isnan(value) else value


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


def discover_ramp_time(data_dir: Path) -> float:
    """Read ramp time from a Dixon Twix header, with a documented fallback."""
    try:
        import mapvbvd

        from utils import twix_utils

        dixon_file = next(
            path for path in data_dir.rglob("*.dat") if "dixon" in path.name.lower()
        )
        ramp_time = float(twix_utils.get_ramp_time(mapvbvd.mapVBVD(str(dixon_file))))
        if math.isfinite(ramp_time) and ramp_time > 0:
            return ramp_time
    except (ImportError, OSError, StopIteration, TypeError, ValueError):
        pass

    logging.warning(
        "Could not read ramp time for %s; using fallback %.0f microseconds.",
        data_dir,
        DEFAULT_RAMP_TIME,
    )
    return DEFAULT_RAMP_TIME


def write_discovered_manifest(rows: list[dict[str, str]]) -> None:
    """Write automatically discovered subjects to the editable CSV manifest."""
    with DISCOVERED_MANIFEST_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=DISCOVERED_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def csv_setting(value: Any) -> str:
    """Convert a config value to an editable CSV cell."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, str) and value.strip().lower() in {"none", "nan"}:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def find_manual_seg_filepath(subject_dir: Path) -> Optional[Path]:
    """Locate the manual ventilation mask beside raw Twix or MRD data.

    A corrected mask is preferred across all raw-data directories before using
    an uncorrected mask. This lets a subject contain both Twix and MRD inputs
    without accidentally choosing an older uncorrected mask first.
    """
    raw_dirs = sorted(
        {
            path.parent
            for path in subject_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".dat", ".h5"}
        },
        key=lambda path: str(path).lower(),
    )
    for filename in ("mask_reg_corrected.nii", "mask_reg.nii"):
        for raw_dir in raw_dirs:
            candidate = raw_dir / filename
            if candidate.is_file():
                return candidate
    return None


def read_spectroscopy_rbcm(csv_path: Path) -> Optional[float]:
    """Read the first valid RBC:M value from a spectroscopy CSV file."""
    try:
        with csv_path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.reader(file)
            header = next(reader, None)
            if not header:
                return None

            normalized_header = [cell.strip().lower() for cell in header]
            try:
                rbcm_index = normalized_header.index("rbcm")
            except ValueError:
                if len(header) < 5:
                    logging.warning(
                        "Skipping %s: no rbcm column and fewer than five columns.",
                        csv_path,
                    )
                    return None
                rbcm_index = 4

            for row in reader:
                if rbcm_index >= len(row) or not row[rbcm_index].strip():
                    continue
                try:
                    value = float(row[rbcm_index].strip())
                except ValueError:
                    logging.warning("Skipping invalid rbcm value in %s.", csv_path)
                    continue
                if math.isfinite(value) and value > 0:
                    return value
                logging.warning("Skipping invalid rbcm value in %s.", csv_path)
    except (OSError, UnicodeError, csv.Error) as error:
        logging.warning("Could not read spectroscopy CSV %s: %s", csv_path, error)
    return None


def find_spectroscopy_rbcm(subject_dir: Path) -> Optional[float]:
    """Locate a subject spectroscopy CSV and return its RBC:M value."""
    spectroscopy_dirs = sorted(
        (
            path
            for path in subject_dir.iterdir()
            if path.is_dir() and path.name.lower() == "spectroscopy"
        ),
        key=lambda path: str(path).lower(),
    )
    for spectroscopy_dir in spectroscopy_dirs:
        csv_paths = sorted(
            (
                path
                for path in spectroscopy_dir.rglob("*")
                if path.is_file() and path.suffix.lower() == ".csv"
            ),
            key=lambda path: str(path).lower(),
        )
        for csv_path in csv_paths:
            value = read_spectroscopy_rbcm(csv_path)
            if value is not None:
                return value
    return None


def populate_discovered_defaults(row: dict[str, str], data_dir: Path) -> None:
    """Expand non-subject-specific runtime defaults into a discovered CSV row."""
    config = build_config(row, data_dir)
    row.update(
        recon_proton=csv_setting(config.recon.recon_proton),
        recon_key=csv_setting(config.recon.recon_key),
        scan_type=csv_setting(config.recon.scan_type),
        ramp_time=csv_setting(config.recon.ramp_time),
        oscillation_analysis=csv_setting(config.osc_recon.oscillation_analysis),
        output_folder=csv_setting(config.output_folder),
        vc_correction=csv_setting(config.osc_recon.vc_correction),
        segmentation_key=csv_setting(config.segmentation_key),
        registration_key=csv_setting(config.registration_key),
        bias_key=csv_setting(config.bias_key),
        reference_data_key=csv_setting(config.reference_data_key),
        vent_normalization_method=csv_setting(config.vent_normalization_method),
        n_skip_end=csv_setting(config.recon.n_skip_end),
        traj_type=csv_setting(config.recon.traj_type),
        traj_scaling_factor=csv_setting(config.recon.traj_scaling_factor),
        multi_echo=csv_setting(config.multi_echo),
    )


def discover_subject_rows(data_root: Path) -> list[dict[str, str]]:
    """Return one automatic batch row for each subject folder with supported input.

    Raw data is preferred when both raw and previously exported .mat files are
    present, allowing a later ``--discover --run`` invocation to reconstruct the
    subject. Discovered rows use manual ventilation masks found beside raw Twix
    or MRD files, and copy RBC:M from a subject spectroscopy CSV when available.
    Discovered subjects use Plummer reconstruction with oscillation analysis and
    VC correction enabled. Reconstruction settings controlled by base_config
    remain blank in the generated manifest.
    """
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    rows: list[dict[str, str]] = []
    for subject_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        suffixes = {
            path.suffix.lower() for path in subject_dir.rglob("*") if path.is_file()
        }
        if ".dat" in suffixes or ".h5" in suffixes:
            process_mode = "recon"
        elif ".mat" in suffixes:
            process_mode = "readin"
        else:
            logging.info("Skipping %s: no supported input files", subject_dir)
            continue

        row = {
            "subject_id": subject_dir.name,
            "data_dir": str(subject_dir),
            "process_mode": process_mode,
            "rbc_m_ratio": "",
            "segmentation_key": constants.SegmentationKey.MANUAL_VENT.value,
            "manual_seg_filepath": "",
            "recon_key": "plummer",
            "oscillation_analysis": "true",
            "vc_correction": "true",
        }
        manual_seg_filepath = find_manual_seg_filepath(subject_dir)
        if manual_seg_filepath is None:
            logging.warning(
                "Could not find mask_reg_corrected.nii or mask_reg.nii beside "
                "Twix/MRD data for %s.",
                subject_dir,
            )
        else:
            row["manual_seg_filepath"] = str(manual_seg_filepath)

        rbc_m_ratio = find_spectroscopy_rbcm(subject_dir)
        if rbc_m_ratio is None:
            logging.warning(
                "Could not find a valid spectroscopy rbcm value for %s.", subject_dir
            )
        else:
            row["rbc_m_ratio"] = f"{rbc_m_ratio:.3f}"
        if process_mode == "recon":
            row.update(
                ramp_time=str(discover_ramp_time(subject_dir)),
            )
        populate_discovered_defaults(row, subject_dir)
        demographics, demographics_source = resolve_demographics(
            row, subject_dir, "twix"
        )
        if demographics_source == "twix_header":
            row.update(
                age=csv_setting(demographics["age"]),
                sex=csv_setting(demographics["sex"]),
                height_cm=csv_setting(demographics["height_cm"]),
                weight_kg=csv_setting(demographics["weight_kg"]),
            )
        else:
            logging.warning(
                "Could not read demographics from a Twix header for %s; leaving "
                "demographic CSV cells blank.",
                subject_dir,
            )
        rows.append(row)

    return rows


def resolve_demographics(
    row: dict[str, str], data_dir: Path, input_type: str
) -> tuple[dict[str, Any], str]:
    """Use manifest values, then Twix metadata, then documented defaults."""
    demographics: dict[str, Any] = {}
    supplied = {
        "age": optional_int(row.get("age", "")),
        "sex": row.get("sex", "").strip(),
        "height_cm": optional_float(row.get("height_cm", "")),
        "weight_kg": optional_float(row.get("weight_kg", "")),
    }

    if all(supplied.values()):
        return supplied, "manifest"

    if input_type == "twix":
        try:
            import mapvbvd

            from utils import twix_utils

            dixon_file = next(
                path
                for path in data_dir.rglob("*.dat")
                if "dixon" in path.name.lower()
            )
            twix = mapvbvd.mapVBVD(str(dixon_file))
            demographics = {
                "age": twix_utils.get_patient_age(twix),
                "sex": twix_utils.get_patient_sex(twix),
                "height_cm": twix_utils.get_patient_height(twix),
                "weight_kg": twix_utils.get_patient_weight(twix),
            }
            if all(value == value and value != "" for value in demographics.values()):
                return demographics, "twix_header"
        except (ImportError, OSError, StopIteration, ValueError):
            pass

    return DEFAULT_DEMOGRAPHICS.copy(), "defaults"


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

    value = row.get("combine_reports", "").strip()
    if value:
        config.combine_reports = is_true(value)

    # Optional subject-level pipeline settings.
    for field in (
        "segmentation_key",
        "manual_seg_filepath",
        "vent_normalization_method",
        "trachea_plus_lung_mask_filepath",
        "registration_key",
        "manual_reg_filepath",
        "bias_key",
        "dicom_proton_dir",
        "reference_data_key",
        "phase_gas_acq_diss",
        "area_gas_acq_diss",
        "git_compare_branch",
    ):
        value = row.get(field, "").strip()
        if value:
            setattr(config, field, value)

    for field in ("bag_volume", "patient_frc"):
        value = optional_float(row.get(field, ""))
        if value is not None:
            setattr(config, field, value)

    for field in ("auto_make_trachea_plus_lung_mask", "multi_echo", "git_always_show"):
        value = row.get(field, "").strip()
        if value:
            setattr(config, field, is_true(value))

    # Optional RBC:M override. Blank leaves the existing spectroscopy workflow.
    rbc_m_ratio = optional_float(row.get("rbc_m_ratio", ""))
    if rbc_m_ratio is not None:
        config.rbc_m_ratio = rbc_m_ratio

    # Nonblank demographics are explicit CSV overrides for raw-input metadata.
    demographics_override: dict[str, Any] = {}
    age = optional_int(row.get("age", ""))
    if age is not None:
        demographics_override["age"] = age
    sex = row.get("sex", "").strip()
    if sex:
        demographics_override["sex"] = sex
    height_cm = optional_float(row.get("height_cm", ""))
    if height_cm is not None:
        demographics_override["height_cm"] = height_cm
    weight_kg = optional_float(row.get("weight_kg", ""))
    if weight_kg is not None:
        demographics_override["weight_kg"] = weight_kg
    config.manifest_demographics = demographics_override

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

    for field in ("recon_size", "matrix_size", "n_skip_start", "n_skip_end"):
        value = optional_int(row.get(field, ""))
        if value is not None:
            setattr(config.recon, field, value)

    for field in (
        "kernel_sharpness_lr",
        "kernel_sharpness_hr",
        "traj_scaling_factor",
        "optimized_conta_phase",
        "ramp_time",
    ):
        value = optional_float(row.get(field, ""))
        if value is not None:
            setattr(config.recon, field, value)

    traj_type = row.get("traj_type", "").strip()
    if traj_type:
        config.recon.traj_type = traj_type

    for field in ("gas_contamination_correction", "remove_contamination", "remove_noisy_projections"):
        value = row.get(field, "").strip()
        if value:
            setattr(config.recon, field, is_true(value))

    # Gradient delays are optional per-subject overrides.
    for field in ("del_x", "del_y", "del_z"):
        value = optional_float(row.get(field, ""))
        if value is not None:
            setattr(config.recon, field, value)

    config.osc_recon.oscillation_analysis = is_true(
        row.get("oscillation_analysis", "false")
    )
    for field in ("key_radius_pct",):
        value = optional_int(row.get(field, ""))
        if value is not None:
            setattr(config.osc_recon, field, value)
    value = row.get("vc_correction", "").strip()
    if value:
        config.osc_recon.vc_correction = is_true(value)

    # Select exactly one existing pipeline entry point.
    config.processes.gx_mapping_recon = row["process_mode"] == "recon"
    config.processes.gx_mapping_readin = row["process_mode"] == "readin"

    return config


def config_snapshot(
    config: base_config.Config,
    input_type: str,
    demographics: dict[str, Any],
    demographics_source: str,
) -> dict[str, Any]:
    """Return the relevant generated settings as a portable JSON record."""
    return {
        "subject_id": config.subject_id,
        "data_dir": config.data_dir,
        "output_folder": config.output_folder,
        "combine_reports": config.combine_reports,
        "input_type": input_type,
        "rbc_m_ratio": config.rbc_m_ratio,
        "hb": config.hb,
        "hb_correction_key": config.hb_correction_key,
        "corrected_lung_volume": config.corrected_lung_volume,
        "vol_correction_key": config.vol_correction_key,
        "demographics": demographics,
        "demographics_source": demographics_source,
        "pipeline": {
            "segmentation_key": config.segmentation_key,
            "manual_seg_filepath": config.manual_seg_filepath,
            "vent_normalization_method": config.vent_normalization_method,
            "bag_volume": config.bag_volume,
            "patient_frc": config.patient_frc,
            "auto_make_trachea_plus_lung_mask": config.auto_make_trachea_plus_lung_mask,
            "trachea_plus_lung_mask_filepath": config.trachea_plus_lung_mask_filepath,
            "registration_key": config.registration_key,
            "manual_reg_filepath": config.manual_reg_filepath,
            "bias_key": config.bias_key,
            "dicom_proton_dir": config.dicom_proton_dir,
            "multi_echo": config.multi_echo,
            "reference_data_key": config.reference_data_key,
            "phase_gas_acq_diss": config.phase_gas_acq_diss,
            "area_gas_acq_diss": config.area_gas_acq_diss,
        },
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
            "recon_size": config.recon.recon_size,
            "matrix_size": config.recon.matrix_size,
            "kernel_sharpness_lr": config.recon.kernel_sharpness_lr,
            "kernel_sharpness_hr": config.recon.kernel_sharpness_hr,
            "n_skip_start": json_value(config.recon.n_skip_start),
            "n_skip_end": config.recon.n_skip_end,
            "ramp_time": config.recon.ramp_time,
            "traj_type": config.recon.traj_type,
            "traj_scaling_factor": config.recon.traj_scaling_factor,
            "gas_contamination_correction": config.recon.gas_contamination_correction,
            "remove_contamination": config.recon.remove_contamination,
            "remove_noisy_projections": config.recon.remove_noisy_projections,
            "optimized_conta_phase": config.recon.optimized_conta_phase,
        },
        "osc_recon": {
            "oscillation_analysis": config.osc_recon.oscillation_analysis,
            "key_radius_pct": config.osc_recon.key_radius_pct,
            "vc_correction": config.osc_recon.vc_correction,
        },
    }


def write_json_config(
    config: base_config.Config,
    input_type: str,
    demographics: dict[str, Any],
    demographics_source: str,
) -> Path:
    """Save a frozen generated configuration before processing."""
    config_dir = MANIFEST_ROOT / "generated_configs"
    config_dir.mkdir(exist_ok=True)

    config_path = config_dir / f"{config.subject_id}_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(
            config_snapshot(config, input_type, demographics, demographics_source),
            file,
            indent=2,
        )

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
        demographics, demographics_source = resolve_demographics(
            normalized_row, data_dir, input_type
        )
        config_path = write_json_config(
            config, input_type, demographics, demographics_source
        )
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

    data_root = args.data_root.resolve()
    if args.discover:
        rows = discover_subject_rows(data_root)
        if not rows:
            logging.warning("No supported subject inputs found under %s", data_root)
        else:
            write_discovered_manifest(rows)
            logging.info("Wrote discovered subjects to %s", DISCOVERED_MANIFEST_PATH)
    else:
        manifest_path = args.manifest.resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        rows = read_manifest(manifest_path)

    status_rows = []

    for row in rows:
        status_rows.append(process_subject(row, data_root, args.run))
        write_status(status_rows)

    summary = {
        state: sum(row["status"] == state for row in status_rows)
        for state in ("completed", "planned", "failed", "needs_review")
    }

    logging.info("Batch complete: %s", summary)


if __name__ == "__main__":
    main()
