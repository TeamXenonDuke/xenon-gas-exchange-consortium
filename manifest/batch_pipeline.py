"""Discover subjects, write an editable manifest, and run the pipeline serially."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = MANIFEST_ROOT / "manifest.csv"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MANIFEST_FIELDS = [
    "subject_id",
    "subject_root",
    "data_dir",
    "source_file",
    "process_mode",
    "input_type",
    "available_inputs",
    "discovery_status",
    "discovery_message",
    "rbc_m_ratio",
    "hb",
    "corrected_lung_volume",
    "age",
    "sex",
    "height_cm",
    "weight_kg",
    "segmentation_key",
    "manual_seg_filepath",
    "registration_key",
    "manual_reg_filepath",
    "bias_key",
    "reference_data_key",
    "vent_normalization_method",
    "bag_volume",
    "patient_frc",
    "recon_proton",
    "recon_key",
    "recon_size",
    "matrix_size",
    "scan_type",
    "del_x",
    "del_y",
    "del_z",
    "ramp_time",
    "n_skip_start",
    "n_skip_end",
    "traj_type",
    "traj_scaling_factor",
    "kmax_factor",
    "remove_noisy_projections",
    "oscillation_analysis",
    "key_radius",
    "vc_correction",
    "output_folder",
    "dicom_proton_dir",
    "multi_echo",
]

STATUS_FIELDS = [
    "subject_id",
    "status",
    "process_mode",
    "input_type",
    "data_dir",
    "planned_config",
    "final_config",
    "output_dir",
    "started_at",
    "finished_at",
    "message",
]

TRUE_VALUES = {"1", "true", "yes", "y"}
FALSE_VALUES = {"0", "false", "no", "n"}
EMPTY_NUMBERS = {"", "na", "nan", "none"}
MRD_SUFFIXES = {".h5", ".mrd"}


@dataclass(frozen=True)
class SubjectInventory:
    """Files belonging to one discovered subject root."""

    root: Path
    files: tuple[Path, ...]
    dixon_mrd: tuple[Path, ...]
    dixon_twix: tuple[Path, ...]
    subject_mats: tuple[Path, ...]
    masks: tuple[Path, ...]
    spectroscopy_csvs: tuple[Path, ...]


@dataclass(frozen=True)
class InputSelection:
    process_mode: str
    input_type: str
    data_dir: Path
    source_file: Path
    available_inputs: str
    messages: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="CSV manifest to create with --discover or read for a batch run.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Large directory containing subject folders at any supported depth.",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Scan --data-root and write one manifest row per subject.",
    )
    parser.add_argument(
        "--prefer",
        choices=("mrd", "twix"),
        default="mrd",
        help="Input to select when both MRD and Twix are available.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run all valid manifest rows after validation.",
    )
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Acknowledge running a newly discovered manifest without manual review.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow --discover to replace an existing manifest.",
    )
    return parser.parse_args()


def strict_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def optional_string(value: str) -> str | None:
    value = value.strip()
    return value or None


def optional_number(value: str, parser: Callable[[str], Any]) -> Any | None:
    value = value.strip()
    if value.lower() in EMPTY_NUMBERS:
        return None
    return parser(value)


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return str(value).lower() if isinstance(value, bool) else str(value)


def is_dixon(path: Path) -> bool:
    return "dixon" in path.name.lower()


def source_root(path: Path, data_root: Path) -> Path | None:
    """Infer a subject root from a Dixon or subject-named MAT file."""
    ancestors = list(path.parents)
    for ancestor in ancestors:
        if ancestor == data_root.parent:
            break
        if ancestor.name.lower() == "data":
            return ancestor.parent

    if is_dixon(path):
        return path.parent

    if path.suffix.lower() == ".mat":
        for ancestor in ancestors:
            if ancestor == data_root.parent:
                break
            if ancestor.name.lower() == path.stem.lower():
                return ancestor
    return None


def scan_inventories(data_root: Path) -> list[SubjectInventory]:
    """Scan the filesystem once, then group all files by subject root."""
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    all_files = tuple(sorted(path for path in data_root.rglob("*") if path.is_file()))
    source_files = [
        path
        for path in all_files
        if (
            is_dixon(path)
            and (path.suffix.lower() == ".dat" or path.suffix.lower() in MRD_SUFFIXES)
        )
        or path.suffix.lower() == ".mat"
    ]
    roots = {
        root
        for path in source_files
        if (root := source_root(path, data_root)) is not None
    }

    inventories = []
    for root in sorted(roots):
        files = tuple(path for path in all_files if path == root or root in path.parents)
        subject_id = root.name.lower()
        dixon_mrd = tuple(
            path
            for path in files
            if is_dixon(path) and path.suffix.lower() in MRD_SUFFIXES
        )
        dixon_twix = tuple(
            path
            for path in files
            if is_dixon(path) and path.suffix.lower() == ".dat"
        )
        subject_mats = tuple(
            path
            for path in files
            if path.suffix.lower() == ".mat" and path.stem.lower() == subject_id
        )
        masks = tuple(
            path
            for path in files
            if path.name.lower() in {"mask_reg_corrected.nii", "mask_reg.nii"}
        )
        spectroscopy_csvs = tuple(
            path
            for path in files
            if path.suffix.lower() == ".csv"
            and any(parent.name.lower() == "spectroscopy" for parent in path.parents)
        )
        inventories.append(
            SubjectInventory(
                root=root,
                files=files,
                dixon_mrd=dixon_mrd,
                dixon_twix=dixon_twix,
                subject_mats=subject_mats,
                masks=masks,
                spectroscopy_csvs=spectroscopy_csvs,
            )
        )
    return inventories


def select_input(inventory: SubjectInventory, prefer: str) -> InputSelection:
    available = []
    if inventory.dixon_mrd:
        available.append("mrd")
    if inventory.dixon_twix:
        available.append("twix")
    if inventory.subject_mats:
        available.append("mat")

    candidates = (
        ("mrd", inventory.dixon_mrd),
        ("twix", inventory.dixon_twix),
    )
    if prefer == "twix":
        candidates = tuple(reversed(candidates))

    for input_type, paths in candidates:
        if not paths:
            continue
        messages = []
        if len(paths) > 1:
            messages.append(
                f"multiple {input_type} Dixon files found; select source_file manually"
            )
        source = paths[0]
        return InputSelection(
            process_mode="recon",
            input_type=input_type,
            data_dir=source.parent,
            source_file=source,
            available_inputs=",".join(available),
            messages=tuple(messages),
        )

    if inventory.subject_mats:
        messages = ()
        if len(inventory.subject_mats) > 1:
            messages = ("multiple subject MAT files found; select source_file manually",)
        source = inventory.subject_mats[0]
        return InputSelection(
            process_mode="readin",
            input_type="mat",
            data_dir=source.parent,
            source_file=source,
            available_inputs=",".join(available),
            messages=messages,
        )

    raise ValueError(f"No supported input under {inventory.root}")


def select_mask(inventory: SubjectInventory, data_dir: Path) -> tuple[Path | None, str]:
    """Choose a mask only from the selected acquisition directory or its parent."""
    for directory in (data_dir, data_dir.parent):
        candidates = [path for path in inventory.masks if path.parent == directory]
        for filename in ("mask_reg_corrected.nii", "mask_reg.nii"):
            matches = sorted(
                path for path in candidates if path.name.lower() == filename
            )
            if len(matches) == 1:
                return matches[0], ""
            if len(matches) > 1:
                return None, f"multiple {filename} files found in {directory}"
    return None, "manual mask not found near selected input"


def read_metadata(input_type: str, source_file: Path) -> dict[str, Any]:
    """Read demographics, scan date, and ramp time from one Dixon header."""
    if input_type == "twix":
        import mapvbvd

        from utils import twix_utils

        header = mapvbvd.mapVBVD(str(source_file))
        metadata = {
            "age": twix_utils.get_patient_age(header),
            "sex": twix_utils.get_patient_sex(header),
            "height_cm": twix_utils.get_patient_height(header),
            "weight_kg": twix_utils.get_patient_weight(header),
            "scan_date": twix_utils.get_scan_date(header),
        }
        try:
            metadata["ramp_time"] = twix_utils.get_ramp_time(header)
        except (AttributeError, TypeError, ValueError):
            metadata["ramp_time"] = None
        return metadata

    if input_type == "mrd":
        import ismrmrd

        from utils import mrd_utils

        dataset = ismrmrd.Dataset(str(source_file), "dataset", create_if_needed=False)
        try:
            header = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())
            metadata = {
                "age": mrd_utils.get_patient_age(header),
                "sex": mrd_utils.get_patient_sex(header),
                "height_cm": mrd_utils.get_patient_height(header),
                "weight_kg": mrd_utils.get_patient_weight(header),
                "scan_date": mrd_utils.get_scan_date(header),
            }
            try:
                metadata["ramp_time"] = mrd_utils.get_ramp_time(header)
            except (AttributeError, IndexError, TypeError, ValueError):
                metadata["ramp_time"] = None
            return metadata
        finally:
            dataset.close()

    return {}


def read_rbcm_candidates(paths: Iterable[Path]) -> list[dict[str, Any]]:
    candidates = []
    for csv_path in sorted(paths):
        with csv_path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                continue
            names = {name.strip().lower(): name for name in reader.fieldnames}
            rbcm_name = names.get("rbcm")
            if rbcm_name is None:
                continue
            for record in reader:
                try:
                    value = optional_number(record.get(rbcm_name, ""), float)
                except ValueError:
                    continue
                if value is None or not math.isfinite(value) or value <= 0:
                    continue
                candidates.append(
                    {
                        "path": csv_path,
                        "value": value,
                        "subject_id": record.get(names.get("subject_id", ""), "").strip(),
                        "scan_date": record.get(names.get("scan_date", ""), "").strip(),
                    }
                )
    return candidates


def select_rbcm(
    inventory: SubjectInventory,
    scan_date: str,
) -> tuple[float | None, str]:
    candidates = read_rbcm_candidates(inventory.spectroscopy_csvs)
    subject_matches = [
        candidate
        for candidate in candidates
        if candidate["subject_id"].lower() == inventory.root.name.lower()
    ]
    if subject_matches:
        candidates = subject_matches

    date_matches = [
        candidate for candidate in candidates if scan_date and candidate["scan_date"] == scan_date
    ]
    if date_matches:
        candidates = date_matches

    unique_values = sorted({candidate["value"] for candidate in candidates})
    if len(unique_values) == 1:
        return unique_values[0], ""
    if len(unique_values) > 1:
        return None, "multiple spectroscopy RBC:M values found; select manually"
    return None, "spectroscopy RBC:M not found"


def discovered_row(inventory: SubjectInventory, prefer: str) -> dict[str, str]:
    selection = select_input(inventory, prefer)
    review_messages = list(selection.messages)
    notes = []

    try:
        metadata = read_metadata(selection.input_type, selection.source_file)
    except (
        AttributeError,
        ImportError,
        IndexError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        metadata = {}
        review_messages.append(
            f"metadata unavailable: {type(error).__name__}: {error}"
        )

    mask, mask_message = select_mask(inventory, selection.data_dir)
    if mask_message:
        target = review_messages if mask_message.startswith("multiple") else notes
        target.append(mask_message)

    try:
        rbcm, rbcm_message = select_rbcm(
            inventory,
            csv_value(metadata.get("scan_date")),
        )
    except (OSError, UnicodeError, csv.Error) as error:
        rbcm = None
        rbcm_message = f"could not read spectroscopy CSV: {error}"
    if rbcm_message:
        target = (
            notes
            if rbcm_message == "spectroscopy RBC:M not found"
            else review_messages
        )
        target.append(rbcm_message)

    row = {field: "" for field in MANIFEST_FIELDS}
    row.update(
        subject_id=inventory.root.name,
        subject_root=str(inventory.root.resolve()),
        data_dir=str(selection.data_dir.resolve()),
        source_file=str(selection.source_file.resolve()),
        process_mode=selection.process_mode,
        input_type=selection.input_type,
        available_inputs=selection.available_inputs,
        discovery_status="needs_review" if review_messages else "ready",
        discovery_message="; ".join(review_messages + notes),
        rbc_m_ratio=csv_value(rbcm),
        segmentation_key="manual_vent" if mask else "cnn_vent",
        manual_seg_filepath=str(mask.resolve()) if mask else "",
        registration_key="skip",
        bias_key="n4itk",
        reference_data_key="duke_reference",
        vent_normalization_method="glb_99",
        recon_proton="true",
        recon_key="plummer",
        oscillation_analysis="true",
        vc_correction="true",
        output_folder="gx_batch",
        multi_echo="false",
    )
    for field in ("age", "sex", "height_cm", "weight_kg", "ramp_time"):
        row[field] = csv_value(metadata.get(field))
    return row


def discover(data_root: Path, prefer: str) -> list[dict[str, str]]:
    return [discovered_row(item, prefer) for item in scan_inventories(data_root)]


def write_manifest(
    path: Path,
    rows: list[dict[str, str]],
    overwrite: bool = False,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {
            "subject_id",
            "data_dir",
            "source_file",
            "process_mode",
            "input_type",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Manifest requires columns: {', '.join(sorted(required))}")
        return list(reader)


def parse_optional_field(
    row: dict[str, str],
    field: str,
    parser: Callable[[str], Any],
) -> Any | None:
    value = row.get(field, "")
    if parser is str:
        return optional_string(value)
    return optional_number(value, parser)


def set_values(
    owner: Any,
    row: dict[str, str],
    fields: tuple[str, ...],
    parser: Callable[[str], Any],
) -> None:
    for field in fields:
        value = parse_optional_field(row, field, parser)
        if value is not None:
            setattr(owner, field, value)


def validate_row(row: dict[str, str]) -> list[str]:
    errors = []
    subject_id = row.get("subject_id", "").strip()
    process_mode = row.get("process_mode", "").strip().lower()
    input_type = row.get("input_type", "").strip().lower()
    data_dir_value = row.get("data_dir", "").strip()
    source_file_value = row.get("source_file", "").strip()
    data_dir = Path(data_dir_value) if data_dir_value else None
    source_file = Path(source_file_value) if source_file_value else None

    if not subject_id:
        errors.append("subject_id is blank")
    if process_mode not in {"recon", "readin"}:
        errors.append("process_mode must be recon or readin")
    if data_dir is None or not data_dir.is_dir():
        errors.append("data_dir does not exist")
    if source_file is None or not source_file.is_file():
        errors.append("source_file does not exist")
    elif data_dir is not None and data_dir.is_dir():
        if source_file.parent.resolve() != data_dir.resolve():
            errors.append("source_file must be inside data_dir")

    if row.get("discovery_status", "").strip().lower() == "needs_review":
        message = row.get("discovery_message", "").strip()
        errors.append(f"discovery needs review: {message or 'resolve discovery choices'}")

    if process_mode == "recon" and input_type not in {"twix", "mrd"}:
        errors.append("recon mode requires input_type twix or mrd")
    if process_mode == "readin" and input_type != "mat":
        errors.append("readin mode requires input_type mat")
    suffix = source_file.suffix.lower() if source_file is not None else ""
    if input_type == "twix" and suffix != ".dat":
        errors.append("twix source_file must end in .dat")
    if input_type == "mrd" and suffix not in MRD_SUFFIXES:
        errors.append("mrd source_file must end in .h5 or .mrd")
    if input_type == "mat" and suffix != ".mat":
        errors.append("mat source_file must end in .mat")

    if row.get("segmentation_key", "").strip() == "manual_vent":
        mask = Path(row.get("manual_seg_filepath", ""))
        if not mask.is_file():
            errors.append("manual_vent requires an existing manual_seg_filepath")

    if input_type == "twix":
        for field in ("del_x", "del_y", "del_z"):
            try:
                value = optional_number(row.get(field, ""), float)
            except ValueError:
                errors.append(f"{field} must be numeric")
                continue
            if value is None:
                errors.append(f"twix input requires {field}")

    numeric_rules = {
        "rbc_m_ratio": (0, None),
        "age": (0, 120),
        "height_cm": (50, 250),
        "weight_kg": (2, 500),
        "n_skip_start": (0, None),
        "n_skip_end": (0, None),
        "key_radius": (0, None),
        "kmax_factor": (0, None),
    }
    for field, (minimum, maximum) in numeric_rules.items():
        try:
            value = optional_number(row.get(field, ""), float)
        except ValueError:
            errors.append(f"{field} must be numeric")
            continue
        if value is None:
            continue
        if value <= minimum and field in {"rbc_m_ratio", "kmax_factor"}:
            errors.append(f"{field} must be greater than {minimum}")
        elif value < minimum:
            errors.append(f"{field} must be at least {minimum}")
        if maximum is not None and value > maximum:
            errors.append(f"{field} must be at most {maximum}")

    for field in (
        "recon_proton",
        "remove_noisy_projections",
        "oscillation_analysis",
        "vc_correction",
        "multi_echo",
    ):
        value = row.get(field, "").strip()
        if value:
            try:
                strict_bool(value)
            except ValueError as error:
                errors.append(f"{field}: {error}")
    return errors


def build_config(row: dict[str, str]) -> Any:
    """Build a normal pipeline Config directly from a validated manifest row."""
    from config import base_config
    from utils import constants

    process_mode = row["process_mode"].strip().lower()
    config = base_config.Config()
    config.subject_id = row["subject_id"].strip()
    config.subject_root = row.get("subject_root", "").strip()
    config.data_dir = row["data_dir"].strip()
    config.input_type = row["input_type"].strip().lower()
    config.source_file = row["source_file"].strip()
    config.output_folder = row.get("output_folder", "").strip() or "gx_batch"
    config.trachea_plus_lung_mask_output_dir = config.data_dir

    set_values(
        config,
        row,
        (
            "segmentation_key",
            "manual_seg_filepath",
            "registration_key",
            "manual_reg_filepath",
            "bias_key",
            "reference_data_key",
            "vent_normalization_method",
            "dicom_proton_dir",
        ),
        str,
    )
    set_values(config, row, ("rbc_m_ratio", "bag_volume", "patient_frc"), float)
    set_values(config, row, ("multi_echo",), strict_bool)

    hb = optional_number(row.get("hb", ""), float)
    if hb is not None:
        config.hb = hb
        config.hb_correction_key = constants.HbCorrectionKey.RBC_AND_MEMBRANE.value

    lung_volume = optional_number(row.get("corrected_lung_volume", ""), float)
    if lung_volume is not None:
        config.corrected_lung_volume = lung_volume
        config.vol_correction_key = constants.VolCorrectionKey.RBC_AND_MEMBRANE.value

    config.manifest_demographics = {}
    for manifest_field, config_field, parser in (
        ("age", "age", int),
        ("sex", "sex", str),
        ("height_cm", "height_cm", float),
        ("weight_kg", "weight_kg", float),
    ):
        value = parse_optional_field(row, manifest_field, parser)
        if value is not None:
            config.manifest_demographics[config_field] = value

    set_values(config.recon, row, ("recon_key", "scan_type", "traj_type"), str)
    set_values(
        config.recon,
        row,
        ("recon_size", "matrix_size", "n_skip_start", "n_skip_end"),
        int,
    )
    set_values(
        config.recon,
        row,
        ("del_x", "del_y", "del_z", "ramp_time", "traj_scaling_factor", "kmax_factor"),
        float,
    )
    set_values(
        config.recon,
        row,
        ("recon_proton", "remove_noisy_projections"),
        strict_bool,
    )
    set_values(
        config.osc_recon,
        row,
        ("oscillation_analysis", "vc_correction"),
        strict_bool,
    )
    set_values(config.osc_recon, row, ("key_radius",), int)

    config.processes.gx_mapping_recon = process_mode == "recon"
    config.processes.gx_mapping_readin = process_mode == "readin"
    return config


def write_snapshot(config: Any) -> Path:
    directory = MANIFEST_ROOT / "generated_configs"
    directory.mkdir(exist_ok=True)
    path = directory / f"{config.subject_id}_planned_config.json"
    path.write_text(config.to_json_best_effort(indent=2), encoding="utf-8")
    return path


def status_row(row: dict[str, str], **updates: str) -> dict[str, str]:
    status = {field: "" for field in STATUS_FIELDS}
    status.update(
        subject_id=row.get("subject_id", ""),
        process_mode=row.get("process_mode", ""),
        input_type=row.get("input_type", ""),
        data_dir=row.get("data_dir", ""),
    )
    status.update(updates)
    return status


def write_status(rows: list[dict[str, str]]) -> None:
    path = MANIFEST_ROOT / "batch_status.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def process_subject(row: dict[str, str], run: bool) -> dict[str, str]:
    errors = validate_row(row)
    if errors:
        return status_row(row, status="needs_review", message="; ".join(errors))

    try:
        config = build_config(row)
        snapshot = write_snapshot(config)
    except (TypeError, ValueError) as error:
        return status_row(row, status="needs_review", message=str(error))

    output_dir = Path(config.data_dir) / config.output_folder
    final_config = output_dir / f"{config.subject_id}_config_gx_imaging.json"
    status = status_row(
        row,
        status="planned",
        planned_config=str(snapshot),
        final_config=str(final_config),
        output_dir=str(output_dir),
    )
    if not run:
        return status

    status["status"] = "running"
    status["started_at"] = datetime.now().isoformat(timespec="seconds")
    try:
        previous_cwd = Path.cwd()
        os.chdir(PROJECT_ROOT)
        try:
            if config.processes.gx_mapping_recon:
                from main import gx_mapping_reconstruction

                gx_mapping_reconstruction(config)
            else:
                from main import gx_mapping_readin

                gx_mapping_readin(config)
        finally:
            os.chdir(previous_cwd)
        status["status"] = "completed"
    except Exception as error:
        logging.exception("Subject %s failed", config.subject_id)
        status["status"] = "failed"
        status["message"] = f"{type(error).__name__}: {error}"
    status["finished_at"] = datetime.now().isoformat(timespec="seconds")
    return status


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    manifest_path = args.manifest.resolve()

    if args.discover:
        if args.run and not args.allow_unreviewed:
            raise ValueError(
                "--discover --run requires --allow-unreviewed; review the CSV first"
            )
        rows = discover(args.data_root.resolve(), args.prefer)
        write_manifest(manifest_path, rows, overwrite=args.overwrite)
        logging.info("Discovered %d subjects in %s", len(rows), manifest_path)
        if not args.run:
            return
    else:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        rows = read_manifest(manifest_path)

    statuses: list[dict[str, str]] = []
    for row in rows:
        statuses.append(process_subject(row, args.run))
        write_status(statuses)

    summary = {
        state: sum(row["status"] == state for row in statuses)
        for state in ("completed", "planned", "failed", "needs_review")
    }
    logging.info("Batch complete: %s", summary)


if __name__ == "__main__":
    main()
