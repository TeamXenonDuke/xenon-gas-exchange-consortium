"""Scripts to run gas exchange mapping pipeline."""

import numpy as np
from absl import app, flags
from ml_collections import config_flags

from config import base_config
from subject_classmap import Subject
from utils import constants

import logging

for n in list(logging.root.manager.loggerDict):
    if n.startswith("fontTools"):
        logging.getLogger(n).setLevel(logging.WARNING)
        logging.getLogger(n).disabled = True
        logging.getLogger(n).propagate = False

FLAGS = flags.FLAGS

_CONFIG = config_flags.DEFINE_config_file("config", None, "config file.")
flags.DEFINE_boolean("force_recon", False, "force reconstruction for the subject")
flags.DEFINE_boolean("force_readin", False, "force read in .mat for the subject")
flags.DEFINE_bool("force_segmentation", False, "run segmentation again.")
flags.DEFINE_string("folder", None, "relative path to subject data folder.")


def apply_manifest_demographics(subject: Subject, config: base_config.Config) -> None:
    """Apply demographics supplied by the batch manifest after input loading."""
    overrides = getattr(config, "manifest_demographics", {})
    fields = {
        "age": constants.IOFields.AGE,
        "sex": constants.IOFields.SEX,
        "height_cm": constants.IOFields.HEIGHT,
        "weight_kg": constants.IOFields.WEIGHT,
    }
    for name, io_field in fields.items():
        if name in overrides:
            subject.dict_dis[io_field] = overrides[name]


def read_raw_data(subject: Subject, config: base_config.Config) -> None:
    """Read the manifest-selected input type or auto-detect for legacy configs."""
    input_type = getattr(config, "input_type", "").lower()
    if input_type == "twix":
        subject.read_twix_files()
        return
    if input_type == "mrd":
        subject.read_mrd_files()
        return
    if input_type:
        raise ValueError(f"Unsupported reconstruction input_type: {input_type}")

    try:
        subject.read_twix_files()
    except ValueError:
        logging.info("No Twix input found; trying MRD input.")
        subject.read_mrd_files()


def gx_mapping_reconstruction(config: base_config.Config):
    """Run the gas exchange mapping pipeline with reconstruction.

    Args:
        config (config_dict.ConfigDict): config dict
    """
    subject = Subject(config=config)
    subject.clear_temp_file()
    if config.vent_normalization_method not in ["glb_99", "glb_fv", "glb_ma","threshold_ma"]:
        msg = (
            f"You choose a wrong normalization method: {config.vent_normalization_method}! It has to be: GLB_99, GLB_FV, GLB_MA, or THRESHOLD_MA"
        )
        raise ValueError(msg)
    read_raw_data(subject, config)
    apply_manifest_demographics(subject, config)
    subject.calculate_rbc_m_ratio()
    logging.info("Reconstructing images")
    subject.preprocess()
    subject.reconstruction_gas()
    subject.reconstruction_dissolved()
    if config.recon.recon_proton:
        if getattr(subject, "dict_ute", None):
            subject.reconstruction_ute()
        elif config.dicom_proton_dir:
            subject.read_dicom_files()
        else:
            subject.image_proton = np.zeros_like(subject.image_gas_highreso)
    elif config.dicom_proton_dir:
        subject.read_dicom_files()
    else:
        subject.image_proton = np.zeros_like(subject.image_gas_highreso)
    subject.segmentation()
    subject.registration()
    subject.biasfield_correction()
    subject.gas_binning()
    subject.dixon_decomposition()
    subject.hb_correction()
    subject.vol_correction()
    subject.dissolved_analysis()
    subject.dissolved_binning()
    if config.osc_recon.oscillation_analysis:
        subject.reconstruction_rbc_oscillation()
        subject.oscillation_analysis()
        subject.oscillation_binning()
    subject.get_statistics()
    subject.get_info()
    subject.save_subject_to_mat()
    subject.write_stats_to_csv()
    subject.generate_figures()
    subject.generate_pdf()
    subject.save_files()
    subject.save_config_as_json()
    subject.move_output_files()
    subject.check_git_version()
    logging.info("Complete")


def gx_mapping_readin(
    config: base_config.Config,
    force_segmentation: bool = False,
):
    """Run the gas exchange imaging pipeline by reading in .mat file.

    Args:
        config (config_dict.ConfigDict): config dict
    """
    subject = Subject(config=config)
    subject.clear_temp_file()
    if config.vent_normalization_method not in ["glb_99", "glb_fv", "glb_ma","threshold_ma"]:
        msg = (
            f"You choose a wrong normalization method: {config.vent_normalization_method}! It has to be: GLB_99, GLB_FV, GLB_MA, or THRESHOLD_MA"
        )
        raise ValueError(msg)
    subject.read_mat_file()
    apply_manifest_demographics(subject, config)
    if force_segmentation:
        subject.segmentation()
    subject.gas_binning()
    subject.dixon_decomposition()
    subject.hb_correction()
    subject.vol_correction()
    subject.dissolved_analysis()
    subject.dissolved_binning()
    if config.osc_recon.oscillation_analysis:
        subject.oscillation_analysis()
        subject.oscillation_binning()
    subject.get_statistics()
    subject.get_info()
    subject.save_subject_to_mat()
    subject.write_stats_to_csv()
    subject.generate_figures()
    subject.generate_pdf()
    subject.save_files()
    subject.save_config_as_json()
    subject.move_output_files()
    subject.check_git_version()
    logging.info("Complete")


def main(argv):
    """Run the gas exchange imaging pipeline.

    Either run the reconstruction or read in the .mat file.
    """
    config = _CONFIG.value
    if FLAGS.folder:
        config.data_dir = FLAGS.folder
    if FLAGS.force_recon:
        logging.info("Gas exchange imaging mapping with reconstruction.")
        gx_mapping_reconstruction(config)
    elif FLAGS.force_readin:
        logging.info("Gas exchange imaging mapping with reconstruction.")
        gx_mapping_readin(config, force_segmentation=FLAGS.force_segmentation)
    elif config.processes.gx_mapping_recon:
        logging.info("Gas exchange imaging mapping with reconstruction.")
        gx_mapping_reconstruction(config)
    elif config.processes.gx_mapping_readin:
        logging.info("Gas exchange imaging mapping with reconstruction.")
        gx_mapping_readin(config, force_segmentation=FLAGS.force_segmentation)
    else:
        pass


if __name__ == "__main__":
    app.run(main)
