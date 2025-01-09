"""Base configuration file."""

import sys

import numpy as np
from ml_collections import config_dict

from config import config_utils

# parent directory
sys.path.append("..")

from utils import constants


class Config(config_dict.ConfigDict):
    """Base config file.

    Attributes:
        data_dir: str, path to directory with subject imaging files
        hb_correction_key: str, hemoglobin correction key
        hb: float, subject hb value in g/dL
        manual_reg_filepath: str, path to manual registration nifti file
        manual_seg_filepath: str, path to the manual segmentation nifti file
        dicom_proton_dir: str, path to the DICOM proton images
        processes: Process, the evaluation processes
        rbc_m_ratio: float, the RBC to M ratio from spectroscopy
        reference_data_key: str, reference data key
        remove_contamination: bool, whether to remove gas contamination
        remove_noisy_projections: bool, whether to remove noisy projections
        segmentation_key: str, the segmentation key (CNN_VENT, MANUAL)
        subject_id: str, the subject id
    """

    def __init__(self):
        """Initialize config parameters."""
        super().__init__()
        # Standard parameters - MUST be specified
        self.data_dir = ""
        self.subject_id = "test"
        self.rbc_m_ratio = 0.0
        self.segmentation_key = constants.SegmentationKey.CNN_VENT.value
        self.manual_seg_filepath = ""

        # Additional options
        self.reference_data_key = constants.ReferenceDataKey.DUKE_REFERENCE.value
        self.registration_key = constants.RegistrationKey.SKIP.value
        self.bias_key = constants.BiasfieldKey.N4ITK.value
        self.hb_correction_key = constants.HbCorrectionKey.NONE.value
        self.hb = 0.0
        self.dicom_proton_dir = ""
        self.multi_echo = False
        self.registration_key = constants.RegistrationKey.SKIP.value
        self.manual_reg_filepath = ""
        self.processes = Process()
        self.recon = Recon()


class Recon(object):
    """Define reconstruction configurations.

    Attributes:
        del_x: str, the x direction gradient delay in microseconds
        del_y: str, the y direction gradient delay in microseconds
        del_z: str, the z direction gradient delay in microseconds
        traj_type: str, the trajectory type
        recon_key: str, the reconstruction key
        recon_proton: bool, whether to reconstruct proton images
        remove_contamination: bool, whether to remove gas contamination
        remove_noisy_projections: bool, whether to remove noisy projections
        scan_type: str, the scan type
        kernel_sharpness_lr: float, the kernel sharpness for low resolution, higher
            SNR images
        kernel_sharpness_hr: float, the kernel sharpness for high resolution, lower
            SNR images
        n_skip_start: int, the number of frames to skip at the beginning
        n_skip_end: int, the number of frames to skip at the end
        key_radius: int, the key radius for the keyhole image
        matrix_size: int, the final matrix size
    """

    def __init__(self):
        """Initialize the reconstruction parameters."""
        # Gradient delays - MUST be specified
        self.del_x = "None"
        self.del_y = "None"
        self.del_z = "None"

        # Reconstruction and matrix sizes
        self.recon_size = 64
        self.matrix_size = 128

        # Additional options
        self.recon_proton = True
        self.recon_key = constants.ReconKey.ROBERTSON.value
        self.scan_type = constants.ScanType.NORMALDIXON.value
        self.kernel_sharpness_lr = 0.14
        self.kernel_sharpness_hr = 0.32
        self.n_skip_start = config_utils.get_n_skip_start(self.scan_type)
        self.n_skip_end = 0
        self.remove_contamination = False
        self.remove_noisy_projections = True
        self.traj_type = constants.TrajType.HALTONSPIRAL


class Process(object):
    """Define the evaluation processes.

    Attributes:
        gx_mapping_recon: bool, whether to perform gas exchange mapping
            with reconstruction
        gx_mapping_readin: bool, whether to perform gas exchange mapping
            by reading in the mat file
    """

    def __init__(self):
        """Initialize the process parameters."""
        self.gx_mapping_recon = True
        self.gx_mapping_readin = False


def get_config() -> config_dict.ConfigDict:
    """Return the config dict. This is a required function.

    Returns:
        a ml_collections.config_dict.ConfigDict
    """
    return Config()
