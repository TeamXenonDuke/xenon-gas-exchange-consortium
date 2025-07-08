"""Base configuration file."""

import sys

import numpy as np
from ml_collections import config_dict

from config import config_utils

# parent directory
sys.path.append("..")

from utils import constants

### If using known volume to compare multiple subject scans, I define the subject.vol only, set baggie and frc to 0. 
### If using a predicted volume, set both frc and baggie vol and leave subject.vol to 0.
class Config(config_dict.ConfigDict):
    """Base config file.

    Attributes:
        data_dir: str, path to the data directory
        hb_correction_key: str, hemoglobin correction key
        hb: float, subject hb value in g/dL
        manual_reg_filepath: str, path to manual registration nifti file
        manual_seg_filepath: str, path to the manual segmentation nifti file
        dicom_proton_dir: str, path to the DICOM proton images
        processes: Process, the evaluation processes
        rbc_m_ratio: float, the RBC to M ratio
        reference_data_key: str, reference data key
        reference_data: ReferenceData, reference data
        remove_contamination: bool, whether to remove gas contamination
        remove_noisy_projections: bool, whether to remove noisy projections
        segmentation_key: str, the segmentation key
        subject_id: str, the subject id
        vol: float, actual subject volume in L
        baggie_vol: float, baggie volume in L
        frc: float, GLI predicted FRC
    """

    def __init__(self):
        """Initialize config parameters."""
        super().__init__()
        self.data_dir = ""
        self.manual_seg_filepath = ""
        self.manual_reg_filepath = ""
        self.dicom_proton_dir = ""
        self.processes = Process()
        self.recon = Recon()
        self.reference_data_key = constants.ReferenceDataKey.REFERENCE_218_PPM_01.value
        self.reference_data = ReferenceData(self.reference_data_key)
        self.segmentation_key = constants.SegmentationKey.CNN_VENT.value
        self.registration_key = constants.RegistrationKey.SKIP.value
        self.bias_key = constants.BiasfieldKey.N4ITK.value
        self.hb_correction_key = constants.HbCorrectionKey.NONE.value
        self.vol_correction_key = constants.VolCorrectionKey.RBC_AND_MEMBRANE.value
        self.hb = 0.0
        self.baggie_vol = 0.0 
        self.subject_id = "test"
        self.rbc_m_ratio = 0.0
        self.vol = 0.0
        self.frc = 0.0 


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
        self.recon_key = constants.ReconKey.ROBERTSON.value
        self.scan_type = constants.ScanType.NORMALDIXON.value
        self.kernel_sharpness_lr = 0.14
        self.kernel_sharpness_hr = 0.32
        self.n_skip_start = config_utils.get_n_skip_start(self.scan_type)
        self.n_skip_end = 0
        self.recon_size = 128 #64 for new subjects, 128 for old subjects
        self.matrix_size = 128
        self.recon_proton = True
        self.remove_contamination = False
        self.remove_noisy_projections = True
        self.del_x = "None"
        self.del_y = "None"
        self.del_z = "None"
        self.traj_type = constants.TrajType.HALTONSPIRAL


class ReferenceData(object):
    """Define reference data.

    Attributes:
        threshold_vent (np.array): ventilation thresholds for binning
        threshold_rbc (np.array): rbc thresholds for binning
        threshold_membrane (np.array): membrane thresholds for binning
        reference_fit_vent (tuple): scaling factor, mean, and std of reference ventilation distribution
        reference_fit_rbc (tuple): scaling factor, mean, and std of reference rbc distribution
        reference_fit_membrane (tuple): scaling factor, mean, and std of reference membrane distribution
        reference_stats (dict): mean and std of defect, low, and high percentage of ventilation,
                                membrane, and rbc reference data
    """

    def __init__(self, reference_data_key):
        """Initialize the reconstruction parameters."""
        if (
            reference_data_key == constants.ReferenceDataKey.REFERENCE_218_PPM_01.value
        ) or (reference_data_key == constants.ReferenceDataKey.MANUAL.value):
            self.threshold_vent = np.array([0.185, 0.418, 0.647, 0.806, 0.933])
            self.threshold_rbc = np.array([0.1007, 0.2723, 0.512, 0.814, 1.1743]) * 1e-2
            self.threshold_membrane = (
                np.array([0.3826, 0.5928, 0.8486, 1.1498, 1.4964, 1.8883, 2.3254])
                * 1e-2
            )
            self.reference_fit_vent = (0.04074, 0.619, 0.196)
            self.reference_fit_rbc = (0.06106, 0.543 * 1e-2, 0.277 * 1e-2)
            self.reference_fit_membrane = (0.0700, 0.871 * 1e-2, 0.284 * 1e-2)
            self.reference_stats = {
                "vent_defect_avg": "5",
                "vent_defect_std": "3",
                "vent_low_avg": "16",
                "vent_low_std": "8",
                "vent_high_avg": "15",
                "vent_high_std": "5",
                "membrane_defect_avg": "1",
                "membrane_defect_std": "<1",
                "membrane_low_avg": "8",
                "membrane_low_std": "2",
                "membrane_high_avg": "1",
                "membrane_high_std": "1",
                "rbc_defect_avg": "4",
                "rbc_defect_std": "2",
                "rbc_low_avg": "14",
                "rbc_low_std": "6",
                "rbc_high_avg": "15",
                "rbc_high_std": "10",
                "rbc_m_ratio_avg": "0.59",
                "rbc_m_ratio_std": "0.12",
                "inflation_avg": "3.4",
                "inflation_std": "0.33",
            }
        else:
            raise ValueError("Invalid reference data key")


def get_config() -> config_dict.ConfigDict:
    """Return the config dict. This is a required function.

    Returns:
        a ml_collections.config_dict.ConfigDict
    """
    return Config()
