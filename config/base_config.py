"""Base configuration file."""

import sys
import os

import numpy as np
from ml_collections import config_dict

# parent directory
sys.path.append("..")

from utils import constants

class Config(config_dict.ConfigDict):
    """Base config file.

    Attributes:
        data_dir: str, path to directory with subject imaging files
        subject_id: str, the subject id
        rbc_m_ratio: float, the RBC to M ratio from spectroscopy
        output_folder: str, name of folder to output gx files
        patient_frc: str, patient FRC value in L

        recon_proton: bool, whether to reconstruct proton images
        oscillation_analysis: bool, whether to perform oscillation imaging analysis
        recon_key: str, the type of reconstruction to perform (ROBERTSON, PLUMMER)
        segmentation_key: str, the segmentation key (CNN_VENT/PROTON, MANUAL_VENT/PROTON, THRESHOLD_VENT/FRACTIONAL_VENTILATION, SKIP)
        manual_seg_filepath: str, path to the manual segmentation nifti file
        del_x: str, the x direction gradient delay in microseconds
        del_y: str, the y direction gradient delay in microseconds
        del_z: str, the z direction gradient delay in microseconds

        vent_normalization_method: str, the method to normalize ventilation images (GLB_99, GLB_FV, GLB_MA, THRESHOLD_MA)
        reference_data_key: str, reference data key (DUKE_REFERENCE, MANUAL_REFERENCE)

        auto_make_trachea_plus_lung_mask: bool, whether to auto-generate all-signal mask if filepath missing or file not found
        trachea_plus_lung_mask_output_dir: str, where to write the auto-generated trachea plus lung mask
        trachea_plus_lung_mask_filepath: str, optional user override of big mask
        bag_volume: str, the bag volume in L

        key_radius_pct: int, percentage of points used to recon the keyhole image
        osc_recon_key: str, the type of reconstruction to perform for oscillation imaging (ROBERTSON, PLUMMER)
        vc_correction: bool, whether to correction oscillation images for relative capillary blood volume
    """

    def __init__(self):
        ### Initialize config parameters
        super().__init__()

        ### Subject-Specific Settings
        self.data_dir = ""
        self.subject_id = "test"
        self.rbc_m_ratio = 0.0
        self.output_folder = "gx"
        self.patient_frc = "None"

        ### Recon and Analysis

        # Recon Options
        self.recon_proton = True
        self.oscillation_analysis = False
        self.recon_key = constants.ReconKey.ROBERTSON.value

        # Segmentation
        self.segmentation_key = constants.SegmentationKey.CNN_VENT.value
        self.manual_seg_filepath = os.path.join(self.data_dir, "")

        # Gradient Delays - Required ONLY for Siemens .dat files
        self.del_x = "None"
        self.del_y = "None"
        self.del_z = "None"

        ### Histogram Analysis

        # Ventilation Normalization Method
        self.vent_normalization_method = constants.NormalizationMethods.GLB_99.value

        # Reference Thresholds
        self.reference_data_key = constants.ReferenceDataKey.DUKE_REFERENCE.value

        ### Fractional Ventilation Parameters (if using FV)

        # Auto-generate all-signal mask if filepath missing or file not found
        self.auto_make_trachea_plus_lung_mask = True

        # Where to write it if auto-generated
        self.trachea_plus_lung_mask_output_dir = self.data_dir
        self.trachea_plus_lung_mask_filepath = "" # Optional user override of big mask
        self.bag_volume = "None"

        ### Oscillation Imaging Options

        # Keyhole Radius
        self.key_radius_pct = 14
        self.osc_recon_key = constants.ReconKey.ROBERTSON.value
        self.vc_correction = False

        ### Do Not Edit Anyuthing Past This Point

        self.multi_echo = False
        self.corrections = Corrections()
        self.recon = Recon()
        self.proton = Proton()
        self.processes = Process()

class Corrections(object):
    """Define correction configurations.

    Attributes:
        bias_key: str, the bias field correction key (N4ITK, SKIP, RF_DEPOLARIZATION)
        hb_correction_key: str, the hemoglobin correction key (NONE, RBC_AND_MEMBRANE)
        hb: float, subject hb value in g/dL
        remove_noisy_projections: bool, whether to remove FIDs with suspected spike noise
        vol_correction_key: str, the volume correction key (NONE, RBC_AND_MEMBRANE)
        corrected_lung_volume: str, the corrected lung volume in L
        optimized_conta_phase: float, the optimized contamination phase in degrees
        remove_contamination: bool, whether to remove gas phase contamination
        gas_contamination_correction: bool, whether to perform gas contamination correction
        phase_gas_acq_diss: str, the phase of gas acquisition in degrees
        area_gas_acq_diss: str, the area of gas acquisition in degrees
    """

    def __init__(self):
        ### Initialize correction parameters

        # Bias Field Correction
        self.bias_key = constants.BiasfieldKey.N4ITK.value

        # Hemoglobin Correction
        self.hb_correction_key = constants.HbCorrectionKey.NONE.value
        self.hb = 0.0

        # Remove FIDs With Suspected Spike Noise
        self.remove_noisy_projections = False

        # Volume Correction (work in progress, use at your own risk)
        self.vol_correction_key = constants.VolCorrectionKey.NONE.value
        self.corrected_lung_volume = "NA"

        # Gas Phase Contamination Correction
        self.optimized_conta_phase = 49.9  # degree
        self.remove_contamination = False
        self.gas_contamination_correction = False
        self.phase_gas_acq_diss = "None" # degrees
        self.area_gas_acq_diss = "None"

class Recon(object):
    """Define reconstruction configurations.

    Attributes:
        recon_size: int, size to which the images are reconstructed
        matrix_size: int, the final matrix size
        ramp_time: str, gradient ramp time in microseconds
        traj_type: str, the trajectory type (SPIRAL, HALTON, HALTONSPIRAL, SPIRALRANDOM, ARCHIMEDIAN, GOLDENMEAN)
        kernel_sharpness_lr: float, the kernel sharpness for low resolution, higher SNR images
        kernel_sharpness_hr: float, the kernel sharpness for high resolution, lower SNR images
        n_skip_start: int, the number of frames to skip at the beginning
        n_skip_end: int, the number of frames to skip at the end
        traj_scaling_factor: str, scaling factor to apply to trajectories     
    """

    def __init__(self):
        ### Initialize reconstruction parameters
        
        # Reconstruction and Matrix Sizes
        self.recon_size = 64
        self.matrix_size = 128

        # Ramp time will read in by default, but may be specified in us
        self.ramp_time = "None"

        # Trajectory Type
        self.traj_type = constants.TrajType.HALTONSPIRAL.value

        # Additional Options
        self.kernel_sharpness_lr = 0.14
        self.kernel_sharpness_hr = 0.32
        self.n_skip_start = np.nan
        self.n_skip_end = 0

        # Scaling is calcuated automatically if not reading in trajectories directly
        # Will default to 1 when reading in trajectories, but may be specified if desired
        # Use 0.903 for Halton Spiral Cincy data
        self.traj_scaling_factor = "None"

class Proton(object):
    """Define the proton image reconstruction configurations.

    Attributes:
        registration_key: str, the registration key (MANUAL, MASK2GAS, PROTON2GAS, SKIP)
        manual_reg_filepath: str, path to the manual registration nifti file
        dicom_proton_dir: str, path to the proton dicom directory
    """

    def __init__(self):
        ### Initialize the proton image reconstruction parameters

        self.registration_key = constants.RegistrationKey.SKIP.value
        self.manual_reg_filepath = ""
        self.dicom_proton_dir = ""
        

class Process(object):
    """Define the evaluation processes.

    Attributes:
        gx_mapping_recon: bool, whether to perform gas exchange mapping with reconstruction
        gx_mapping_readin: bool, whether to perform gas exchange mapping by reading in the mat file
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