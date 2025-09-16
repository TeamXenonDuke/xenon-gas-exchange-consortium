"""Define important constants used throughout the pipeline."""

import enum

import numpy as np

FOVINFLATIONSCALE3D = 1000.0
GRYOMAGNETIC_RATIO = 11.777  # MHz/T
T2STAR_GAS = 1.8e-2  # seconds
T2STAR_RBC_3T = 1.044575 * 1e-3  # seconds
T2STAR_MEMBRANE_3T = 0.988588  * 1e-3  # seconds


KCO_ALPHA = 11.2  # membrane
KCO_BETA = 14.6  # RBC
VA_ALPHA = 1.43

NONE = "None"


class IOFields(object):
    """General IOFields constants."""

    AGE = "age"
    SEX = "sex"
    HEIGHT = "height"
    BANDWIDTH = "bandwidth"
    USER_LUNG_VOLUME_VALUE ="user_lung_volume_value"
    BIASFIELD_KEY = "biasfield_key"
    BONUS_SPECTRA_LABELS = "bonus_spectra_labels"
    CONTRAST_LABELS = "contrast_labels"
    SAMPLE_TIME = "sample_time"
    FA_DIS = "fa_dis"
    FA_GAS = "fa_gas"
    FIDS = "fids"
    FIDS_DIS = "fids_dis"
    FIDS_GAS = "fids_gas"
    FIELD_STRENGTH = "field_strength"
    FLIP_ANGLE_FACTOR = "flip_angle_factor"
    FOV = "fov"
    XE_CENTER_FREQUENCY = "xe_center_frequency"
    XE_DISSOLVED_OFFSET_FREQUENCY = "xe_dissolved_offset_frequency"
    GIT_BRANCH = "git_branch"
    GRAD_DELAY_X = "grad_delay_x"
    GRAD_DELAY_Y = "grad_delay_y"
    GRAD_DELAY_Z = "grad_delay_z"
    HB_CORRECTION_KEY = "hb_correction_key"
    HB = "hb"
    INSTITUTION = "institution"
    RBC_HB_CORRECTION_FACTOR = "rbc_hb_correction_factor"
    MEMBRANE_HB_CORRECTION_FACTOR = "membrane_hb_correction_factor"
    KERNEL_SHARPNESS = "kernel_sharpness"
    N_FRAMES = "n_frames"
    N_SKIP_END = "n_skip_end"
    N_SKIP_START = "n_skip_start"
    N_DIS_REMOVED = "n_dis_removed"
    N_GAS_REMOVED = "n_gas_removed"
    N_POINTS = "n_points"
    ORIENTATION = "orientation"
    PIPELINE_VERSION = "pipeline_version"
    PROCESS_DATE = "process_date"
    PROTOCOL_NAME = "protocol_name"
    RAMP_TIME = "ramp_time"
    REFERENCE_DATA_KEY = "reference_data_key"
    REGISTRATION_KEY = "registration_key"
    REMOVEOS = "removeos"
    REMOVE_NOISE = "remove_noise"
    SCAN_DATE = "scan_date"
    SEGMENTATION_KEY = "segmentation_key"
    SHAPE_FIDS = "shape_fids"
    SHAPE_IMAGE = "shape_image"
    SLICE_THICKNESS = "slice_thickness"
    SOFTWARE_VERSION = "software_version"
    SUBJECT_ID = "subject_id"
    T2_CORRECTION_FACTOR_MEMBRANE = "t2_correction_factor_membrane"
    T2_CORRECTION_FACTOR_RBC = "t2_correction_factor_rbc"
    TE90 = "te90"
    TR = "tr"
    TR_DIS = "tr_dis"
    TRAJ = "traj"
    SYSTEM_VENDOR = "system_vendor"


class CNNPaths(object):
    """Paths to saved model files."""


class ImageType(enum.Enum):
    """Segmentation flags."""

    VENT = "vent"
    UTE = "ute"


class SegmentationKey(enum.Enum):
    """Segmentation flags."""

    CNN_VENT = "cnn_vent"
    CNN_PROTON = "cnn_proton"
    MANUAL_VENT = "manual_vent"
    MANUAL_PROTON = "manual_proton"
    SKIP = "skip"
    THRESHOLD_VENT = "threshold_vent"
    THRESHOLD_FRACTIONAL_VENTILATION = "threshold_fractional_ventilation"


class RegistrationKey(enum.Enum):
    """Registration flags.

    Defines how and if registration is performed. Options:
    PROTON2GAS: Register ANTs to register proton image (moving) to gas image (fixed).
        Also uses the transformation and applies on the mask if segmented on proton
        image.
    MASK2GAS: Register ANTs to register mask (moving) to gas image (fixed).
        Also uses the transformation and applies on the proton image.
    MANUAL: Read in Nifti file of manually registered proton image.
    SKIP: Skip registration entirely.
    """

    MANUAL = "manual"
    MASK2GAS = "mask2gas"
    PROTON2GAS = "proton2gas"
    SKIP = "skip"


class BiasfieldKey(enum.Enum):
    """Biasfield correction flags.

    Defines how and if biasfield correction is performed. Options:
    N4ITK: Use N4ITK bias field correction.
    SKIP: Skip bias field ocrrection entirely.
    """

    N4ITK = "n4itk"
    SKIP = "skip"
    RF_DEPOLARIZATION = "rf_depolarization"


class ReconKey(enum.Enum):
    """Reconstruction flags.

    Options:
    ROBERTSON: scott recon
    PLUMMER: joey p. recon
    """

    ROBERTSON = "robertson"
    PLUMMER = "plummer"


class HbCorrectionKey(enum.Enum):
    """Hb correction flags.

    Defines what level of Hb correction to apply to dissolved-phase signal. Options:
    NONE: Apply no hb correction
    RBC_AND_MEMBRANE: Apply Hb correction to both RBC and membrane signals
    RBC_ONLY: Apply Hb correction only to RBC signal
    """

    NONE = "none"
    RBC_AND_MEMBRANE = "rbc_and_membrane"
    RBC_ONLY = "rbc_only"


class ReferenceDataKey(enum.Enum):
    """Reference data flags.

    Defines which reference data to use. Options:
    DUKE_REFERENCE: Reference data for 218 or 208 ppm dissolved-phase rf excitation
    MANUAL_REFERENCE: Use when manualy adjusting default reference data
    """

    DUKE_REFERENCE = "duke_reference"
    MANUAL_REFERENCE = "manual_reference"


class Institution(enum.Enum):
    """Institution name."""

    DUKE = "duke"
    UVA = "uva"
    CCHMC = "cchmc"
    IOWA = "university of iowa"


class SystemVendor(enum.Enum):
    """Scanner system_vendor."""

    SIEMENS = "SIEMENS"
    GE = "GE"
    PHILIPS = "Philips"


class TrajType(object):
    """Trajectory type."""

    SPIRAL = "spiral"
    HALTON = "halton"
    HALTONSPIRAL = "haltonspiral"
    SPIRALRANDOM = "spiralrandom"
    ARCHIMEDIAN = "archimedian"
    GOLDENMEAN = "goldenmean"


class Orientation(object):
    """Image orientation."""

    CORONAL = "coronal"
    AXIAL = "axial"
    TRANSVERSE = "transverse"
    NONE = "none"


class DCFSpace(object):
    """Defines the DCF space."""

    GRIDSPACE = "gridspace"
    DATASPACE = "dataspace"


class Methods(object):
    """Defines the method to calculate the RBC oscillation image."""

    ELEMENTWISE = "elementwise"
    MEAN = "mean"
    SMOOTH = "smooth"
    BSPLINE = "bspline"


class BinningMethods(object):
    """Define the method to preprocess and bin RBC oscillation image."""

    BANDPASS = "bandpass"
    FIT_SINE = "fitsine"
    NONE = "none"
    THRESHOLD_STRETCH = "threshold_stretch"
    THRESHOLD = "threshold"
    PEAKS = "peaks"


class StatsIOFields(object):
    """Statistic IO Fields."""

    INFLATION = "inflation"
    INFLATION_PCT='inflation_percentage'
    INFLATION_AVG='inflation_avg'
    INFLATION_DISPLAY='inflation_display'
    RBC_M_RATIO = "rbc_m_ratio"
    RBC_SNR = "rbc_snr"
    MEMBRANE_SNR = "membrane_snr"
    VENT_SNR = "vent_snr"
    RBC_HIGH_PCT = "rbc_high_pct"
    RBC_LOW_PCT = "rbc_low_pct"
    RBC_DEFECT_PCT = "rbc_defect_pct"
    MEMBRANE_HIGH_PCT = "membrane_high_pct"
    MEMBRANE_LOW_PCT = "membrane_low_pct"
    MEMBRANE_DEFECT_PCT = "membrane_defect_pct"
    VENT_HIGH_PCT = "vent_high_pct"
    VENT_LOW_PCT = "vent_low_pct"
    VENT_DEFECT_PCT = "vent_defect_pct"
    RBC_MEAN = "rbc_mean"
    MEMBRANE_MEAN = "membrane_mean"
    VENT_MEAN = "vent_mean"
    RBC_MEDIAN = "rbc_median"
    MEMBRANE_MEDIAN = "membrane_median"
    VENT_MEDIAN = "vent_median"
    RBC_STDDEV = "rbc_stddev"
    MEMBRANE_STDDEV = "membrane_stddev"
    VENT_STDDEV = "vent_stddev"
    DLCO_EST = "dlco_est"
    KCO_EST = "kco_est"
    RDP_BA = "rdp_ba"
    ALVEOLAR_VOLUME = "alveolar_volume"


class VENTHISTOGRAMFields(object):
    """Ventilation histogram fields."""

    COLOR = (0.4196, 0.6824, 0.8392)
    XLIM = 1.0
    YLIM = 0.25
    NUMBINS = 50
    XTICKS = np.linspace(0, XLIM, 4)
    YTICKS = np.linspace(0, YLIM, 5)
    XTICKLABELS = ["{:.2f}".format(x) for x in XTICKS]
    YTICKLABELS = ["{:.2f}".format(x) for x in YTICKS]
    TITLE = "Ventilation"


class RBCHISTOGRAMFields(object):
    """Ventilation histogram fields."""

    COLOR = (247.0 / 255, 96.0 / 255, 111.0 / 255)
    XLIM = 0.012
    YLIM = 0.1
    NUMBINS = 50
    XTICKS = np.linspace(0, XLIM, 4)
    YTICKS = np.linspace(0, YLIM, 5)
    XTICKLABELS = ["{:.2f}".format(x * 1e2) for x in XTICKS]
    YTICKLABELS = ["{:.2f}".format(x) for x in YTICKS]
    TITLE = "RBC:Gas x 100"


class MEMBRANEHISTOGRAMFields(object):
    """Membrane histogram fields."""

    COLOR = (0.4, 0.7608, 0.6471)
    XLIM = 0.025
    YLIM = 0.18
    NUMBINS = 70
    XTICKS = np.linspace(0, XLIM, 4)
    YTICKS = np.linspace(0, YLIM, 5)
    XTICKLABELS = ["{:.2f}".format(x * 1e2) for x in XTICKS]
    YTICKLABELS = ["{:.2f}".format(x) for x in YTICKS]
    TITLE = "Membrane:Gas x 100"


class PDFOPTIONS(object):
    """PDF Options dict."""

    VEN_PDF_OPTIONS = {
        "page-width": 256,  # 320,
        "page-height": 160,  # 160,
        "margin-top": 1,
        "margin-right": 0.1,
        "margin-bottom": 0.1,
        "margin-left": 0.1,
        "dpi": 300,
        "encoding": "UTF-8",
        "enable-local-file-access": None,
    }


class NormalizationMethods(object):
    """Image normalization methods."""

    MAX = "max"
    PERCENTILE_MASKED = "percentile_masked"
    PERCENTILE = "percentile"
    MEAN = "mean"
    FRAC_VENT = "frac_vent"


class CMAP(object):
    """Maps of binned values to color values."""

    RBC_BIN2COLOR = {
        0: [0, 0, 0],
        1: [1, 0, 0],
        2: [1, 0.7143, 0],
        3: [0.4, 0.7, 0.4],
        4: [0, 1, 0],
        5: [0, 0.57, 0.71],
        6: [0, 0, 1],
    }

    VENT_BIN2COLOR = {
        0: [0, 0, 0],
        1: [1, 0, 0],
        2: [1, 0.7143, 0],
        3: [0.4, 0.7, 0.4],
        4: [0, 1, 0],
        5: [0, 0.57, 0.71],
        6: [0, 0, 1],
    }

    MEMBRANE_BIN2COLOR = {
        0: [0, 0, 0],
        1: [1, 0, 0],
        2: [1, 0.7143, 0],
        3: [0.4, 0.7, 0.4],
        4: [0, 1, 0],
        5: [184.0 / 255.0, 226.0 / 255.0, 145.0 / 255.0],
        6: [243.0 / 255.0, 205.0 / 255.0, 213.0 / 255.0],
        7: [225.0 / 255.0, 129.0 / 255.0, 162.0 / 255.0],
        8: [197.0 / 255.0, 27.0 / 255.0, 125.0 / 255.0],
    }


class HbCorrection(object):
    """Coefficients for hb correction scaling factor equations.

    Reference: https://onlinelibrary.wiley.com/doi/10.1002/mrm.29712
    """

    HB_REF = 14.0  # reference hb value in g/dL
    R1 = 0.288  # coefficient of rbc hb correction equation
    M1 = 0.029  # first coefficient of membrane hb correction equation
    M2 = 0.011  # second coefficient of membrane hb correction equation


class ContrastLabels(object):
    """Numbers for labelling type of FID acquisition excitation."""

    PROTON = 0  # proton acquisition
    GAS = 1  # gas phase 129Xe acquisition
    DISSOLVED = 2  # dissolved phase 129Xe acquisition


class BonusSpectraLabels(object):
    """Numbers for labelling if FID acquisition is part of bonus spectra."""

    NOT_BONUS = 0  # not part of bonus spectra
    BONUS = 1  # part of bonus spectra


class PipelineVersion(object):
    """Pipeline version."""

    VERSION_NUMBER = 4


class ReferenceDistribution(object):
    """Reference distributions for binning based on RF excitation.

    Reference: Sup's reference distribution paper when published """

    REFERENCE_218_PPM = {
    "title": "REFERENCE_218_PPM",
    "threshold_vent": [0.3891, 0.5753, 0.7203, 0.8440, 0.9539],
    "threshold_fractional_ventilation": [0.126229, 0.198441, 0.271045, 0.343936, 0.417054],
    "threshold_rbc": [0.001393, 0.002891, 0.004772, 0.006991, 0.009518],
    "threshold_membrane": [0.004881, 0.006522, 0.008603, 0.011216, 0.014466, 0.018471, 0.023370],
    "reference_fit_vent": (0.04074, 0.7085, 0.1408),
    "reference_fit_rbc": (0.06106, 0.004942, 0.002060),
    "reference_fit_membrane": (0.0700, 0.008871, 0.002420),
    "reference_stats": {
        "vent_defect_avg": "2",
        "vent_defect_std": "",
        "vent_low_avg": "14",
        "vent_low_std": "",
        "vent_high_avg": "16",
        "vent_high_std": "",
        "membrane_defect_avg": "2",
        "membrane_defect_std": "0",
        "membrane_low_avg": "14",
        "membrane_low_std": "0",
        "membrane_high_avg": "2",
        "membrane_high_std": "0",
        "rbc_defect_avg": "2",
        "rbc_defect_std": "",
        "rbc_low_avg": "14",
        "rbc_low_std": "",
        "rbc_high_avg": "16",
        "rbc_high_std": "",
        "rbc_m_ratio_avg": "0.55",
        "rbc_m_ratio_std": "0.12",
        "inflation_avg": "3.4",
        "inflation_std": "0.33",
        "inflation_percentage":"0.0",
        "inflation_display":"0.0",
        }
    }

    REFERENCE_208_PPM = {
        "title": "REFERENCE_208_PPM",
        "threshold_vent": [0.3891, 0.5753, 0.7203, 0.8440, 0.9539],
        "threshold_fractional_ventilation": [0.126229, 0.198441, 0.271045, 0.343936, 0.417054],
        "threshold_rbc": [0.001351, 0.002804, 0.004629, 0.006781, 0.009232],
        "threshold_membrane": [0.005320, 0.007108, 0.009377, 0.012224, 0.015766, 0.020132, 0.025471],
        "reference_fit_vent": (0.04074, 0.7085, 0.1408),
        "reference_fit_rbc": (0.06106, 0.004794, 0.001998),
        "reference_fit_membrane": (0.0700, 0.009668, 0.002638),
        "reference_stats": {
            "vent_defect_avg": "2",
            "vent_defect_std": "",
            "vent_low_avg": "14",
            "vent_low_std": "",
            "vent_high_avg": "16",
            "vent_high_std": "",
            "membrane_defect_avg": "2",
            "membrane_defect_std": "0",
            "membrane_low_avg": "14",
            "membrane_low_std": "0",
            "membrane_high_avg": "2",
            "membrane_high_std": "0",
            "rbc_defect_avg": "2",
            "rbc_defect_std": "",
            "rbc_low_avg": "14",
            "rbc_low_std": "",
            "rbc_high_avg": "16",
            "rbc_high_std": "",
            "rbc_m_ratio_avg": "0.49",
            "rbc_m_ratio_std": "0.11",
            "inflation_avg": "3.4",
            "inflation_std": "0.33",
            "inflation_percentage":"0.0",
            "inflation_display":"0.0",
            }
        }
    
    REFERENCE_MANUAL = {
        "title": "MANUAL",
        "threshold_vent": [0.3891, 0.5753, 0.7203, 0.8440, 0.9539],
        "threshold_fractional_ventilation": [0.126229, 0.198441, 0.271045, 0.343936, 0.417054],
        "threshold_rbc": [0.001393, 0.002891, 0.004772, 0.006991, 0.009518],
        "threshold_membrane": [0.004881, 0.006522, 0.008603, 0.011216, 0.014466, 0.018471, 0.023370],
        "reference_fit_vent": (0.04074, 0.7085, 0.1408),
        "reference_fit_rbc": (0.06106, 0.004942, 0.002060),
        "reference_fit_membrane": (0.0700, 0.008871, 0.002420),
        "reference_stats": {
            "vent_defect_avg": "2.15",
            "vent_defect_std": "",
            "vent_low_avg": "13.59",
            "vent_low_std": "",
            "vent_high_avg": "15.74",
            "vent_high_std": "",
            "membrane_defect_avg": "2.15",
            "membrane_defect_std": "0",
            "membrane_low_avg": "13.59",
            "membrane_low_std": "0",
            "membrane_high_avg": "2.28",
            "membrane_high_std": "0",
            "rbc_defect_avg": "2.15",
            "rbc_defect_std": "",
            "rbc_low_avg": "13.59",
            "rbc_low_std": "",
            "rbc_high_avg": "15.74",
            "rbc_high_std": "",
            "rbc_m_ratio_avg": "0.59",
            "rbc_m_ratio_std": "0.12",
            "inflation_avg": "3.4",
            "inflation_std": "0.33",
            "inflation_percentage":"0.0",
            "inflation_display":"0.0",
            }
        }

