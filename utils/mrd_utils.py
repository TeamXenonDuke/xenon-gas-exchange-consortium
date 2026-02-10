"""MRD util functions."""
import logging
import sys
from typing import Any, Dict

import ismrmrd
import numpy as np

sys.path.append("..")
from utils import constants


def get_subject_id(
    header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader,
) -> str:
    """Get subject ID from the MRD header.

    Args
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        subject ID (str)
    """
    return header.subjectInformation.patientID


def get_system_vendor(
    header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader,
) -> str:
    """Get system vendor from the MRD header.

    Args
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        system vendor (str)
    """
    return header.acquisitionSystemInformation.systemVendor


def get_institution_name(
    header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader,
) -> str:
    """Get the institution name from the MRD header.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        institution name (str)
    """
    return header.acquisitionSystemInformation.institutionName.lower()


def get_field_strength(
    header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader,
) -> float:
    """Get the magnetic field strength from the MRD header in Tesla.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        magnetic field strength in Tesla (float)
    """
    return header.acquisitionSystemInformation.systemFieldStrength_T


def get_sample_time(dataset: ismrmrd.hdf5.Dataset) -> float:
    """Get the sample time from the MRD data set object.

    Sample time is stored for every FID acquisition. Assumes sample time is the same
    for each acquisition and reads sample time from header of first acquisition.

    Args:
        dataset (ismrmrd.hdf5.Dataset): MRD data object
    Returns:
        float: dwell time in seconds
    """
    acq_header = dataset.read_acquisition(0).getHead()
    return acq_header.sample_time_us * 1e-6


def get_dyn_fids(dataset: ismrmrd.hdf5.Dataset, n_skip_end: int = 20) -> np.ndarray:
    """Get the dissolved phase FIDS used for dyn. spectroscopy from mrd object.

    Args:
        header (ismrmrd.hdf5.Dataset): MRD dataset
        n_skip_end: number of fids to skip from the end. Usually they are calibration
            frames.
    Returns:
        dissolved phase FIDs in shape (number of points in ray, number of projections).
    """
    raw_fids = []
    n_projections = dataset.number_of_acquisitions() - n_skip_end
    for i in range(0, int(n_projections)):  # type: ignore
        raw_fids.append(dataset.read_acquisition(i).data[0].flatten())
    return np.transpose(np.asarray(raw_fids))


def get_excitation_freq(
    header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader,
) -> float:
    """Get the excitation frequency from the MRD header.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header

    Returns:
        excitation frequency in ppm (float)
    """
    var_names = [
        header.userParameters.userParameterLong[i].name
        for i in range(len(header.userParameters.userParameterLong))
    ]

    try:
        freq_excitation_hz = float(
            header.userParameters.userParameterLong[
                var_names.index(constants.IOFields.XE_DISSOLVED_OFFSET_FREQUENCY)
            ].value
        )
        freq_excitation_ppm = (freq_excitation_hz) / (
        get_field_strength(header) * constants.GRYOMAGNETIC_RATIO
    )

    except:
        freq_excitation_ppm = float(
            header.userParameters.userParameterDouble[0].value
        )



    return freq_excitation_ppm


def get_center_freq(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
    """Get the center frequency from the MRD header.

    See: https://mriquestions.com/center-frequency.html for definition of center freq.
    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header

    Returns:
        center frequency in MHz (float)
    """

    var_names = [
        header.userParameters.userParameterLong[i].name
        for i in range(len(header.userParameters.userParameterLong))
    ]

    try:
        xe_center_frequency = float(
            header.userParameters.userParameterLong[
                var_names.index(constants.IOFields.XE_CENTER_FREQUENCY)
            ].value
        )
    except:
        xe_center_frequency = float(
            header.userParameters.userParameterLong[0].value
        )
    
    return xe_center_frequency * 1e-6


def get_TR(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
    """Get the TR from the MRD header.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header

    Returns:
        float: TR in seconds
    """
    return 1e-3 * header.sequenceParameters.TR[0]


def get_scan_date(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> str:
    """Get the scan date from the MRD header in MM-DD-YYYY format.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header

    Returns:
        str: scan date in MM-DD-YYYY format.
    """



    try:
        xml_date = header.studyInformation.studyDate
    except:
        xml_date =  header.measurementInformation.seriesDate
    YYYY = str(xml_date[0])
    MM = str(xml_date[1])
    DD = str(xml_date[2])
    return YYYY + "-" + MM + "-" + DD


def get_flipangle_dissolved(
    header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader,
) -> float:
    """Get the dissolved phase flip angle in degrees.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        flip angle in degrees (float)
    """
    return header.sequenceParameters.flipAngle_deg[1]


def get_flipangle_gas(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
    """Get the gasd phase flip angle in degrees.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        flip angle in degrees (float)
    """
    return header.sequenceParameters.flipAngle_deg[0]

def get_prep_pulses(
    header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader,
) -> str:
    """Get the prep pulse.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        prep_pulse (string)
    """
    var_names = [
        up.name for up in header.userParameters.userParameterString
    ]
    # Check if PREP_PULSES exists
    if constants.IOFields.PREP_PULSES not in var_names:
        return "prep_pulses does not exist in the MRD file."

    prep_pulses = str(
        header.userParameters.userParameterString[
            var_names.index(constants.IOFields.PREP_PULSES)
        ]
        .value
    )
    return prep_pulses

def get_FOV(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
    """Get the FOV in cm.

    For now, assumes same FOV in all three dimensions.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        FOV in cm (float)
    """
    return header.encoding[0].reconSpace.fieldOfView_mm.x * 1e-1


def get_orientation(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> str:
    """Get the orientation of the image.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        orientation (str): orientation of reconstructed image (coronal, transverse, axial).
            Returns coronal if not found.
    """
    orientation = ""
    system_vendor = get_system_vendor(header)

    try:
        var_names = [
            header.userParameters.userParameterString[i].name
            for i in range(len(header.userParameters.userParameterString))
        ]
        orientation = header.userParameters.userParameterString[
            var_names.index(constants.IOFields.ORIENTATION)
        ].value
    except:
        logging.info("Unable to find orientation from twix object, returning coronal.")
    
    orientation = orientation.lower() if orientation else ""

    supported_vendors = {
        constants.SystemVendor.SIEMENS.value,
        constants.SystemVendor.PHILIPS.value,
        constants.SystemVendor.GE.value,
    }
    valid_orientations = {
        constants.Orientation.CORONAL,
        constants.Orientation.AXIAL,
        constants.Orientation.TRANSVERSE,
        constants.Orientation.SAGITTAL,
    }
    if system_vendor in supported_vendors and orientation in valid_orientations:
        return orientation
    return constants.Orientation.CORONAL


def get_protocol_name(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> str:
    """Get the protocol name.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        protocol name. Returns "unknown" if not found.
    """
    try:
        return str(header.measurementInformation.protocolName)
    except:
        return "unknown"


def get_ramp_time(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
    """Get the ramp time in micro-seconds.

    See: https://mriquestions.com/gradient-specifications.html

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        ramp time in us
    """
    var_names = [
        header.encoding[0].trajectoryDescription.userParameterLong[i].name
        for i in range(len(header.encoding[0].trajectoryDescription.userParameterLong))
    ]
    ramp_time = float(
        header.encoding[0]
        .trajectoryDescription.userParameterLong[
            var_names.index(constants.IOFields.RAMP_TIME)
        ]
        .value
    )
    return ramp_time


def get_TE90(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
    """Get the TE90 in seconds.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        TE90 in seconds (float)
    """
    return header.sequenceParameters.TE[0] * 1e-3


def get_TR_dissolved(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
    """Get the TR in seconds for dissolved phase.

    The dissolved phase TR is defined to be the time between two consecutive dissolved
    phase-FIDS. This is different from the TR in the mrd header as the mrd header
    provides the dissolved and gas phase interleaf durations.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        TR in seconds (float)
    """
    try:
        tr_gas_to_dissolved = header.sequenceParameters.TR[0]
        tr_dissolved_to_gas = header.sequenceParameters.TR[1]
        return (tr_gas_to_dissolved + tr_dissolved_to_gas) * 1e-3
    except:
        tr_gas_to_dissolved = header.sequenceParameters.TR[0]
        return (tr_gas_to_dissolved + tr_gas_to_dissolved ) * 1e-3


def get_gx_data(dataset: ismrmrd.hdf5.Dataset, multi_echo: bool) -> Dict[str, Any]:
    """Get the FID acquisition data from dixon MRD file.

    Args:
        dataset: ismrmrd dataset object
    Returns:
        a dictionary containing
            - all raw fids of shape (number of projections for gas and dissolved phase combined,
                number of points in ray)
            - gas phase fids in shape (number of projections, number of points in ray)
            - dissolved phase fids in shape (number fo projections, number of points in ray)
            - k space trajectory of gas and dissolved acquisitions (for standard 1 pt Dixon
                these are the same)
    """
    # get the raw FIDs, contrast labels, and bonus spectra labels
    raw_fids = []
    contrast_labels = []
    bonus_spectra_labels = []
    set_labels = []
    set_included = True
    n_projections = dataset.number_of_acquisitions()
    for i in range(0, int(n_projections)):
        acquisition_header = dataset.read_acquisition(i).getHead()
        raw_fids.append(dataset.read_acquisition(i).data[0].flatten())
        contrast_labels.append(acquisition_header.idx.contrast)
        bonus_spectra_labels.append(acquisition_header.measurement_uid)
        try:
            set_labels.append(acquisition_header.idx.set)
        except:
            logging.info("#############################################")
            logging.info("Unable to find set paramater from mrd object.")
            set_included = False
    raw_fids = np.asarray(raw_fids)
    contrast_labels = np.asarray(contrast_labels)
    bonus_spectra_labels = np.asarray(bonus_spectra_labels)
    set_labels = np.asarray(set_labels)

    # remove bonus spectra
    raw_fids_truncated = raw_fids[
        bonus_spectra_labels == constants.BonusSpectraLabels.NOT_BONUS, :
    ]
    contrast_labels_truncated = contrast_labels[
        bonus_spectra_labels == constants.BonusSpectraLabels.NOT_BONUS
    ]
    set_labels_truncated = set_labels[
        bonus_spectra_labels == constants.BonusSpectraLabels.NOT_BONUS
    ]

    # get the trajectories
    raw_traj = np.empty((raw_fids_truncated.shape[0], raw_fids_truncated.shape[1], 3))
    for i in range(0, raw_fids_truncated.shape[0]):
        raw_traj[i, :, :] = dataset.read_acquisition(i).traj


    if(set_included):
        unique_set_labels = np.unique(set_labels_truncated)
        
        gas_fids_all = []
        dis_fids_all = []
        gas_trajectories_all = []
        dis_trajectories_all = []

        for set_label in unique_set_labels:
            gas_fids_set = raw_fids_truncated[
                (contrast_labels_truncated == constants.ContrastLabels.GAS) & (set_labels_truncated == set_label)
            ]
            dis_fids_set = raw_fids_truncated[
                (contrast_labels_truncated == constants.ContrastLabels.DISSOLVED) & (set_labels_truncated == set_label)
            ]
            gas_traj_set = raw_traj[
                (contrast_labels_truncated == constants.ContrastLabels.GAS) & (set_labels_truncated == set_label)
            ]
            dis_traj_set = raw_traj[
                (contrast_labels_truncated == constants.ContrastLabels.DISSOLVED) & (set_labels_truncated == set_label)
            ]

            # Append gas_fids_set, dis_fids_set, and traj_set with an additional axis to represent the set dimension
            gas_fids_all.append(np.expand_dims(gas_fids_set, axis=-1))
            dis_fids_all.append(np.expand_dims(dis_fids_set, axis=-1))
            gas_trajectories_all.append(np.expand_dims(gas_traj_set, axis=-1))
            dis_trajectories_all.append(np.expand_dims(dis_traj_set, axis=-1))

        gas_fids_all = np.concatenate(gas_fids_all, axis=-1)
        dis_fids_all = np.concatenate(dis_fids_all, axis=-1)
        gas_trajectories_all = np.concatenate(gas_trajectories_all, axis=-1)
        dis_trajectories_all = np.concatenate(dis_trajectories_all, axis=-1)
      
        if (multi_echo):
            all_traj = [gas_trajectories_all , dis_trajectories_all];
            return {
                constants.IOFields.FIDS: raw_fids_truncated,
                constants.IOFields.FIDS_GAS: gas_fids_all,
                constants.IOFields.FIDS_DIS: dis_fids_all,
                constants.IOFields.TRAJ: all_traj,
            }
        else:
            all_traj = [gas_trajectories_all[...,0] , dis_trajectories_all[...,0]];
            return {
                constants.IOFields.FIDS: raw_fids_truncated,
                constants.IOFields.FIDS_GAS: gas_fids_all[...,0],
                constants.IOFields.FIDS_DIS: dis_fids_all[...,0],
                constants.IOFields.TRAJ: all_traj,
            }

    else:
        gas_traj = raw_traj[
                contrast_labels_truncated == constants.ContrastLabels.GAS, :, :
            ];

        dis_traj = raw_traj[
                contrast_labels_truncated == constants.ContrastLabels.DISSOLVED, :, :
            ];

        all_traj = [gas_traj , dis_traj];

        return {
            constants.IOFields.FIDS: raw_fids_truncated,
            constants.IOFields.FIDS_GAS: raw_fids_truncated[
                contrast_labels_truncated == constants.ContrastLabels.GAS, :
            ],
            constants.IOFields.FIDS_DIS: raw_fids_truncated[
                contrast_labels_truncated == constants.ContrastLabels.DISSOLVED, :
            ],
            constants.IOFields.TRAJ: all_traj,
        }

def get_gx_data_cinci(dataset: ismrmrd.hdf5.Dataset) -> Dict[str, Any]:

    # Reading the dataset
    dset = dataset
    nFids = dset.number_of_acquisitions()

    # Getting Information from the Header associated with the scan
    header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
  

    # Prepare the K-space and trajectories >>>>>>>>>>>>>>
    #%% ======================= Reshaping K-space =============================
    k_space_reshaped_all = []
    nFrames = dset.number_of_acquisitions()
    npts_fid = dset.read_acquisition(0).data[0].shape[0]  # 64 for cchmc

    for i in range(0, nFrames):
        k_space_reshaped = dset.read_acquisition(i).data[0].reshape(1, npts_fid)
        k_space_reshaped_all.append(k_space_reshaped)

    data_dixon = np.concatenate(k_space_reshaped_all, axis=0)

    # Separating Gas and Dissolved Fids
    data_gas = data_dixon[0::2, :]
    nFrames_gas_all = data_gas.shape[0]

    data_dis_all = data_dixon[1::2, :]
    data_dis = data_dis_all



    #%% ===================== Reshaping trajectories ===========================
    traj_list_all = []
    traj_all = np.empty((nFrames, npts_fid, 3))

    for i in range(0, nFrames):
        traj_all[i, :, :] = dset.read_acquisition(i).traj

    # %% Separate Gas and Dissolved Trajectories - here gas and dissolved trajs are same
    traj_gas = traj_all[0::2, :].astype(np.float64)
    traj_dis = traj_all[1::2, :].astype(np.float64)

    return {
        constants.IOFields.FIDS: data_dixon,
        constants.IOFields.FIDS_GAS: data_gas, 
        constants.IOFields.FIDS_DIS: data_dis,
        constants.IOFields.TRAJ: [traj_gas,traj_dis],
    }

def get_ute_data(dataset: ismrmrd.hdf5.Dataset) -> Dict[str, Any]:
    """Get the FID acquisition data from proton MRD file.

    Args:
        dataset: ismrmrd dataset object
    Returns:
        a dictionary containing
            - all proton fids of shape (number of projections, number of points in ray)
            - k space trajectory of proton acquisitions
    """
    # get the raw FIDs, contrast labels, and bonus spectra labels
    raw_fids = []
    contrast_labels = []
    bonus_spectra_labels = []
    n_projections = dataset.number_of_acquisitions()
    for i in range(0, int(n_projections)):
        acquisition_header = dataset.read_acquisition(i).getHead()
        raw_fids.append(dataset.read_acquisition(i).data[0].flatten())
        contrast_labels.append(acquisition_header.idx.contrast)
        bonus_spectra_labels.append(acquisition_header.measurement_uid)
    raw_fids = np.asarray(raw_fids)
    contrast_labels = np.asarray(contrast_labels)
    bonus_spectra_labels = np.asarray(bonus_spectra_labels)

    # remove bonus spectra
    raw_fids_truncated = raw_fids[
        bonus_spectra_labels == constants.BonusSpectraLabels.NOT_BONUS, :
    ]
    contrast_labels_truncated = contrast_labels[
        bonus_spectra_labels == constants.BonusSpectraLabels.NOT_BONUS
    ]

    # get the trajectories
    raw_traj = np.empty((raw_fids_truncated.shape[0], raw_fids_truncated.shape[1], 3))
    for i in range(0, raw_fids_truncated.shape[0]):
        raw_traj[i, :, :] = dataset.read_acquisition(i).traj

    return {
        constants.IOFields.FIDS: raw_fids_truncated[
            contrast_labels_truncated == constants.ContrastLabels.PROTON, :
        ],
        constants.IOFields.TRAJ: raw_traj[
            contrast_labels_truncated == constants.ContrastLabels.PROTON, :, :
        ],
    }
