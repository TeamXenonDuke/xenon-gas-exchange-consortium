"""Import and export util functions."""

import os
import sys

sys.path.append("..")
import csv
import glob
import shutil
from typing import Any, Dict, List, Optional, Tuple

import ismrmrd
import mapvbvd
import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import scipy.io as sio
import logging
import ml_collections
from ml_collections import config_dict

from utils import constants, img_utils, mrd_utils, twix_utils


def import_np(path: str) -> np.ndarray:
    """Import npy file to np.ndarray.

    Args:
        path: str file path of npy file
    Returns:
        np.ndarray loaded from npy file
    """
    return np.load(path)


def import_nii(path: str) -> np.ndarray:
    """Import image as np.ndarray.

    Args:
        path: str file path of nifti file
    Returns:
        np.ndarray loaded from nifti file
    """
    return nib.load(path).get_fdata()


def import_mat(path: str) -> Dict[str, Any]:
    """Import  matlab file as dictionary.

    Args:
        path: str file path of matlab file
    Returns:
        dictionary loaded from matlab file
    """
    return sio.loadmat(path)


def import_matstruct_to_dict(struct: np.ndarray) -> Dict[str, Any]:
    """Import matlab  as dictionary.

    Args:
        path: str file path of matlab file
    Returns:
        dictionary loaded from matlab file
    """
    out_dict = {}

    for field in struct.dtype.names:
        value = struct[field].flatten()[0]
        if isinstance(value[0], str):
            # strings
            out_dict[field] = str(value[0])
        elif len(value) == 1:
            # numbers
            out_dict[field] = value[0][0]
        else:
            # arrays
            out_dict[field] = np.asarray(value)
    return out_dict


def get_dyn_twix_files(path: str) -> str:
    """Get list of dynamic spectroscopy twix files.

    Args:
        path: str directory path of twix files
    Returns:
        str file path of twix file
    """
    try:
        return (
            glob.glob(os.path.join(path, "**cali**.dat"))
            + glob.glob(os.path.join(path, "**dynamic**.dat"))
            + glob.glob(os.path.join(path, "**Dynamic**.dat"))
            + glob.glob(os.path.join(path, "**dyn**.dat"))
        )[0]
    except:
        raise ValueError("Can't find twix file in path.")


def get_dis_twix_files(path: str) -> str:
    """Get list of gas exchange twix files.

    Args:
        path: str directory path of twix files
    Returns:
        str file path of twix file
    """
    try:
        return (
            glob.glob(os.path.join(path, "**dixon***.dat"))
            + glob.glob(os.path.join(path, "**Dixon***.dat"))
        )[0]
    except:
        raise ValueError("Can't find twix file in path.")


def get_ute_twix_files(path: str) -> str:
    """Get list of UTE twix files.

    Args:
        path: str directory path of twix files
    Returns:
        str file path of twix file
    """
    try:
        return (
            glob.glob(os.path.join(path, "**1H***.dat"))
            + glob.glob(os.path.join(path, "**BHUTE***.dat"))
            + glob.glob(os.path.join(path, "**ute***.dat"))
            + glob.glob(os.path.join(path, "**h_radial***.dat"))
        )[0]
    except:
        raise ValueError("Can't find twix file in path.")


def get_dyn_mrd_files(path: str) -> str:
    """Get list of dynamic spectroscopy MRD files.

    Args:
        path: str directory path of MRD files
    Returns:
        str file path of MRD file
    """
    try:
        return (
            glob.glob(os.path.join(path, "**Calibration***.h5"))
            + glob.glob(os.path.join(path, "**calibration***.h5")) 
            + glob.glob(os.path.join(path, "**Calibration***.mrd"))
            + glob.glob(os.path.join(path, "**calibration***.mrd"))
        )[0]
    except:
        raise ValueError("Can't find MRD file in path.")


def get_dis_mrd_files(path: str) -> str:
    """Get list of gas exchange MRD files.

    Args:
        path: str directory path of MRD files
    Returns:
        str file path of MRD file
    """
    try:
        return (
            glob.glob(os.path.join(path, "**dixon***.h5"))
            + glob.glob(os.path.join(path, "**dixon***.mrd"))
        )[0]
    except:
        raise ValueError("Can't find MRD file in path.")


def get_ute_mrd_files(path: str) -> str:
    """Get list of proton MRD files.

    Args:
        path (str): directory path of MRD files
    Returns:
        str file path of MRD file
    """
    try:
        return (
            glob.glob(os.path.join(path, "**proton***.h5"))
            + glob.glob(os.path.join(path, "**proton***.mrd"))
        )[0]
    except:
        raise ValueError("Can't find MRD file in path.")


def get_mat_file(path: str) -> str:
    """Get list of mat file of reconstructed images.

    Args:
        path: str directory path of mat file.
    Returns:
        str file path of mat file
    """
    try:
        return (glob.glob(os.path.join(path, "**.mat")))[0]
    except:
        raise ValueError("Can't find mat file in path.")


def read_dyn_twix(path: str) -> Dict[str, Any]:
    """Read dynamic spectroscopy twix file.

    Args:
        path: str file path of twix file
    Returns: dictionary containing data and metadata extracted from the twix file.
    This includes:
        1. scan date in MM-DD-YYY format.
        2. dwell time in seconds.
        3. TR in seconds.
        4. Center frequency in MHz.
        5. excitation frequency in ppm.
        6. dissolved phase FIDs in format (n_points, n_projections).
    """
    try:
        twix_obj = mapvbvd.mapVBVD(path)
    except:
        raise ValueError("Invalid twix file.")
    twix_obj.image.squeeze = True
    twix_obj.image.flagIgnoreSeg = True
    twix_obj.image.flagRemoveOS = False

    # Get scan information
    sample_time = twix_utils.get_sample_time(twix_obj=twix_obj)
    fids_dis = twix_utils.get_dyn_fids(twix_obj)
    xe_center_frequency = twix_utils.get_center_freq(twix_obj=twix_obj)
    xe_dissolved_offset_frequency = twix_utils.get_excitation_freq(twix_obj=twix_obj)
    scan_date = twix_utils.get_scan_date(twix_obj=twix_obj)
    tr = twix_utils.get_TR(twix_obj=twix_obj)

    return {
        constants.IOFields.SAMPLE_TIME: sample_time,
        constants.IOFields.FIDS_DIS: fids_dis,
        constants.IOFields.XE_CENTER_FREQUENCY: xe_center_frequency,
        constants.IOFields.XE_DISSOLVED_OFFSET_FREQUENCY: xe_dissolved_offset_frequency,
        constants.IOFields.SCAN_DATE: scan_date,
        constants.IOFields.TR: tr,
    }


def read_dis_twix(path: str, config: Optional[ml_collections.ConfigDict] = None) -> Dict[str, Any]:
    """Read 1-point dixon disssolved phase imaging twix file.

    Args:
        path: str file path of twix file
    Returns: dictionary containing data and metadata extracted from the twix file.
    This includes:
        - Age: Patient age as a float.
        - Sex: Patient sex as a string ("M" or "F").
        - Height: Patient height in centimeters as a float.
        - dwell time in seconds.
        - flip angle applied to dissolved phase in degrees.
        - flip angle applied to gas phase in degrees.
        - dissolved phase FIDs in format (n_projections, n_points).
        - gas phase FIDs in format (n_projections, n_points).
        - field of view in cm.
        - center frequency in MHz.
        - excitation frequency in ppm.
        - gradient delay (x, y, z) in microseconds.
        - number of frames (projections) used to calculate trajectory.
        - number of projections to skip at the beginning of the scan.
        - number of projections to skip at the end of the scan.
        - orientation of the scan.
        - protocol name
        - ramp time in microseconds.
        - scan date in YYYY-MM-DD format.
        - software version
        - TE90 in seconds.
        - TR in seconds.
    """
    try:
        twix_obj = mapvbvd.mapVBVD(path)
    except:
        raise ValueError("Invalid twix file.")
    twix_obj.image.squeeze = True
    twix_obj.image.flagIgnoreSeg = True
    twix_obj.image.flagRemoveOS = False

    data_dict = twix_utils.get_gx_data(twix_obj=twix_obj)
    filename = os.path.basename(path)

    if config or config.recon.del_x is constants.NONE:  # type: ignore
        logging.error("Gradient delay is not properly set in the config file")

    return {
        constants.IOFields.AGE: twix_utils.get_patient_age(twix_obj),
        constants.IOFields.SEX: twix_utils.get_patient_sex(twix_obj),
        constants.IOFields.HEIGHT: twix_utils.get_patient_height(twix_obj),
        constants.IOFields.SAMPLE_TIME: twix_utils.get_sample_time(twix_obj),
        constants.IOFields.FA_DIS: twix_utils.get_flipangle_dissolved(twix_obj),
        constants.IOFields.FA_GAS: twix_utils.get_flipangle_gas(twix_obj),
        constants.IOFields.FIELD_STRENGTH: twix_utils.get_field_strength(twix_obj),
        constants.IOFields.FIDS: data_dict[constants.IOFields.FIDS],
        constants.IOFields.FIDS_DIS: data_dict[constants.IOFields.FIDS_DIS],
        constants.IOFields.FIDS_GAS: data_dict[constants.IOFields.FIDS_GAS],
        constants.IOFields.FOV: twix_utils.get_FOV(twix_obj),
        constants.IOFields.XE_CENTER_FREQUENCY: twix_utils.get_center_freq(twix_obj),
        constants.IOFields.XE_DISSOLVED_OFFSET_FREQUENCY: twix_utils.get_excitation_freq(
            twix_obj
        ),
        constants.IOFields.GRAD_DELAY_X: config.recon.del_x,
        constants.IOFields.GRAD_DELAY_Y: config.recon.del_y,
        constants.IOFields.GRAD_DELAY_Z: config.recon.del_z,
        constants.IOFields.INSTITUTION: twix_utils.get_institution_name(twix_obj),
        constants.IOFields.SYSTEM_VENDOR: twix_utils.get_system_vendor(twix_obj),
        constants.IOFields.N_FRAMES: data_dict[constants.IOFields.N_FRAMES],
        constants.IOFields.N_SKIP_END: data_dict[constants.IOFields.N_SKIP_END],
        constants.IOFields.N_SKIP_START: data_dict[constants.IOFields.N_SKIP_START],
        constants.IOFields.ORIENTATION: twix_utils.get_orientation(twix_obj),
        constants.IOFields.PROTOCOL_NAME: twix_utils.get_protocol_name(twix_obj),
        constants.IOFields.RAMP_TIME: twix_utils.get_ramp_time(twix_obj),
        constants.IOFields.REMOVEOS: twix_utils.get_flag_removeOS(twix_obj),
        constants.IOFields.SCAN_DATE: twix_utils.get_scan_date(twix_obj),
        constants.IOFields.SOFTWARE_VERSION: twix_utils.get_software_version(twix_obj),
        constants.IOFields.TE90: twix_utils.get_TE90(twix_obj),
        constants.IOFields.TR: twix_utils.get_TR_dissolved(twix_obj),
        constants.IOFields.BANDWIDTH: twix_utils.get_bandwidth(
            twix_obj, data_dict, filename
        ),
    }


def read_ute_twix(path: str) -> Dict[str, Any]:
    """Read proton ute imaging twix file.

    Args:
        path: str file path of twix file
    Returns: dictionary containing data and metadata extracted from the twix file.
    This includes:
        TODO
    """
    try:
        twix_obj = mapvbvd.mapVBVD(path)
    except:
        raise ValueError("Invalid twix file.")
    try:
        twix_obj.image.squeeze = True
    except:
        # this is old data, need to get the 2nd element
        twix_obj = twix_obj[1]
    try:
        twix_obj.image.squeeze = True
        twix_obj.image.flagIgnoreSeg = True
        twix_obj.image.flagRemoveOS = False
    except:
        raise ValueError("Cannot get data from twix object.")
    data_dict = twix_utils.get_ute_data(twix_obj=twix_obj)

    return {
        constants.IOFields.SAMPLE_TIME: twix_utils.get_sample_time(twix_obj),
        constants.IOFields.FIDS: data_dict[constants.IOFields.FIDS],
        constants.IOFields.INSTITUTION: twix_utils.get_institution_name(twix_obj),
        constants.IOFields.SYSTEM_VENDOR: twix_utils.get_system_vendor(twix_obj),
        constants.IOFields.RAMP_TIME: twix_utils.get_ramp_time(twix_obj),
        constants.IOFields.GRAD_DELAY_X: data_dict[constants.IOFields.GRAD_DELAY_X],
        constants.IOFields.GRAD_DELAY_Y: data_dict[constants.IOFields.GRAD_DELAY_Y],
        constants.IOFields.GRAD_DELAY_Z: data_dict[constants.IOFields.GRAD_DELAY_Z],
        constants.IOFields.N_SKIP_END: data_dict[constants.IOFields.N_SKIP_END],
        constants.IOFields.N_SKIP_START: data_dict[constants.IOFields.N_SKIP_START],
        constants.IOFields.N_FRAMES: data_dict[constants.IOFields.N_FRAMES],
        constants.IOFields.ORIENTATION: twix_utils.get_orientation(twix_obj),
    }


def read_dyn_mrd(path: str) -> Dict[str, Any]:
    """Read dynamic spectroscopy MRD file.

    Args:
        path: str file path of MRD file
    Returns: dictionary containing data and metadata extracted from the MRD file.
    This includes:
        1. scan date in MM-DD-YYY format.
        2. dwell time in seconds.
        3. TR in seconds.
        4. Center frequency in MHz.
        5. excitation frequency in ppm.
        6. dissolved phase FIDs in format (n_points, n_projections).
    """
    try:
        dataset = ismrmrd.Dataset(path, "dataset", create_if_needed=False)
        header = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())
    except:
        raise ValueError("Invalid mrd file.")
    # Get scan information
    sample_time = mrd_utils.get_sample_time(dataset=dataset)
    fids_dis = mrd_utils.get_dyn_fids(dataset=dataset)
    xe_center_frequency = mrd_utils.get_center_freq(header=header)
    xe_dissolved_offset_frequency = mrd_utils.get_excitation_freq(header=header)
    scan_date = mrd_utils.get_scan_date(header=header)
    tr = mrd_utils.get_TR(header=header)

    return {
        constants.IOFields.SAMPLE_TIME: sample_time,
        constants.IOFields.FIDS_DIS: fids_dis,
        constants.IOFields.XE_CENTER_FREQUENCY: xe_center_frequency,
        constants.IOFields.XE_DISSOLVED_OFFSET_FREQUENCY: xe_dissolved_offset_frequency,
        constants.IOFields.SCAN_DATE: scan_date,
        constants.IOFields.TR: tr,
    }


def read_dis_mrd(path: str, multi_echo: bool) -> Dict[str, Any]:
    """Read 1-point dixon disssolved phase imaging mrd file.

    Args:
        path: str file path of mrd file
        multi_echo: option to perform multi echo
    Returns: dictionary containing data and metadata extracted from the mrd file.
    This includes:
        - Age: Patient age as a float.
        - Sex: Patient sex as a string ("M" or "F").
        - Height: Patient height in centimeters as a float.
        - dwell time in seconds.
        - flip angle applied to dissolved phase in degrees.
        - flip angle applied to gas phase in degrees.
        - dissolved phase FIDs in format (n_projections, n_points).
        - gas phase FIDs in format (n_projections, n_points).
        - field of view in cm.
        - center frequency in MHz.
        - excitation frequency in ppm.
        - gradient delay (x, y, z) in microseconds.
        - number of frames (projections) used to calculate trajectory.
        - number of projections to skip at the beginning of the scan.
        - number of projections to skip at the end of the scan.
        - orientation of the scan.
        - protocol name
        - ramp time in microseconds.
        - scan date in YYYY-MM-DD format.
        - software version
        - TE90 in seconds.
        - TR in seconds.
    """
    try:
        dataset = ismrmrd.Dataset(path, "dataset", create_if_needed=False)
        header = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())
    except:
        raise ValueError("Invalid mrd file.")

    data_dict = mrd_utils.get_gx_data(dataset, multi_echo)
    return {
        constants.IOFields.AGE: np.nan,
        constants.IOFields.SEX: np.nan,
        constants.IOFields.HEIGHT: np.nan,
        constants.IOFields.BANDWIDTH: np.nan,
        constants.IOFields.SAMPLE_TIME: mrd_utils.get_sample_time_gas_exchange(dataset),
        constants.IOFields.FA_DIS: mrd_utils.get_flipangle_dissolved(header),
        constants.IOFields.FA_GAS: mrd_utils.get_flipangle_gas(header),
        constants.IOFields.FIDS: data_dict[constants.IOFields.FIDS],
        constants.IOFields.FIDS_DIS: data_dict[constants.IOFields.FIDS_DIS],
        constants.IOFields.FIDS_GAS: data_dict[constants.IOFields.FIDS_GAS],
        constants.IOFields.FIELD_STRENGTH: mrd_utils.get_field_strength(header),
        constants.IOFields.FOV: mrd_utils.get_FOV(header),
        constants.IOFields.XE_CENTER_FREQUENCY: mrd_utils.get_center_freq(header),
        constants.IOFields.XE_DISSOLVED_OFFSET_FREQUENCY: mrd_utils.get_excitation_freq(
            header
        ),
        constants.IOFields.GRAD_DELAY_X: np.nan,
        constants.IOFields.GRAD_DELAY_Y: np.nan,
        constants.IOFields.GRAD_DELAY_Z: np.nan,
        constants.IOFields.INSTITUTION: mrd_utils.get_institution_name(header),
        constants.IOFields.SYSTEM_VENDOR: mrd_utils.get_system_vendor(header),
        constants.IOFields.ORIENTATION: mrd_utils.get_orientation(header),
        constants.IOFields.PROTOCOL_NAME: mrd_utils.get_protocol_name(header),
        constants.IOFields.RAMP_TIME: mrd_utils.get_ramp_time(header),
        constants.IOFields.REMOVEOS: False,
        constants.IOFields.SCAN_DATE: mrd_utils.get_scan_date(header),
        constants.IOFields.SYSTEM_VENDOR: mrd_utils.get_system_vendor(header),
        constants.IOFields.SOFTWARE_VERSION: "NA",
        constants.IOFields.TE90: mrd_utils.get_TE90(header),
        constants.IOFields.TR: mrd_utils.get_TR_dissolved(header),
        constants.IOFields.TRAJ: data_dict[constants.IOFields.TRAJ],
    }


def read_ute_mrd(path: str) -> Dict[str, Any]:
    """Read proton MRD file.

    Args:
        path (str): file path of mrd file
    Returns: dictionary containing data and metadata extracted from the mrd file.
    This includes:
        TODO
    """
    try:
        dataset = ismrmrd.Dataset(path, "dataset", create_if_needed=False)
        header = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())
    except:
        raise ValueError("Invalid mrd file.")

    data_dict = mrd_utils.get_ute_data(dataset)
    return {
        constants.IOFields.SAMPLE_TIME: mrd_utils.get_sample_time(dataset),
        constants.IOFields.FIDS: data_dict[constants.IOFields.FIDS],
        constants.IOFields.ORIENTATION: mrd_utils.get_orientation(header),
        constants.IOFields.SYSTEM_VENDOR: mrd_utils.get_system_vendor(header),
        constants.IOFields.RAMP_TIME: mrd_utils.get_ramp_time(header),
        constants.IOFields.GRAD_DELAY_X: np.nan,
        constants.IOFields.GRAD_DELAY_Y: np.nan,
        constants.IOFields.GRAD_DELAY_Z: np.nan,
        constants.IOFields.TRAJ: data_dict[constants.IOFields.TRAJ],
    }


def read_dicom(path: str, shape_gas: Tuple[int, int, int]) -> np.ndarray:
    """Read in DICOM image as numpy array.

    Args:
        path (str): path to DICOM proton image
        shape_gas (tuple[int, int, int]): size of Dixon gas image

    Returns:
        np.ndarray: numpy array from DICOM image.
    """
    files = os.listdir(path)
    file1 = os.path.join(path, files[0])
    RefDs = pydicom.dcmread(file1)
    ConstPixelDims = (int(RefDs.Columns), int(RefDs.Rows), len(files))
    dicom = np.zeros(ConstPixelDims)

    for i in range(0, len(files)):
        if files[i] == ".DS_Store":
            continue
        file_with_path = os.path.join(path, files[i])
        RefDs = pydicom.dcmread(file_with_path)
        slice_num = int(RefDs.InstanceNumber)
        ds = pydicom.read_file(file_with_path)
        dicom[:, :, slice_num - 1] = ds.pixel_array

    image = dicom.astype("float64")

    if image.shape[0] != shape_gas[0]:
        image = img_utils.interp(image, shape_gas[0] / image.shape[0])

    return image


def export_nii(image: np.ndarray, path: str, fov: Optional[float] = None):
    """Export image as nifti file.

    Args:
        image: np.ndarray 3D image to be exported
        path: str file path of nifti file
        fov: float field of view in cm
    """
    nii_imge = nib.Nifti1Image(image, np.eye(4))
    if fov:
        nii_imge.header["pixdim"][1:4] = [
            fov / np.shape(image)[0] / 10,
            fov / np.shape(image)[0] / 10,
            fov / np.shape(image)[0] / 10,
        ]
    nib.save(nii_imge, path)


def export_nii_4d(image, path, fov=None):
    """Export 4d image image as nifti file.

    Args:
        image: np.ndarray 4D image to be exported of shape (x,y,z,3)
        path: str file path of nifti file
        fov: float field of view in cm
    """
    # Scale values and clip to ensure they stay in the [0, 255] range
    color = np.copy(image)*255  
    color = np.clip(color, 0, 255).astype("uint8")

    # some fancy and tricky re-arrange
    color = np.transpose(color, [2, 3, 0, 1])
    cline = np.reshape(color, (1, np.size(color)))
    color = np.reshape(cline, np.shape(image), order="A")
    color = np.transpose(color, [2, 1, 0, 3])
    # stake the RGB channels
    shape_3d = image.shape[0:3]
    rgb_dtype = np.dtype([("R", "u1"), ("G", "u1"), ("B", "u1")])
    nii_data = (
        color.copy().view(dtype=rgb_dtype).reshape(shape_3d)
    )  # copy used to force fresh internal structure

    # write voxel dimensions to nifti header if available
    nii_imge = nib.Nifti1Image(nii_data, np.eye(4))
    if fov:
        nii_imge.header["pixdim"][1:4] = [
            fov / np.shape(image)[0] / 10,
            fov / np.shape(image)[0] / 10,
            fov / np.shape(image)[0] / 10,
        ]
    nib.save(nii_imge, path)


def export_subject_mat(subject: object, path: str):
    """Export select subject instance variables to mat file.

    Args:
        subject: subject instance
        path: str file path of mat file
    """
    sio.savemat(path, vars(subject), long_field_names=True)


def export_np(arr: np.ndarray, path: str):
    """Export numpy array to npy file.

    Args:
        arr: np.ndarray array to be exported
        path: str file path of npy file
    """
    np.save(path, arr)


def export_subject_csv(dict_stats: Dict[str, Any], path: str, overwrite=False):
    """Export statistics to running csv file.

    Uses the csv.DictWriter class to write a csv file. First, checks if the csv
    file exists and the header has been written. If not, writes the header. Then,
    writes to a new file or new row of data in existing file.

    Args:
        dict_stats (dict): dictionary containing statistics to be exported
        path (str): file path of csv file
        overwrite (bool): if True, overwrite existing csv file
    """
    header = dict_stats.keys()
    if overwrite or (not os.path.exists(path)):
        with open(path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            writer.writerow(dict_stats)
    else:
        with open(path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writerow(dict_stats)


def export_config_to_json(config: config_dict, path: str) -> None:
    """
    Save a dictionary to a JSON file.

    Args:
    - data (ml_collections.ConfigDict): The config dictionary to save.
    - path (str): The name of the file to save the dictionary to.

    Returns:
    - None
    """
    with open(path, "w") as f:
        f.write(config.to_json_best_effort(indent=4))


def move_files(source_paths: list, destination_path: str) -> None:
    """Move files to a new directory.

    If target directory does not exist, it is created.
    Args:
        source_paths (list): list of paths of files to move
        destination_path (str): path to move files to
    """
    # if target directory doesn't exist, create it
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    # move files to target directory
    for path in source_paths:
        fname = os.path.basename(path)
        if os.path.isfile(path):
            shutil.move(path, os.path.join(destination_path, fname))
