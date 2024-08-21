"""Twix file util functions."""

import logging
import sys

sys.path.append("..")
import datetime
from typing import Any, Dict

import mapvbvd
import numpy as np

from utils import constants


def get_scan_date(twix_obj: mapvbvd._attrdict.AttrDict) -> str:
    """Get the scan date in MM-DD-YYYY format.

    Args:
        twix_obj: twix object returned from mapVBVD function
    Returns:
        scan date string in MM-DD-YYYY format
    """
    try:
        tReferenceImage0 = str(twix_obj.hdr.MeasYaps[("tReferenceImage0",)]).strip('"')
        scan_date = tReferenceImage0.split(".")[-1][:8]
    except KeyError:
        SeriesLOID = twix_obj.hdr.Config[("SeriesLOID")]
        scan_date = SeriesLOID.split(".")[-4][:8]
    return scan_date[:4] + "-" + scan_date[4:6] + "-" + scan_date[6:]


def get_sample_time(twix_obj: mapvbvd._attrdict.AttrDict) -> float:
    """Get the dwell time in seconds.

    Args:
        twix_obj: twix object returned from mapVBVD function
    Returns:
        dwell time in seconds
    """
    try:
        return float(twix_obj.hdr.Phoenix[("sRXSPEC", "alDwellTime", "0")]) * 1e-9
    except:
        pass
    try:
        return float(twix_obj.hdr.Meas.alDwellTime.split(" ")[0]) * 1e-9
    except:
        pass
    raise ValueError("Could not find dwell time from twix object")


def get_TR(twix_obj: mapvbvd._attrdict.AttrDict) -> float:
    """Get the TR in seconds.

    Args:
        twix_obj: twix object returned from mapVBVD function
    Returns:
        TR in seconds
    """
    try:
        return float(twix_obj.hdr.Config.TR.split(" ")[0]) * 1e-6
    except:
        pass

    try:
        return float(twix_obj.hdr.Phoenix[("alTR", "0")]) * 1e-6
    except:
        pass

    raise ValueError("Could not find TR from twix object")


def get_TR_dissolved(twix_obj: mapvbvd._attrdict.AttrDict) -> float:
    """Get the TR in seconds for dissolved phase.

    The dissolved phase TR is defined to be the time between two consecutive dissolved
    phase-FIDS. This is different from the TR in the twix header as the twix header
    provides the TR for two consecutive FIDS. Here, we assume an interleaved sequence.

    Args:
        twix_obj: twix object returned from mapVBVD function
    Returns:
        TR in seconds
    """
    try:
        return 2 * float(twix_obj.hdr.Config.TR) * 1e-6
    except:
        pass
    try:
        return 2 * int(twix_obj.hdr.Config.TR.split(" ")[0]) * 1e-6
    except:
        pass

    raise ValueError("Could not find TR from twix object")


def get_center_freq(twix_obj: mapvbvd._attrdict.AttrDict) -> float:
    """Get the center frequency in MHz.

    See: https://mriquestions.com/center-frequency.html for definition of center freq.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        center frequency in MHz.
    """
    try:
        return twix_obj.hdr.Meas.lFrequency * 1e-6
    except:
        pass

    try:
        return int(twix_obj.hdr.Dicom["lFrequency"]) * 1e-6
    except:
        pass

    raise ValueError("Could not find center frequency (MHz) from twix object")


def get_excitation_freq(twix_obj: mapvbvd._attrdict.AttrDict) -> float:
    """Get the excitation frequency in MHz.

    See: https://mriquestions.com/center-frequency.html.
    Return 218.0 if not found.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        rf excitation frequency in ppm. 0 if not found.
    """
    excitation = 0
    try:
        excitation = twix_obj.hdr.Phoenix["sWipMemBlock", "alFree", "4"]
        return round(
            excitation
            / (constants.GRYOMAGNETIC_RATIO * get_field_strength(twix_obj=twix_obj))
        )
    except:
        pass
    try:
        excitation = twix_obj.hdr.MeasYaps[("sWiPMemBlock", "adFree", "8")]
        return round(
            excitation
            / (constants.GRYOMAGNETIC_RATIO * get_field_strength(twix_obj=twix_obj))
        )
    except:
        logging.warning("Could not get excitation frequency from twix object.")

    return 218.0


def get_field_strength(twix_obj: mapvbvd._attrdict.AttrDict) -> float:
    """Get the magnetic field strength in Tesla.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        magnetic field strength in Tesla.
    """
    try:
        field_strength = twix_obj.hdr.Dicom.flMagneticFieldStrength
    except:
        logging.warning("Could not find magnetic field strength, using 3T.")
        field_strength = 3.0
    return field_strength


def get_ramp_time(twix_obj: mapvbvd._attrdict.AttrDict) -> float:
    """Get the ramp time in micro-seconds.

    See: https://mriquestions.com/gradient-specifications.html

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        ramp time in us
    """
    ramp_time = 0.0
    scan_date = get_scan_date(twix_obj=twix_obj)
    YYYY, MM, DD = scan_date.split("-")
    scan_datetime = datetime.datetime(int(YYYY), int(MM), int(DD))

    try:
        ramp_time = float(twix_obj.hdr.Meas.RORampTime)
        if scan_datetime > datetime.datetime(2018, 9, 21):
            return ramp_time
    except:
        pass

    try:
        ramp_time = float(twix_obj["hdr"]["Meas"]["alRegridRampupTime"].split()[0])
    except:
        pass

    return max(100, ramp_time) if ramp_time < 100 else ramp_time


def get_flag_removeOS(twix_obj: mapvbvd._attrdict.AttrDict) -> bool:
    """Get the flag to remove oversampling.

    Returns false by default.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        flag to remove oversampling
    """
    try:
        return twix_obj.image.flagRemoveOS
    except:
        return False


def get_software_version(twix_obj: mapvbvd._attrdict.AttrDict) -> str:
    """Get the software version.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        software version
    """
    try:
        return twix_obj.hdr.Dicom.SoftwareVersions
    except:
        pass

    return "unknown"


def get_FOV(twix_obj: mapvbvd._attrdict.AttrDict) -> float:
    """Get the FOV in cm.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        FOV in cm. 40cm if not found.
    """
    try:
        return float(twix_obj.hdr.Config.ReadFoV) / 10.0
    except:
        pass
    logging.warning("Could not find FOV from twix object. Returning 40cm.")
    return 40.0


def get_TE90(twix_obj: mapvbvd._attrdict.AttrDict) -> float:
    """Get the TE90 in seconds.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        TE90 in seconds
    """
    return twix_obj.hdr.Phoenix[("alTE", "0")] * 1e-6


def get_flipangle_dissolved(twix_obj: mapvbvd._attrdict.AttrDict) -> float:
    """Get the dissolved phase flip angle in degrees.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        flip angle in degrees
    """
    scan_date = get_scan_date(twix_obj=twix_obj)
    YYYY, MM, DD = scan_date.split("-")
    if datetime.datetime(int(YYYY), int(MM), int(DD)) < datetime.datetime(2021, 5, 30):
        logging.info("Checking for flip angle in old format.")
        try:
            return float(twix_obj.hdr.MeasYaps[("sWipMemBlock", "adFree", "6")])
        except:
            pass
        try:
            return float(twix_obj.hdr.MeasYaps[("sWiPMemBlock", "adFree", "6")])
        except:
            pass
    try:
        return float(twix_obj.hdr.Meas["adFlipAngleDegree"].split(" ")[1])
    except:
        pass
    try:
        return float(twix_obj.hdr.MeasYaps[("adFlipAngleDegree", "1")])
    except:
        pass
    try:
        return float(twix_obj.hdr.MeasYaps[("adFlipAngleDegree", "0")])
    except:
        pass
    raise ValueError("Unable to find dissolved-phase flip angle in twix object.")


def get_flipangle_gas(twix_obj: mapvbvd._attrdict.AttrDict) -> float:
    """Get the gas phase flip angle in degrees.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        flip angle in degrees. Returns 0.5 degrees if not found.
    """
    try:
        return float(twix_obj.hdr.Meas["adFlipAngleDegree"].split(" ")[0])
    except:
        pass
    try:
        assert float(twix_obj.hdr.MeasYaps[("adFlipAngleDegree", "0")]) < 10.0
        return float(twix_obj.hdr.MeasYaps[("adFlipAngleDegree", "0")])
    except:
        pass
    try:
        return float(twix_obj.hdr.MeasYaps[("sWipMemBlock", "adFree", "5")])
    except:
        pass
    try:
        return float(twix_obj.hdr.MeasYaps[("sWiPMemBlock", "adFree", "5")])
    except:
        pass
    logging.info("Returning default flip angle of 0.5 degrees.")
    return 0.5


def get_orientation(twix_obj: mapvbvd._attrdict.AttrDict) -> str:
    """Get the orientation of the image.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        orientation. Returns coronal if not found.
    """
    orientation = ""
    try:
        orientation = str(twix_obj.hdr.Dicom.tOrientation)
    except:
        logging.info("Unable to find orientation from twix object, returning coronal.")
    return orientation.lower() if orientation else constants.Orientation.CORONAL


def get_protocol_name(twix_obj: mapvbvd._attrdict.AttrDict) -> str:
    """Get the protocol name.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        protocol name. Returns "unknown" if not found.
    """
    try:
        return str(twix_obj.hdr.Config.ProtocolName)
    except:
        return "unknown"


def get_institution_name(twix_obj: mapvbvd._attrdict.AttrDict) -> str:
    """Get institution name.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        institution name. Returns "unknown" if not found.
    """
    try:
        return str(twix_obj.hdr.Dicom.InstitutionName)
    except:
        return "unknown"


def get_system_vendor(
    twix_obj: mapvbvd._attrdict.AttrDict
) -> str:
    """Get system vendor from the Twix header.

    Args
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        system vendor (str)
    """
    try :
        return str(twix_obj.hdr.Dicom.Manufacturer)
    except:
        return "Siemens"


def get_dyn_fids(
    twix_obj: mapvbvd._attrdict.AttrDict, n_skip_end: int = 20
) -> np.ndarray:
    """Get the dissolved phase FIDS used for dyn. spectroscopy from twix object.

    Args:
        twix_obj: twix object returned from mapVBVD function
        n_skip_end: number of fids to skip from the end. Usually they are calibration
            frames.
    Returns:
        dissolved phase FIDs in shape (number of points in ray, number of projections).
    """
    raw_fids = twix_obj.image[""].astype(np.cdouble)
    return raw_fids[:, 0 : -(1 + n_skip_end)]


def get_bandwidth(
    twix_obj: mapvbvd._attrdict.AttrDict, data_dict: Dict[str, Any], filename: str
) -> float:
    """Get the bandwidth in Hz/pixel.

    If the filename contains "BW", then this is a Ziyi-era sequence and the bandwidth
    must be calculated differently.

    Args:
        twix_obj: twix object returned from mapVBVD function.
        data_dict: dictionary containing the output of get_gx_data function.
        filename: filename of the twix file.
    Returns:
        bandwidth in Hz/pixel
    """
    sample_time = get_sample_time(twix_obj=twix_obj)
    npts = data_dict[constants.IOFields.FIDS_DIS].shape[1]
    return (
        1.0 / (2 * sample_time * npts)
        if "BW" not in filename
        else 1.0 / (2 * npts * sample_time / 2)
    )


def get_gx_data(twix_obj: mapvbvd._attrdict.AttrDict) -> Dict[str, Any]:
    """Get the dissolved phase and gas phase FIDs from twix object.

    For reconstruction, we also need important information like the gradient delay,
    number of fids in each phase, etc. Note, this cannot be trivially read from the
    twix object, and need to hard code some values. For example, the gradient delay
    is slightly different depending on the scanner.
    Args:
        twix_obj: twix object returned from mapVBVD function
    Returns:
        a dictionary containing
        1. dissolved phase FIDs in shape (number of projections,
            number of points in ray).
        2. gas phase FIDs in shape (number of projections, number of points in ray).
        3. number of fids in each phase, used for trajectory calculation. Note:
            this may not always be equal to the shape in 1 and 2.
        4. number of FIDs to skip from the beginning. This may be due to a noise frame.
        5. number of FIDs to skip from the end. This may be due to calibration.
        6. gradient delay x in microseconds.
        7. gradient delay y in microseconds.
        8. gradient delay z in microseconds.
        9. raw fids in shape (number of projections, number of points in ray).
    """
    raw_fids = np.transpose(twix_obj.image.unsorted().astype(np.cdouble))
    flip_angle_dissolved = get_flipangle_dissolved(twix_obj)
    # get the scan date
    scan_date = get_scan_date(twix_obj=twix_obj)
    YYYY, MM, DD = scan_date.split("-")
    scan_datetime = datetime.datetime(int(YYYY), int(MM), int(DD))
    # check the flip angle and scan date to get the data
    if flip_angle_dissolved == 12:
        if raw_fids.shape[0] == 4200:
            logging.info("Reading in fast dixon data on Siemens Prisma.")
            data_gas = raw_fids[0::2, :]
            data_dis = raw_fids[1::2, :]
            n_frames = data_dis.shape[0]
            n_skip_start = 0
            n_skip_end = 0
            grad_delay_x, grad_delay_y, grad_delay_z = -5, -5, -5
        elif raw_fids.shape[0] == 4230:
            logging.info("Reading in fast dixon data on Siemens Prisma.")
            data_gas = raw_fids[:-30][0::2, :]
            data_dis = raw_fids[:-30][1::2, :]
            n_frames = data_dis.shape[0]
            n_skip_start = 0
            n_skip_end = 0
            grad_delay_x, grad_delay_y, grad_delay_z = -5, -5, -5
        elif raw_fids.shape[0] // 100 == 42:
            logging.info("Reading in fast dixon data on Siemens Prisma.")
            num_spectra = raw_fids.shape[0] % 100
            data_gas = raw_fids[:-num_spectra][0::2, :]
            data_dis = raw_fids[:-num_spectra][1::2, :]
            n_frames = data_dis.shape[0]
            n_skip_start = 0
            n_skip_end = 0
            grad_delay_x, grad_delay_y, grad_delay_z = -5, -5, -5
        else:
            raise ValueError("Cannot get data from 'fast' dixon twix object.")
    elif flip_angle_dissolved == 15:
        if raw_fids.shape[0] == 2430:
            logging.info("Reading in medium dixon data on Siemens Prisma.")
            data_gas = raw_fids[:-30][0::2, :]
            data_dis = raw_fids[:-30][1::2, :]
            n_frames = data_dis.shape[0]
            n_skip_start = 0
            n_skip_end = 0
            grad_delay_x, grad_delay_y, grad_delay_z = -5, -5, -5
        elif raw_fids.shape[0] // 100 == 24:
            logging.info("Reading in medium dixon data on Siemens Prisma.")
            num_spectra = raw_fids.shape[0] % 100
            data_gas = raw_fids[:-num_spectra][0::2, :]
            data_dis = raw_fids[:-num_spectra][1::2, :]
            n_frames = data_dis.shape[0]
            n_skip_start = 0
            n_skip_end = 0
            grad_delay_x, grad_delay_y, grad_delay_z = -5, -5, -5
        else:
            raise ValueError("Cannot get data from 'medium' dixon twix object.")
    elif flip_angle_dissolved == 20:
        if raw_fids.shape[0] == 2030:
            logging.info("Reading in 'normal' dixon data on Siemens Prisma w/ bonus.")
            data_gas = raw_fids[:-30][0::2, :]
            data_dis = raw_fids[:-30][1::2, :]
            n_frames = data_dis.shape[0]
            n_skip_start = 0
            n_skip_end = 0
            grad_delay_x, grad_delay_y, grad_delay_z = -5, -5, -5
        elif raw_fids.shape[0] == 2002:
            if scan_datetime > datetime.datetime(2017, 12, 31):
                logging.info("Reading in 'normal' dixon data on Siemens Trio.")
                data_gas = raw_fids[:-2][2::2, :]
                data_dis = raw_fids[:-2][3::2, :]
                n_frames = 1001
                n_skip_start = 1
                n_skip_end = 1
                grad_delay_x, grad_delay_y, grad_delay_z = 0, -4, -3
            else:
                logging.info("Reading in 'normal' dixon data on Siemens Trio.")
                data_gas = raw_fids[:-2][2::2, :]
                data_dis = raw_fids[:-2][3::2, :]
                n_frames = 1001
                n_skip_start = 1
                n_skip_end = 1
                grad_delay_x, grad_delay_y, grad_delay_z = 24, 22, 22
        elif raw_fids.shape[0] == 2032:
            logging.info("Reading in normal dixon on Siemens Trio w/ bonus spectra.")
            data_gas = raw_fids[:-32][2::2, :]
            data_dis = raw_fids[:-32][3::2, :]
            n_frames = 1016
            n_skip_start = 1
            n_skip_end = 16
            grad_delay_x, grad_delay_y, grad_delay_z = 0, -4, -3
        elif raw_fids.shape[0] == 2000:
            logging.info("Reading in normal dixon on Siemens Trio 2007 or 2008.")
            data_gas = raw_fids[0::2, :] * np.exp(1j * np.pi / 2)
            data_dis = raw_fids[1::2, :] * np.exp(1j * np.pi / 2)
            n_frames = 1000
            n_skip_start = 0
            n_skip_end = 0
            grad_delay_x, grad_delay_y, grad_delay_z = 0, -4, -3
        elif raw_fids.shape[0] // 100 == 20:
            logging.info("Reading in normal dixon on Siemens Trio w/ bonus spectra.")
            num_spectra = raw_fids.shape[0] % 100
            data_gas = raw_fids[:-num_spectra][0::2, :]
            data_dis = raw_fids[:-num_spectra][1::2, :]
            n_frames = data_dis.shape[0]
            n_skip_start = 0
            n_skip_end = 0
            grad_delay_x, grad_delay_y, grad_delay_z = -5, -5, -5
        else:
            raise ValueError("Cannot get data from normal dixon twix object.")
    else:
        raise ValueError("Cannot get data from twix object.")

    return {
        constants.IOFields.FIDS_GAS: data_gas,
        constants.IOFields.FIDS_DIS: data_dis,
        constants.IOFields.N_FRAMES: n_frames,
        constants.IOFields.N_SKIP_START: n_skip_start,
        constants.IOFields.N_SKIP_END: n_skip_end,
        constants.IOFields.GRAD_DELAY_X: grad_delay_x,
        constants.IOFields.GRAD_DELAY_Y: grad_delay_y,
        constants.IOFields.GRAD_DELAY_Z: grad_delay_z,
        constants.IOFields.FIDS: raw_fids,
    }


def get_ute_data(twix_obj: mapvbvd._attrdict.AttrDict) -> Dict[str, Any]:
    """Get the UTE FIDs from twix object.

    For reconstruction, we also need important information like the gradient delay,
    number of fids in each phase, etc. Note, this cannot be trivially read from the
    twix object, and need to hard code some values. For example, the gradient delay
    is slightly different depending on the scanner.
    Args:
        twix_obj: twix object returned from mapVBVD function
    Returns:
        a dictionary containing
        1. UTE FIDs in shape (number of projections,
            number of points in ray).
        2. number of FIDs to use for generating trajectory.
        3. number of FIDs to skip from the beginning. This may be due to a noise frame.
        4. number of FIDs to skip from the end. This may be due to blank frame.
        5. gradient delay x in microseconds.
        6. gradient delay y in microseconds.
        7. gradient delay z in microseconds.
    """
    raw_fids = np.array(twix_obj.image.unsorted().astype(np.cdouble))

    if raw_fids.ndim == 3:
        raw_fids = np.squeeze(raw_fids[:, 0, :])

    if raw_fids.shape[1] == 4601:
        # For some reason, the raw data is 4601 points long. We need to remove the
        # last projection.
        raw_fids = raw_fids[:, :4600]
        nframes = 4601
        n_skip_start = 0
        n_skip_end = 1
    elif raw_fids.shape[1] == 4630:
        # bonus spectra at the end
        raw_fids = raw_fids[:, :4600]
        nframes = 4600
        n_skip_start = 0
        n_skip_end = 0
    else:
        nframes = raw_fids.shape[1]
        n_skip_start = 0
        n_skip_end = 0
    data = np.transpose(raw_fids)

    return {
        constants.IOFields.FIDS: data,
        constants.IOFields.N_FRAMES: nframes,
        constants.IOFields.N_SKIP_START: n_skip_start,
        constants.IOFields.N_SKIP_END: n_skip_end,
        constants.IOFields.GRAD_DELAY_X: -5,
        constants.IOFields.GRAD_DELAY_Y: -5,
        constants.IOFields.GRAD_DELAY_Z: -5,
    }
