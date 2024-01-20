"""Spectroscopy util functions."""
import math
import sys

sys.path.append("..")
from typing import Any, Optional, Tuple

import numpy as np

import matlab_engine
import spect.nmr_timefit as fit
from utils import constants, metrics


def _get_frequency_guess(
    data: Optional[np.ndarray], center_freq: float, rf_excitation: int
) -> np.ndarray:
    """Get the three-peak initial frequency guess.

    This can be modified in the future to include automated peak finding.

    Args:
        data (np.ndarray): FID data of shape (n_points, 1) or (n_points, ).
        center_freq (float): center frequency in MHz.
        rf_excitation (int): excitation frequency in ppm.

    Returns: 3-element array of initial frequency guesses corresponding to the RBC,
        membrane, and gas frequencys in MHz
    """
    rbc_freq = 217.2 - rf_excitation
    membrane_freq = 197.7 - rf_excitation
    gas_freq = 0 - rf_excitation

    return np.array([rbc_freq, membrane_freq, gas_freq]) * center_freq


def _get_area_guess(data: Optional[np.ndarray], center_freq: float, rf_excitation: int):
    """Get the three-peak initial area guess.

    This can be modified in the future to include automated peak finding.

    Args:
        data (np.ndarray): FID data of shape (n_points, 1) or (n_points, ).
        center_freq (float): center frequency in MHz.
        rf_excitation (int): excitation frequency in ppm.

    Returns: 3-element array of initial area guesses corresponding to the RBC,
        membrane, and gas frequencys in MHz
    """
    if rf_excitation == 208:
        return np.array([1, 1, 1])
    elif rf_excitation == 218:
        return np.array([1, 1, 1])
    else:
        raise ValueError("Invalid excitation frequency {}".format(rf_excitation))


def _get_positive_phase(phase: np.ndarray) -> np.ndarray:
    """Get the positive phase, in reference to membrane phase.

    Args:
        phase (np.ndarray): phase array in degrees, in the order of RBC, membrane, gas.

    Returns:
        phase (np.ndarray): phase in degrees between 0 and 360.
    """
    phase = phase - phase[1]
    phase[phase < 0] = phase[phase < 0] + 360
    return phase


def get_breathhold_indices(
    t: np.ndarray, start_time: int, end_time: int
) -> Tuple[int, int]:
    """Get the start and stop index based on the start and stop time.

    Find the index in the time array corresponding to the start time and the end time.
    If the start index is not found, return 0.
    If the stop index is not found return the last index of the array.

    Args:
        t (np.ndarray): array of time points each FID is collected in units of seconds.
        start_time (int): start time (in seconds) of window to analyze t.
        end_time (int): stop time (in seconds) of window to analyze t.

    Returns:
        Tuple of the indices corresponding to the start time and stop time.
    """

    def round_up(x: float, decimals: int = 0) -> float:
        """Round number to the nearest decimal place.

        Args:
            x: floating point number to be rounded up.
            decimals: number of decimal places to round by.

        Returns:
            rounded up value of x.
        """
        return math.ceil(x * 10**decimals) / 10**decimals

    start_ind = np.argwhere(np.array([round_up(x, 2) for x in t]) == start_time)
    end_ind = np.argwhere(np.array([round_up(x, 2) for x in t]) == end_time)

    if np.size(start_ind) == 0:
        start_ind = [0]
    if np.size(end_ind) == 0:
        end_ind = [np.size(t)]
    return (
        int(start_ind[int(np.floor(np.size(start_ind) / 2))]),
        int(end_ind[int(np.floor(np.size(end_ind) / 2))]),
    )


def get_frequency_guess(
    data: Optional[np.ndarray], center_freq: float, rf_excitation: int
):
    """Get the three-peak initial frequency guess.

    This can be modified in the future to include automated peak finding.

    Args:
        data (np.ndarray): FID data of shape (n_points, 1) or (n_points, ).
        center_freq (float): center frequency in MHz.
        rf_excitation (int): excitation frequency in ppm.

    Returns: 3-element array of initial frequency guesses corresponding to the RBC,
        membrane, and gas frequencys in MHz
    """
    if rf_excitation == 208:
        return np.array([10, -21.7, -208.4]) * center_freq
    elif rf_excitation == 218:
        return np.array([0, -21.7, -218.0]) * center_freq
    else:
        raise ValueError("Invalid excitation frequency {}".format(rf_excitation))


def get_area_guess(data: Optional[np.ndarray], center_freq: float, rf_excitation: int):
    """Get the three-peak initial area guess.

    This can be modified in the future to include automated peak finding.

    Args:
        data (np.ndarray): FID data of shape (n_points, 1) or (n_points, ).
        center_freq (float): center frequency in MHz.
        rf_excitation (int): excitation frequency in ppm.

    Returns: 3-element array of initial area guesses corresponding to the RBC,
        membrane, and gas frequencys in MHz
    """
    if rf_excitation == 208:
        return np.array([1, 1, 1])
    elif rf_excitation == 218:
        return np.array([1, 1, 1])
    else:
        raise ValueError("Invalid excitation frequency {}".format(rf_excitation))


def fit_static_spectroscopy(
    fids: np.ndarray,
    sample_time: float = 1.95e-05,
    tr: float = 0.015,
    center_freq: float = 34.09,
    rf_excitation: int = 218,
    n_avg: Optional[int] = None,
    n_avg_seconds: int = 1,
    method: str = "voigt",
    average_all: bool = True,
) -> Tuple[float, Any]:
    """Fit static spectroscopy data to Voigt model and extract RBC:M ratio.

    The RBC:M ratio is defined as the ratio of the fitted RBC peak area to the membrane
    peak area.
    Args:
        fid (np.ndarray): Dissolved phase FIDs in format (n_points, n_frames).
        sample_time (float): Dwell time in seconds.
        tr (float): TR in seconds.
        center_freq (float): Center frequency in MHz.
        rf_excitation (int, optional): _description_. Excitation frequency in ppm.
        n_avg (int, optional): Number of FIDs to average for static spectroscopy.
        n_avg_seconds (int): Number of seconds to average for
            static spectroscopy.

    Returns:
        Tuple of RBC:M ratio and fit object
    """
    t = np.array(range(0, np.shape(fids)[0])) * sample_time
    t_tr = np.array(range(1, np.shape(fids)[1] + 1)) * tr

    start_ind, _ = get_breathhold_indices(t=t_tr, start_time=2, end_time=10)
    # calculate number of FIDs to average
    if n_avg:
        n_avg = n_avg
    else:
        n_avg = int(np.ceil(n_avg_seconds / tr))

    end_ind = np.min([len(fids[0, :]) - 1, start_ind + n_avg + 1])
    if average_all:
        start_ind = 0
        end_ind = len(fids[0, :])
    data_dis_avg = np.mean(fids[:, start_ind:end_ind], axis=1)
    area = (
        _get_area_guess(
            data=None, center_freq=center_freq, rf_excitation=rf_excitation
        ),
    )
    freq = (
        _get_frequency_guess(
            data=None, center_freq=center_freq, rf_excitation=rf_excitation
        ),
    )
    fwhmL = (np.array([8.8, 5.0, 1.2]) * center_freq,)
    fwhmG = (np.array([0, 6.1, 0]) * center_freq,)
    phase = (np.array([0, 0, 0]),)
    fit_params0 = np.array([area, freq, fwhmL, fwhmG, phase]).flatten()
    # fit the data
    fit_params = matlab_engine.timefit_lsqcurvefit(
        fit_params0, t, fids[:, start_ind:end_ind]
    )

    # define the fit object
    fit_obj = fit.NMR_TimeFit(
        ydata=data_dis_avg,
        tdata=t,
        area=fit_params[0:3],
        freq=fit_params[3:6],
        fwhmL=fit_params[6:9],
        fwhmG=fit_params[9:12],
        phase=fit_params[12:15],
        line_broadening=0,
        zeropad_size=np.size(t),
        method=method,
    )
    rbc_m_ratio = fit_obj.area[0] / fit_obj.area[1]

    # prepare the output dictionary
    out_dict = {
        constants.SpectIOFields.DATA_DIS_AVG: data_dis_avg,
        constants.SpectIOFields.T_SPECTRA: t,
        constants.SpectIOFields.FIT_PARAMS: fit_params,
        constants.SpectIOFields.MEMBRANE_AREA: (fit_obj.area[1] / fit_obj.area[1]),
        constants.SpectIOFields.MEMBRANE_FWHM: fit_obj.fwhmL[1] / center_freq,
        constants.SpectIOFields.MEMBRANE_FWHMG: fit_obj.fwhmG[1] / center_freq,
        constants.SpectIOFields.MEMBRANE_PHASE: _get_positive_phase(fit_obj.phase)[1],
        constants.SpectIOFields.MEMBRANE_SHIFT_PPM: (fit_obj.freq[1] - fit_obj.freq[-1])
        / center_freq,
        constants.SpectIOFields.MEMBRANE_SNR_TAIL: metrics.get_snr_tail(
            fit_obj.area, fit_obj.ydata, fit_obj.get_time_function(fit_obj.tdata)
        )[1],
        constants.SpectIOFields.RBC_AREA: fit_obj.area[0] / fit_obj.area[1],
        constants.SpectIOFields.RBC_FWHM: fit_obj.fwhmL[0] / center_freq,
        constants.SpectIOFields.RBC_PHASE: _get_positive_phase(fit_obj.phase)[0],
        constants.SpectIOFields.RBC_SHIFT_PPM: (fit_obj.freq[0] - fit_obj.freq[-1])
        / center_freq,
        constants.SpectIOFields.RBC_SNR_TAIL: metrics.get_snr_tail(
            fit_obj.area, fit_obj.ydata, fit_obj.get_time_function(fit_obj.tdata)
        )[0],
    }

    return rbc_m_ratio, out_dict
