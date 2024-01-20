"""Matlab engine wrapper for Python."""
from typing import Optional

import matlab.engine
import numpy as np
import scipy.io

eng = matlab.engine.start_matlab()
s = eng.genpath("matlab_engine")
eng.addpath(s, nargout=0)


def lsqcurvefit(
    fitparams0: np.ndarray, tdata: np.ndarray, ydata: np.ndarray
) -> np.ndarray:
    """Perform least squares curve fitting of Voigt model using a MATLAB engine.

    Parameters:
    - fitparams0 (np.ndarray): Initial guess for the fit parameters of shape (15,)
        organized as [area1, area2, area3, freq1, freq2, freq3, fwhm1, fwhm2, fwhm3,
        fwhmG1, fwhmG2, fwhmG3, phase1, phase2, phase3]
    - tdata (np.ndarray): 1D Time data points.
    - ydata (np.ndarray): 1D Signal data.

    Returns:
    - np.ndarray, shape (15,): Optimized fit parameters  of shape (15,)
        organized as [area1, area2, area3, freq1, freq2, freq3, fwhm1, fwhm2, fwhm3,
        fwhmG1, fwhmG2, fwhmG3, phase1, phase2, phase3]
    """
    tdata = matlab.double(tdata.tolist())
    ydata = matlab.double(ydata.tolist(), is_complex=True)
    fitparams0 = matlab.double(fitparams0.tolist())
    fit_params = eng.nls_curvefit(fitparams0, tdata, ydata)
    return np.array(fit_params)


def multi_lsqcurvefit(
    fitparams0: np.ndarray, tdata: np.ndarray, ydata: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Perform least squares curve fitting of Voigt model on multiple FIDs.

    Parameters:
    - fitparams0 (np.ndarray): Initial guess for the fit parameters of shape (15,)
        organized as [area1, area2, area3, freq1, freq2, freq3, fwhm1, fwhm2, fwhm3,
        fwhmG1, fwhmG2, fwhmG3, phase1, phase2, phase3]
    - tdata (np.ndarray): 1D Time data points.
    - ydata (np.ndarray): Time-domain signal data of shape (n_points, n_frames)

    Returns:
    - Tuple of Optimized fit parameters for the time-domain signal of shape
        (n_frames, 15) and estimated SNR of each frame of shape (n_frames,).
    """
    tdata = matlab.double(tdata.tolist())
    # If ydata is complex, split it into real and imaginary parts
    if np.iscomplexobj(ydata):
        ydata_real_mat = matlab.double(ydata.real.tolist())
        ydata_imag_mat = matlab.double(ydata.imag.tolist())
        ydata_mat = eng.complex(ydata_real_mat, ydata_imag_mat)
    else:
        ydata_mat = matlab.double(ydata.tolist())
    fitparams0 = matlab.double(fitparams0.tolist())
    [fit_params, snr_dyn] = eng.nls_curvefit_multi(
        fitparams0, tdata, ydata_mat, nargout=2
    )
    return np.array(fit_params), np.array(snr_dyn)


def timefit_lsqcurvefit(
    fitparams0: np.ndarray,
    tdata: np.ndarray,
    ydata: np.ndarray,
) -> np.ndarray:
    """Perform time-domain least squares curve fitting using NMR_Fit_v class.

    Parameters:
    - fitparams0 (np.ndarray): Initial guess for the fit parameters of shape (15,)
        organized as [area1, area2, area3, freq1, freq2, freq3, fwhm1, fwhm2, fwhm3,
        fwhmG1, fwhmG2, fwhmG3, phase1, phase2, phase3]
    - tdata (np.ndarray): 1D Time data points.
    - ydata (np.ndarray): 1D Time-domain signal data.

    Returns:
    - np.ndarray, shape (15,): Optimized fit parameters of shape (15,).
    """
    tdata = matlab.double(tdata.tolist())
    if np.iscomplexobj(ydata):
        ydata_real_mat = matlab.double(ydata.real.tolist())
        ydata_imag_mat = matlab.double(ydata.imag.tolist())
        ydata_mat = eng.complex(ydata_real_mat, ydata_imag_mat)
    else:
        ydata_mat = matlab.double(ydata.tolist())
    fitparams0 = matlab.double(fitparams0.tolist())

    fitparams = eng.nls_curvefit_timefit(fitparams0, tdata, ydata_mat)
    return np.array(fitparams)


def sift(fids: np.ndarray) -> np.ndarray:
    """Apply the SIFT algorithm to the input data using a MATLAB engine.

    Parameters:
    - fids (np.ndarray): Input data for the SIFT algorithm of shape (n_points, n_frames)

    Returns:
    - np.ndarray: Processed data of shape (n_points, n_frames)
    """
    if np.iscomplexobj(fids):
        fids_real_mat = matlab.double(fids.real.tolist())
        fids_imag_mat = matlab.double(fids.imag.tolist())
        fids_mat = eng.complex(fids_real_mat, fids_imag_mat)
    else:
        fids_mat = matlab.double(fids.tolist())
    sifted_fids = eng.sift(fids_mat)
    return np.array(sifted_fids)


def fit_sine(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Fit a sine wave to the input data using a MATLAB engine.

    Parameters:
    - y (np.ndarray): Input data of shape (n_points,)
    - x (np.ndarray): Time data of shape (n_points,)

    Returns:
    - np.ndarray: fit parameters of shape (3,) organized as [amplitude, freq., phase]
    """
    x = matlab.double(x.tolist())
    y = matlab.double(y.tolist())
    fit_params = eng.nls_curvefit_sine(y, x)
    return np.array(fit_params).flatten()


def fit_exp1(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Fit a exponetial function to the input data using a MATLAB engine.

    Parameters:
    - y (np.ndarray): Input data of shape (n_points,)
    - x (np.ndarray): Time data of shape (n_points,)

    Returns:
    - np.ndarray: fit parameters of shape (2,) as a and b in y = a * exp(b * x)
    """
    x = matlab.double(x.tolist())
    y = matlab.double(y.tolist())
    fit_params = eng.nls_curvefit_exp1(y, x)
    return np.array(fit_params).flatten()


def filter_highpass(x: np.ndarray) -> np.ndarray:
    """Filter the input data using a high-pass filter.

    Args:
        x (np.ndarray): Input data of shape (n_points,)
    Return:
        np.ndarray: Filtered data of shape (n_points,)
    """
    x = matlab.double(x.tolist())
    return np.array(eng.filter_highpass(x)).flatten()
