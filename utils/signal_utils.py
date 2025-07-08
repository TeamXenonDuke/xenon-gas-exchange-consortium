"""Signal processing util functions."""
import sys
from typing import Tuple
import math

sys.path.append("..")
import numpy as np
import scipy.optimize as optimize
import scipy.signal as signal
import scipy.stats as stats

from utils import constants


def _movmean(x: np.ndarray, n: int) -> np.ndarray:
    """Compute moving mean of x over n points.

    Args:
        x (np.ndarray): input data of shape (n,)
        n (int): number of points to average over

    Returns:
        np.ndarray: moving mean of shape (n,)
    """
    return np.convolve(x, np.ones((n,)) / n, mode="same")


def _getxygrid(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get x and y grid.

    This is a lazy implementation of the Matlab function getxygrid in sethandles.m
    Args:
        x (np.ndarray): x data of shape (n,) must have at least 2 elements
            x cannot also have repeated x entries.
        y (np.ndarray): y data of shape (n,) must have at least 2 elements
    Returns:
        Tuple of x and y grid respectively
    """
    # check number of data points to be > 2
    assert len(x) > 2, "x must have at least 2 elements"
    # sort data points to be in order of increasing x
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    return (x, y)


def _sinnstart(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Get starting points fit for sum of n sine functions.

    Computes a starting for the parameters of a sum of n sine functions. By
    Running the y data through a fft, and then locating peaks in results, we
    can find the starting value of the frequency of each sine function 'b'
    in the function y = a*sin(b*x+c). Because a phase-shifted sine function
    is separable and can be converted to a sum of sine and cosine functions,
    starting values for amplitude 'a' and phase 'c' can be found.

    Returns:
        Starting values for the parameters of a sum of n sine functions of
            shape (n * 3, )
    """
    lenx = len(x)
    x, y = _getxygrid(x, y)
    # if data size is too small, cannot find starting values
    if len(x) < 2:
        return np.random.rand(n * 3)
    # loop for sum of sines functions
    start = np.zeros(3 * n)
    oldpeaks = np.array([])
    freqs = np.zeros(n)
    res = np.copy(y)
    for i in range(n):
        # apply fft to the current residuals
        fy = np.fft.fft(res)
        # omit frequencies already used
        if len(oldpeaks) > 0:
            fy[oldpeaks] = 0
        # get starting value for frequency using fft peaks
        maxloc = np.argmax(np.abs(fy[np.arange(0, np.floor(lenx / 2)).astype(int)]))
        np.append(oldpeaks, maxloc)
        w = 2 * np.pi * max((0.5, maxloc)) / (x[-1] - x[0])
        freqs[i] = w
        # compute fourier terms using all frequencies we have so far
        X = np.zeros((lenx, 2 * (i + 1)))
        for j in range(i + 1):
            X[:, 2 * j] = np.sin(freqs[j] * x)
            X[:, 2 * j + 1] = np.cos(freqs[j] * x)
        # fit these terms to get the non-frequency starting values
        ab = np.linalg.lstsq(X, y, rcond=None)[0]
        if i < n:
            res = y - np.matmul(X, ab)
    # all frequencies found, now compute starting values from all
    # frequencies and the corresponding coefficients.
    for i in range(n):
        start[3 * i] = np.sqrt(ab[2 * i] ** 2 + ab[2 * i + 1] ** 2)
        start[3 * i + 1] = freqs[i]
        start[3 * i + 2] = np.arctan2(ab[2 * i + 1], ab[2 * i])
    return start


def _sinbounds(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get bounds for sum of n sine functions.

    Upper bounds are inf. Lower bounds are [-inf, 0, -inf] repeated n times.

    Args:
        n (int): number of sine functions
    Returns:
        Tuple of lower and upper bounds respectively of length n * 3
    """
    return (
        np.tile([-np.inf, 0, -np.inf], n),
        np.tile([np.inf, np.inf, np.inf], n),
    )


def boxcox(data: np.ndarray):
    """Apply box cox transformation on data.

    Args:
        data (np.ndarray): data to be transformed of shape (n,)
    Returns:
        Tuple of transformed data and box cox lambda
    """
    return stats.boxcox(data)


def inverse_boxcox(
    boxcox_lambda: float, data: np.ndarray, scale_factor: float
) -> np.ndarray:
    """Apply inverse box cox transformation on data.

    Args:
        boxcox_lambda (float): box cox lambda
        data (np.ndarray): data to be transformed of shape (n,)
        scale_factor (float): scale factor to be applied to the data
    """
    return np.power(boxcox_lambda * data + 1, 1 / boxcox_lambda) - scale_factor


def remove_gasphase_contamination(
    data_dissolved: np.ndarray,
    data_gas: np.ndarray,
    sample_time: float,
    freq_gas_acq_diss: float,
    phase_gas_acq_diss: float,
    area_gas_acq_diss: float,
    fa_gas: float,
) -> np.ndarray:
    """Remove gas phase contamination in dissolved k-space.

    Takes gas phase k-space and modifies it using NMR fits and gas phase k0
    to produce the expected gas phase contamination k-space data which is
    then removed from the initial contaminated dissolved phase k-space.

    Args:
        data_dissolved (np.ndarray): dissolved k-space data of shape
            (n_projections, n_points)
        data_gas (np.ndarray): gas phase k-space data of shape
            (n_projections, n_points)
        sample_time (float): dwell time in seconds.
        freq_gas_acq_diss (float): gas frequency offset in dissolved
            spectra acquisition in Hz.
        phase_gas_acq_diss (float): gas phase in dissolved spectra acquisition.
            in degrees.
        area_gas_acq_diss (float): gas area in dissolved spectra acquisition.
        fa_gas (float): gas flip angle in degrees.
    Returns:
        Gas phase corrected dissolved k-space data of shape (n_projections, n_points)
    Author: Matt Willmering
    Paper: https://pubmed.ncbi.nlm.nih.gov/33665905/
    """
    # step 0: calculate parameters
    arr_t = sample_time * np.arange(data_dissolved.shape[1])
    # step 1: modulate contamination (gas) to dissolved frequency - first order
    # phase approximation
    phase_shift1 = 2 * np.pi * freq_gas_acq_diss * arr_t  # calculate phase accumulation
    contamination_kspace1 = data_gas * np.exp(1j * phase_shift1)
    # step 2: zero order phase shift of contamination estimation
    phase_shift2 = phase_gas_acq_diss - 180 / np.pi * np.mean(np.angle(data_gas[:, 0]))
    contamination_kspace2 = contamination_kspace1 * np.exp(
        1j * np.pi / 180 * phase_shift2
    )
    # step 3: scale contamination estimation
    scale_factor = area_gas_acq_diss / _movmean(np.abs(data_gas[:, 0]), 100)[-1]
    contamination_kspace3 = (
        contamination_kspace2 * scale_factor / np.cos(np.pi / 180 * fa_gas)
    )
    # step 4: return subtracted contamination
    return data_dissolved - contamination_kspace3


def calculate_t2star_correction(
    te90: float, t2star_dis: float, field_strength: float
) -> float:
    """Calculate T2* correction factor.

    Args:
        te90 (float): echo time in seconds
        t2star_dis (float): T2* of the dissolved phase in seconds at 3T.
        field_strength (float): B0 field strength in Tesla.
    """
    return np.exp(te90 / (t2star_dis * 3.0 / field_strength)) / np.exp(
        te90 / constants.T2STAR_GAS
    )


def calculate_flipangle_correction(fa_gas: float, fa_dis: float) -> float:
    """Calculate flip angle correction factor.

    Rounds to 3 decimal places.
    Args:
        fa_gas (float): gas flip angle in degrees
        fa_dis (float): dissolved flip angle in degrees
    """
    return np.round((np.sin(fa_gas * np.pi / 180) / np.sin(fa_dis * np.pi / 180)), 3)


def calculate_flipangle_factor(fa_gas: float, fa_dis: float) -> float:
    """Calculate ratio between dissolved and gas flip angles.

    Rounds to 2 decimal places.
    Args:
        fa_gas (float): gas flip angle in degrees
        fa_dis (float): dissolved flip angle in degrees
    """
    return np.round((np.sin(fa_dis * np.pi / 180) / np.sin(fa_gas * np.pi / 180)), 2)


def dixon_decomposition(
    data_dissolved: np.ndarray,
    rbc_m_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply 1-point dixon decomposition on FID data.

    Applies phase shift to the dissolved data such that the RBC and membrane are
    separated into the imaginary and real channel respectively.
    Does NOT also apply B0 inhomogeneity correction.

    Args:
        data_dissolved (np.ndarray): dissolved FID data of shape
            (n_projections, n_points)
        rbc_m_ratio (float): RBC:m ratio
    Returns:
        Tuple of decomposed RBC and membrane data respectively
    """
    desired_angle = np.arctan2(rbc_m_ratio, 1.0)
    # use k0 to determine the phase shift
    total_dissolved = np.sum(data_dissolved[:, 0])
    current_angle = np.arctan2(np.imag(total_dissolved), np.real(total_dissolved))
    delta_angle = desired_angle - current_angle

    rotated_data = np.multiply(data_dissolved, np.exp(1j * delta_angle))
    return np.imag(rotated_data), np.real(rotated_data)


def smooth(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Smooth response data.

    Implements a smoothing function that is equivalent to the MATLAB smooth function.
    Source: https://www.mathworks.com/help/curvefit/smooth.html

    Args:
        data (np.ndarray): 1-D array data to be smoothed.
        window_size (int): size of the smoothing window. Defaults to 5.
    Returns:
        Smoothed data.
    """
    out0 = np.convolve(data, np.ones(window_size, dtype=int), "valid") / window_size
    r = np.arange(1, window_size - 1, 2)
    start = np.cumsum(data[: window_size - 1])[::2] / r
    stop = (np.cumsum(data[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def bandpass(data: np.ndarray, lowcut: float, highcut: float, fs: float) -> np.ndarray:
    """Bandpass filter.

    Implements a bandpass filter using a butterworth filter.
    Equivalent to MATLAB bandpass filter.

    Args:
        data (np.ndarray): 1-D array data to be filtered.
        lowcut (float): lowcut frequency in Hz.
        highcut (float): highcut frequency in Hz.
        fs (float): sampling frequency.
    Returns:
        Filtered data.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(6, [low, high], analog=False, btype="bandpass", output="sos")
    return np.array(signal.sosfiltfilt(sos, data))


def lowpass(data: np.ndarray, highcut: float, fs: float) -> np.ndarray:
    """Bandpass filter.

    Implements a bandpass filter using a butterworth filter.
    Equivalent to MATLAB bandpass filter.

    Args:
        data (np.ndarray): 1-D array data to be filtered.
        highcut (float): highcut frequency in Hz.
        fs (float): sampling frequency.
    Returns:
        Filtered data.
    """
    nyq = 0.5 * fs
    high = highcut / nyq
    sos = signal.butter(6, high, btype="lowpass", output="sos")
    return np.array(signal.sosfiltfilt(sos, data))


def fit_sine(data: np.ndarray) -> np.ndarray:
    """Fit the data to a sum of 8 sine waves.

    Args:
        data (np.ndarray): 1-D array data to be fitted.
    Returns:
        Fitted data. Same shape as input data.
    """
    x = np.arange(data.shape[0])
    y = data

    def func(x, *args):
        return (
            args[0] * np.sin(args[1] * x + args[2])
            + args[3] * np.sin(args[4] * x + args[5])
            + args[6] * np.sin(args[7] * x + args[8])
            + args[9] * np.sin(args[10] * x + args[11])
            + args[12] * np.sin(args[13] * x + args[14])
            + args[15] * np.sin(args[16] * x + args[17])
            + args[18] * np.sin(args[19] * x + args[20])
            + args[21] * np.sin(args[22] * x + args[23])
        )

    p0 = _sinnstart(x, y, 8)
    bounds = _sinbounds(8)
    popt, _ = optimize.curve_fit(
        func,
        x,
        y,
        p0=p0,
        bounds=bounds,
    )
    return func(x, *popt)


def detrend(data: np.ndarray) -> np.ndarray:
    """Remove bi-exponential trend along axis from data.

    Fits the data to a bi-exponential decay function and removes the trend.

    Args:
        data (np.ndarray): 1-D array data to be detrended.
    Returns:
        Detrended data. Same shape as input data.
    """
    x = np.arange(data.shape[0])
    y = data

    def func(x, a, b, c, d):
        return a * np.exp(-b * x) + c * np.exp(-d * x)

    popt, _ = optimize.curve_fit(
        func,
        x,
        y,
        p0=[1, 0.1, 1, 0.1],
        method="trf",
        ftol=1e-6,
        xtol=1e-6,
        max_nfev=600,
    )
    return data - func(x, *popt)


def find_peaks(data: np.ndarray, distance: int = 5) -> np.ndarray:
    """Find peaks in data.

    Implements a peak finding function using scipy.signal.find_peaks.

    Args:
        data (np.ndarray): 1-D array data to be filtered.
        distance (int): minimum distance between peaks. Defaults to 5. Units are
        number of points.

    Returns:
        Array of indices of peaks.
    """
    peaks, _ = signal.find_peaks(data, distance=distance)
    return peaks[np.argwhere(data[peaks] > 0).flatten()]


def get_heartrate(data: np.ndarray, ts: float) -> float:
    """Calculate heart rate from data.

    Implements a heart rate calculation function by finding the strongest peak
    in the fourier domain of the data.

    Args:
        data (np.ndarray): 1-D array data to be filtered.
        ts (float): sampling period in seconds.

    Returns:
        Heart rate in beats per minute.
    """
    fft_data = np.abs(np.fft.fftshift(np.fft.fft(data)))
    freq = np.fft.fftshift(np.fft.fftfreq(len(data), ts))
    return np.abs(freq[np.argmax(fft_data)] * 60)


def awgn(sig: np.ndarray, SNR: float) -> np.ndarray:
    """Add white gaussian noise.

    Args:
        sig (np.ndarray): signal to be added with noise.
        SNR (float): signal to noise ratio in dB.
    """
    sig_power = np.sum(np.abs(sig) ** 2) / len(sig)
    noise_power = sig_power / (10 ** (SNR / 10))

    if np.isreal(sig):
        noise = np.sqrt(noise_power) * np.random.randn(len(sig))
    else:
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(sig)) + 1j * np.random.randn(len(sig))
        )
    return sig + noise


def get_hb_correction(hb: float) -> Tuple[float, float]:
    """Get scaling factors for hb correction.

    Args:
        hb (float): subject hb in g/dL

    Returns:
        rbc_hb_correction_factor (float): rbc hb correction factor
        membrane_hb_correction_factor (float): membrane hb correction factor

    Reference: https://onlinelibrary.wiley.com/doi/10.1002/mrm.29712
    """

    rbc_hb_correction_factor = constants.HbCorrection.R1 + (
        constants.HbCorrection.HB_REF * (1 - constants.HbCorrection.R1) / hb
    )
    membrane_hb_correction_factor = (1 + constants.HbCorrection.M1 * hb) / (
        1
        + constants.HbCorrection.M1 * constants.HbCorrection.HB_REF
        - constants.HbCorrection.M2 * (constants.HbCorrection.HB_REF - hb)
    )

    return rbc_hb_correction_factor, membrane_hb_correction_factor


def get_vol_correction(vol: float, frc: float, baggie_vol: float) -> Tuple[float, float, float, float, float]:
    """Get scaling factors for volume correction.

    Args:
        vol (float): volume of lung mask in L
        frc (float): FRC as defined by GLI calculator in L
        baggie_vol (float): baggie vol in L

    Returns:
        vol_correction_factor_rbc (float): rbc volume correction factor
        vol_correction_factor_membrane (float): membrane volume correction factor

    """

    V2 = frc + baggie_vol

    vol_correction_factor_rbc = (vol *( 1 + constants.VolCorrection.ALPHA_RBC) + V2 * (1 - constants.VolCorrection.ALPHA_RBC))/(vol * (1 - constants.VolCorrection.ALPHA_RBC) + V2 * (1 + constants.VolCorrection.ALPHA_RBC))
    print("VCF_RBC = " + str(vol_correction_factor_rbc))
    vol_correction_factor_membrane = (vol *( 1 + constants.VolCorrection.ALPHA_MEM) + V2 * (1 - constants.VolCorrection.ALPHA_MEM))/(vol * (1 - constants.VolCorrection.ALPHA_MEM) + V2 * (1 + constants.VolCorrection.ALPHA_MEM))
    print("VCF_mem = " + str(vol_correction_factor_membrane))

    return vol_correction_factor_rbc, vol_correction_factor_membrane, V2