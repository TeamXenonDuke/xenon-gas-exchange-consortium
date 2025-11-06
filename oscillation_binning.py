"""Bin dissolved phase data into high and low signal bins."""

from typing import Literal, Tuple

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import numpy as np

from utils import constants, signal_utils


def bin_rbc_oscillations(
    data_gas: np.ndarray,
    data_dissolved: np.ndarray,
    TR: float,
    rbc_m_ratio: float,
    method: str = constants.BinningMethods.BANDPASS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Bin dissolved phase data into high and low signal bins.

    Args:
        data_gas: gas phase data of shape (n_projections, n_points)
        data_dis: dissolved phase data of shape (n_projections, n_points)
        TR: repetition time in seconds
        rbc_m_ratio: RBC:m ratio
        method: method to use for binning
    Returns:
        Tuple of detrendend data, high and low signal indices respectively.
    """
    # get the k0 data for gas, rbc and membrane
    data_rbc, data_membrane = signal_utils.dixon_decomposition(
        data_dissolved, rbc_m_ratio
    )
    data_rbc_k0, data_membrane_k0 = data_rbc[:, 0], data_membrane[:, 0]
    data_gas_k0 = data_gas[:, 0]
    # negate data if mean is negative
    data_rbc_k0_proc = -data_rbc_k0 if np.mean(data_rbc_k0) < 0 else data_rbc_k0

    plt.figure(figsize=(10, 6))
    plt.plot(data_rbc_k0_proc, label="Data")
    plt.title("Raw RBC k0")
    plt.savefig("tmp/Rawk0.png")

    # smooth data
    window_size = int(1 / (5 * TR))
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    data_rbc_k0_proc = signal_utils.smooth(
        data=data_rbc_k0_proc, window_size=window_size
    )

    plt.figure(figsize=(10, 6))
    plt.plot(data_rbc_k0_proc, label="Data")
    plt.title("Smoothed RBC k0")
    plt.savefig("tmp/Smoothedk0.png")

    if method == constants.BinningMethods.BANDPASS:
        # normalize and detrend by gas k0
        # data_rbc_k0_proc = data_rbc_k0_proc / np.abs(data_gas_k0)
        data_rbc_k0_proc = signal_utils.detrend(data_rbc_k0_proc)
        bias = np.mean(data_rbc_k0_proc)

        plt.figure(figsize=(10, 6))
        plt.plot(data_rbc_k0_proc, label="Data")
        plt.title("Normalized and Detrended RBC k0")
        plt.savefig("tmp/normalizedk0.png")

        # apply bandpass filter
        data_rbc_k0_proc = signal_utils.bandpass(
            data=data_rbc_k0_proc, lowcut=0.5, highcut=2.5, fs=1 / TR
        )

        plt.figure(figsize=(10, 6))
        plt.plot(data_rbc_k0_proc, label="Data")
        plt.title("Bandpassed RBC k0")
        plt.savefig("tmp/bandpassedk0.png")

        k0_sine_fit_data, _ = signal_utils.fit_sine(
            data_rbc_k0_proc, np.linspace(0, 15, data_rbc_k0_proc.size)
        )

        plt.figure(figsize=(10, 6))
        plt.plot(
            np.linspace(0, 15, data_rbc_k0_proc.size),
            data_rbc_k0_proc,
            label="Filtered Data",
        )
        plt.plot(
            np.linspace(0, 15, data_rbc_k0_proc.size),
            k0_sine_fit_data,
            label="Sine Fit",
            linestyle="--",
        )
        plt.title("Sine Fit to Bandpass Filtered Data")
        plt.savefig("tmp/sinefitk0.png")

    elif method == constants.BinningMethods.FIT_SINE:
        bias = np.mean(data_rbc_k0_proc)
        # fit data to biexponential decay and remove trend
        data_rbc_k0_proc = signal_utils.detrend(data_rbc_k0_proc)
        # fit sine wave to data
        data_rbc_k0_proc, _ = signal_utils.fit_sine(
            data_rbc_k0_proc, np.arange(data_rbc_k0_proc.shape[0])
        )
    else:
        raise ValueError(f"Invalid binning method: {method}")
    # calculate the heart rate
    heart_rate = signal_utils.get_heartrate(data_rbc_k0_proc, ts=TR)
    # bin data to high and low signal bins
    high_indices, low_indices = signal_utils.find_high_low_indices(
        data=data_rbc_k0_proc, peak_distance=int((60 / heart_rate) / TR)
    )
    # calculate the mean RBC:m ratio for high and low signal bins
    rbc_m_high = np.abs(
        np.mean(data_rbc_k0[high_indices]) / np.mean(data_membrane_k0[high_indices])
    )
    rbc_m_low = np.abs(
        np.mean(data_rbc_k0[low_indices]) / np.mean(data_membrane_k0[low_indices])
    )
    # add the mean to the data
    # data_rbc_k0_proc = data_rbc_k0_proc + bias
    return data_rbc_k0_proc, high_indices, low_indices, rbc_m_high, rbc_m_low


def bin_rbc_oscillations_slidingwindows(
    data_gas: np.ndarray,
    data_dissolved: np.ndarray,
    TR: float,
    rbc_m_ratio: float,
    method: str = constants.BinningMethods.BANDPASS,
) -> Tuple[np.ndarray, list[np.ndarray]]:
    """Bin dissolved phase data into multiple bins in sliding window fashion.

    Args:
        data_gas: gas phase data of shape (n_projections, n_points)
        data_dis: dissolved phase data of shape (n_projections, n_points)
        TR: repetition time in seconds
        rbc_m_ratio: RBC:m ratio
        method: method to use for binning
    Returns:
        TODO
    """
    # get the k0 data for gas, rbc and membrane
    data_rbc, data_membrane = signal_utils.dixon_decomposition(
        data_dissolved, rbc_m_ratio
    )
    data_rbc_k0, _ = data_rbc[:, 0], data_membrane[:, 0]
    data_gas_k0 = data_gas[:, 0]
    # negate data if mean is negative
    data_rbc_k0_proc = -data_rbc_k0 if np.mean(data_rbc_k0) < 0 else data_rbc_k0
    # smooth data
    window_size = int(1 / (5 * TR))
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    data_rbc_k0_proc = signal_utils.smooth(
        data=data_rbc_k0_proc, window_size=window_size
    )
    if method == constants.BinningMethods.BANDPASS:
        # normalize and detrend by gas k0
        data_rbc_k0_proc = data_rbc_k0_proc / np.abs(data_gas_k0)
        # apply bandpass filter
        data_rbc_k0_proc = signal_utils.bandpass(
            data=data_rbc_k0_proc, lowcut=0.5, highcut=2.5, fs=1 / TR
        )
    elif method == constants.BinningMethods.FIT_SINE:
        # fit data to biexponential decay and remove trend
        data_rbc_k0_proc = signal_utils.detrend(data_rbc_k0_proc)
        # fit sine wave to data
        data_rbc_k0_proc, _ = signal_utils.fit_sine(
            y=data_rbc_k0_proc, x=np.arange(data_rbc_k0_proc.shape[0])
        )
    else:
        raise ValueError(f"Invalid binning method: {method}")
    # calculate the heart rate
    heart_rate = signal_utils.get_heartrate(data_rbc_k0_proc, ts=TR)

    indices = signal_utils.find_indices_sliding_window(
        data=data_rbc_k0_proc, peak_distance=int((60 / heart_rate) / TR)
    )
    return data_rbc_k0_proc, indices
