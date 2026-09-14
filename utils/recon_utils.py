"""Reconstruction util functions."""

import sys

sys.path.append("..")
from typing import Tuple

import numpy as np


def get_noisy_projections(
    data: np.ndarray,
    snr_threshold: float = 0.7,
    tail: float = 10,
) -> np.ndarray:
    """Remove noisy FID rays in the k space data by finding indices mask.

    Remove noisy FIDs in the kspace data and their corresponding trajectories.

    Args:
        data (np.ndarray): k space datadata of shape (n_projections, n_points)
        thre_snr (float): threshold SNR value
        tail (float, optional): Index to define the tail of FID. Defaults to 10.

    Returns:
        Returns a boolean mask of the indices of the good FIDs.
    """
    thre_dis = snr_threshold * np.average(abs(data[:, :5]))
    max_tail = np.amax(abs(data[:, tail:]), axis=1)
    return max_tail < thre_dis


def apply_indices_mask(
    data: np.ndarray,
    traj: np.ndarray,
    indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply indices mask to data and trajectory.

    Args:
        data (np.ndarray): k space datadata of shape (n_projections, n_points)
        traj (np.ndarray): trajectory of shape (n_projections, n_points, 3)
        indices (np.ndarray): boolean mask of indices to keep.

    Returns:
        Tuple of the data, and traj coordinates with the noisy FIDs removed
        given by the indices mask.
    """
    return (data[indices], traj[indices])


def flatten_data(data: np.ndarray) -> np.ndarray:
    """Flatten data for reconstruction.

    Args:
        data (np.ndarray): data of shape (n_projections, n_points)

    Returns:
        np.ndarray: flattened data of shape (n_projections * n_points, 1)
    """
    return data.reshape((data.shape[0] * data.shape[1], 1))


def flatten_traj(traj: np.ndarray) -> np.ndarray:
    """Flatten trajectory for reconstruction.

    Args:
        traj (np.ndarray): trajectory of shape (n_projections, n_points, 3)
    Returns:
        np.ndarray: flattened trajectory of shape (n_projections * n_points, 3)
    """
    return traj.reshape((traj.shape[0] * traj.shape[1], 3))


def skip_from_flipangle(fa_dis: float) -> int:
    """Calculate the number of frames to skip at the beginning based on dissolved flip angle.

    Uses the steady-state formula:
        N_skip ≈ ln(0.1) / ln(cos(fa))

    Args:
        fa_dis (float): Dissolved flip angle in degrees.

    Returns:
        int: Number of views to skip, rounded up to the nearest integer.
    """
    cos_fa = np.cos(np.radians(fa_dis))

    # Round *up* to ensure sufficient skip for stabilization
    n_skip = np.log(0.1) / np.log(cos_fa)
    return int(np.ceil(n_skip))


def calculate_key_radius(
    dwell_time: float,
    points_per_view: float,
    ramp_time: float,
) -> int:
    """Calculate the optimal key_radius for keyhole reconstruction, assuming trapezoidal
        gradients. Key radius will be expressed as a number of points per view and
        will reach 9.8% of k_max.

    Args:
        dwell_time (float): time between samples in a readout
        points_per_view (float): number of points in a readout
        ramp_time (float): time it takes for gradients to reach their max value

    Returns:
        int: key radius as the number of points per view that reaches 9.8% of k_max
    """

    total_time = dwell_time * points_per_view
    key_radius_ref = 7.43 / 100

    # area of each region divided by the total trapezoidal area (0.5h(a+b))
    frac_k_ramp_up = ramp_time / (2 * (total_time - ramp_time))
    frac_k_flat = (total_time - (3 * ramp_time / 2)) / (total_time - ramp_time)

    if frac_k_ramp_up >= key_radius_ref:
        # optimal time lies on the ramp up
        t_opt = np.sqrt(2 * key_radius_ref * ramp_time * (total_time - ramp_time))

    elif frac_k_flat >= key_radius_ref:
        # optimal time lies on the flat top of the trapezoid
        t_opt = key_radius_ref * (total_time - ramp_time) + (0.5 * ramp_time)

    else:
        # optimal time lies on the ramp down
        t_opt = np.sqrt(
            ramp_time
            * (
                2 * key_radius_ref * (total_time - ramp_time)
                + 3 * ramp_time
                - 2 * total_time
            )
        )

    n_points_opt = np.round(t_opt / dwell_time)
    return int(n_points_opt)
