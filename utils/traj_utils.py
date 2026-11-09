"""Trajectory calculation util functions."""

import math
import sys
from typing import Callable, Tuple

sys.path.append("..")
import numpy as np
from absl import app, flags

from utils import constants

FLAGS = flags.FLAGS
flags.DEFINE_integer("n_proj", 100, "number of projections per frame.")
flags.DEFINE_string("traj_type", "halton", "trajectory type.")


_GOLDMEAN1 = 0.465571231876768
_GOLDMEAN2 = 0.682327803828019


def _swap(arr: np.ndarray, i: int, j: int):
    """Swap two elements in an array.

    Args:
        arr (np.ndarray): array
        i (int): index of first element
        j (int): index of second element
    """
    arr[i], arr[j] = arr[j], arr[i]


def _partition(
    ht_polar: np.ndarray, sp_polar: np.ndarray, sp_azi: np.ndarray, low: int, high: int
) -> int:
    """Places the pivot element at its correct position in sorted array.

    Also places all smaller (smaller than pivot) to left of pivot and all greater elements
    to right of pivot

    For more details, see: https://www.geeksforgeeks.org/quick-sort/

    Returns:
        int: index of pivot element
    """
    pivot = ht_polar[high]
    i = low - 1
    for j in range(low, high):
        if ht_polar[j] <= pivot:
            i += 1
            _swap(ht_polar, i, j)
            _swap(sp_polar, i, j)
            _swap(sp_azi, i, j)
    _swap(ht_polar, i + 1, high)
    _swap(sp_polar, i + 1, high)
    _swap(sp_azi, i + 1, high)

    return i + 1


def quicksort(
    ht_polar: np.ndarray, sp_polar: np.ndarray, sp_azi: np.ndarray, low: int, high: int
):
    """Sort ht_polar in ascending order. Use sorted indices to sort sp_polar and sp_azi.

    Algorithm explanation: https://www.geeksforgeeks.org/quick-sort/

    """
    if low < high:
        pi = _partition(ht_polar, sp_polar, sp_azi, low, high)
        quicksort(ht_polar, sp_polar, sp_azi, low, pi - 1)
        quicksort(ht_polar, sp_polar, sp_azi, pi + 1, high)


def _halton_number(index: int, base: int) -> float:
    """Calculate halton number.

    Reference: https://en.wikipedia.org/wiki/Halton_sequence

    Args:
        index (int): index of halton sequence
        base (int): base of halton sequence

    Returns:
        float: halton number
    """
    result = 0
    f = 1.0
    i = index
    while i > 0:
        f = f / base
        result += f * math.fmod(i, base)
        i = int(i / base)
    return result


def _halton_seq(
    arr_azimuthal_angle: np.ndarray,
    arr_polar_angle: np.ndarray,
    num_frames: int,
    num_projPerFrame: int,
):
    """Generate halton sequence.

    Updates arrays in place.
    Args:
        arr_azimuthal_angle (np.ndarray): azimuthal angle array.
        arr_polar_angle (np.ndarray): polar angle array.
        num_frames (int): number of frames.
        num_projPerFrame (int): number of projections per frame.
    """
    p1 = 2
    p2 = 3
    for lFrame in range(num_frames):
        for lk in range(num_projPerFrame):
            linter = lk + lFrame * num_projPerFrame
            z = _halton_number(lk + 1, p1) * 2 - 1
            phi = 2 * math.pi * _halton_number(lk + 1, p2)
            arr_polar_angle[linter] = math.acos(z)
            arr_azimuthal_angle[linter] = phi


def _spiral_seq(
    arr_azimuthal_angle: np.ndarray,
    arr_polar_angle: np.ndarray,
    num_frames: int,
    num_projPerFrame: int,
):
    """Generate spiral sequence.

    Updates arrays in place.
    Args:
        arr_azimuthal_angle (np.ndarray): azimuthal angle array.
        arr_polar_angle (np.ndarray): polar angle array.
        num_frames (int): number of frames.
        num_projPerFrame (int): number of projections per frame.
    """
    dPreviousAngle = 0
    num_totalProjections = num_frames * num_projPerFrame
    for lk in range(num_projPerFrame):
        for lFrame in range(num_frames):
            llin = lFrame + lk * num_frames
            linter = lk + lFrame * num_projPerFrame
            dH = -1.0 + 2.0 * llin / float(num_totalProjections)
            arr_polar_angle[linter] = math.acos(dH)
            if llin == 0:
                arr_azimuthal_angle[linter] = 0
            else:
                arr_azimuthal_angle[linter] = math.fmod(
                    dPreviousAngle
                    + 3.6 / (math.sqrt(num_totalProjections * (1.0 - dH * dH))),
                    2.0 * math.pi,
                )
            dPreviousAngle = arr_azimuthal_angle[linter]


def _archimedian_seq(
    arr_azimuthal_angle: np.ndarray,
    arr_polar_angle: np.ndarray,
    num_frames: int,
    num_projPerFrame: int,
):
    """Generate archimedian sequence.

    Updates arrays in place.
    Args:
        arr_azimuthal_angle (np.ndarray): azimuthal angle array.
        arr_polar_angle (np.ndarray): polar angle array.
        num_frames (int): number of frames.
        num_projPerFrame (int): number of projections per frame.
    """
    dAngle = (3.0 - math.sqrt(5.0)) * math.pi
    dZ = 2.0 / (num_projPerFrame - 1.0)

    for lFrame in range(num_frames):
        for lk in range(num_projPerFrame):
            linter = lk + lFrame * num_projPerFrame
            arr_polar_angle[linter] = math.acos(1.0 - dZ * lk)
            arr_azimuthal_angle[linter] = lk * dAngle


def _golden_mean_seq(
    arr_azimuthal_angle: np.ndarray,
    arr_polar_angle: np.ndarray,
    num_frames: int,
    num_projPerFrame: int,
):
    """Generate golden mean sequence.

    Updates arrays in place.
    Args:
        arr_azimuthal_angle (np.ndarray): azimuthal angle array.
        arr_polar_angle (np.ndarray): polar angle array.
        num_frames (int): number of frames.
        num_projPerFrame (int): number of projections per frame.
    """
    for lFrame in range(num_frames):
        for lk in range(num_projPerFrame):
            linter = lk + lFrame * num_projPerFrame
            arr_polar_angle[linter] = math.acos(2.0 * math.fmod(lk * _GOLDMEAN1, 1) - 1)
            arr_azimuthal_angle[linter] = 2 * math.pi * math.fmod(lk * _GOLDMEAN2, 1)


def _random_spiral_seq(
    arr_azimuthal_angle: np.ndarray,
    arr_polar_angle: np.ndarray,
    num_projPerFrame: int,
):
    """Generate random spiral sequence.

    Updates arrays in place.
    Args:
        arr_azimuthal_angle (np.ndarray): azimuthal angle array.
        arr_polar_angle (np.ndarray): polar angle array.
        num_frames (int): number of frames.
        num_projPerFrame (int): number of projections per frame.
    """
    ht_adAzimu = np.zeros(num_projPerFrame)
    ht_adPolar = np.zeros(num_projPerFrame)
    _halton_seq(ht_adAzimu, ht_adPolar, 1, num_projPerFrame)
    quicksort(ht_adPolar, arr_polar_angle, arr_azimuthal_angle, 0, num_projPerFrame - 1)


def _halton_spiral_seq(
    arr_azimuthal_angle: np.ndarray,
    arr_polar_angle: np.ndarray,
    num_frames: int,
    num_projPerFrame: int,
):
    """Generate halton spiral sequence.

    Updates arrays in place.
    Args:
        arr_azimuthal_angle (np.ndarray): azimuthal angle array.
        arr_polar_angle (np.ndarray): polar angle array.
        num_frames (int): number of frames.
        num_projPerFrame (int): number of projections per frame.
    """
    _spiral_seq(arr_azimuthal_angle, arr_polar_angle, num_frames, num_projPerFrame)
    _random_spiral_seq(arr_azimuthal_angle, arr_polar_angle, num_projPerFrame)


def _traj_factory(traj_type: str) -> Callable:
    """Get trajectory generation function.

    Args:
        traj_type: Trajectory type.
    """
    if traj_type == constants.TrajType.SPIRAL:
        return _spiral_seq
    elif traj_type == constants.TrajType.HALTON:
        return _halton_seq
    elif traj_type == constants.TrajType.HALTONSPIRAL:
        return _halton_spiral_seq
    elif traj_type == constants.TrajType.SPIRALRANDOM:
        return _random_spiral_seq
    elif traj_type == constants.TrajType.ARCHIMEDIAN:
        return _archimedian_seq
    elif traj_type == constants.TrajType.GOLDENMEAN:
        return _golden_mean_seq
    else:
        raise ValueError("Invalid trajectory type {}.".format(traj_type))


def _gen_traj(num_projPerFrame: int, traj_type: str) -> np.ndarray:
    """Generate trajectory by trajectory type.

    Args:
        n_ProjectionsPerFrame (int): number of projections per frame.
        traj_type (str): trajectory type.

    Returns:
        np.ndarray: trajectory coordinates of shape
        [coordinates_x + coordinates_y + coordinates z, 1]
    """
    m_adAzimuthalAngle = np.zeros(num_projPerFrame)
    m_adPolarAngle = np.zeros(num_projPerFrame)
    coordinates = np.zeros(num_projPerFrame * 3)
    num_frames = 1

    _traj_factory(traj_type)(
        m_adAzimuthalAngle, m_adPolarAngle, num_frames, num_projPerFrame
    )

    for k in range(num_projPerFrame):
        coordinates[k] = math.sin(m_adPolarAngle[k]) * math.cos(m_adAzimuthalAngle[k])
        coordinates[k + num_projPerFrame] = math.sin(m_adPolarAngle[k]) * math.sin(
            m_adAzimuthalAngle[k]
        )
        coordinates[k + 2 * num_projPerFrame] = math.cos(m_adPolarAngle[k])
    return coordinates


def _generate_radial_1D_traj(
    decay_time: float = 60,
    sample_time: float = 10,
    grad_delay_time: float = 0,
    n_points: int = 64,
    plat_time: float = 2500,
    ramp_time: float = 100,
) -> np.ndarray:
    """Generate 1D radial distance.

    Generate 1d radial distance array based on the timing and the amplitude
        and the gradient delay.

    Args:
        sample_time (float): dwell time in us
        grad_delay_time (float): gradient delay time in us
        ramp_time (float): gradient ramp time in us
        plat_time (float): plateau time in us
        decay_time (float): decay time in us
        n_points (int): number of points in radial projection

    Returns:
        np.ndarray: 1D radial distances
    """
    grad_delay_npts = grad_delay_time / sample_time
    ramp_npts = ramp_time / sample_time
    plat_npts = plat_time / sample_time
    decay_npts = decay_time / sample_time
    pts_vec = np.array(range(0, n_points))
    # calculate sample number of each region boundary
    ramp_start_pt = grad_delay_npts
    plat_start_pt = ramp_start_pt + ramp_npts
    decay_start_pt = plat_start_pt + plat_npts
    decay_end_pt = decay_start_pt + decay_npts
    # calculate binary mask for each region
    in_ramp = (pts_vec >= ramp_start_pt) & (pts_vec < plat_start_pt)
    in_plat = (pts_vec >= plat_start_pt) & (pts_vec < decay_start_pt)
    in_decay = (pts_vec >= decay_start_pt) & (pts_vec < decay_end_pt)
    # calculate times in each region
    ramp_pts_vec = np.multiply((pts_vec - ramp_start_pt), in_ramp)
    plat_pts_vec = np.multiply((pts_vec - plat_start_pt), in_plat)
    decay_pts_vec = np.multiply((pts_vec - decay_start_pt), in_decay)
    # calculate the gradient amplitude  over time(assume plateau is 1)
    ramp_g = ramp_pts_vec / ramp_npts
    plat_g = in_plat
    decay_g = np.multiply((1.0 - decay_pts_vec / decay_npts), in_decay)
    # calculate radial position (0.5)
    ramp_dist = 0.5 * np.multiply(ramp_pts_vec, ramp_g)
    plat_dist = 0.5 * ramp_npts * in_plat + np.multiply(plat_pts_vec, plat_g)
    decay_dist = (0.5 * ramp_npts + plat_npts) * in_decay + np.multiply(
        in_decay, np.multiply(decay_pts_vec * 0.5, (1.0 + decay_g))
    )
    radial_distance = (ramp_dist + plat_dist + decay_dist) / n_points
    return radial_distance


def generate_trajectory(
    decay_time: float = 60,
    del_x: float = 0,
    del_y: float = 0,
    del_z: float = 0,
    sample_time: float = 10,
    n_frames: int = 1000,
    n_points: int = 64,
    plat_time: float = 2500,
    ramp_time: float = 100,
    traj_type: str = constants.TrajType.HALTONSPIRAL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate and vectorize trajectory and data.

    Combines the 1D radial distance and the 3D trajectory coordinates of the edges to
        generate the full 3D trajectory coordinates.

    Args:
        sample_time (float): dwell time in us
        grad_delay_time (float): gradient delay time in us
        ramp_time (float): gradient ramp time in us
        plat_time (float): plateau time in us
        decay_time (float): decay time in us
        npts (int): number of points in radial projection
        del_x (float): gradient delay in x-direction in us
        del_y (float): gradient delay in y-direction in us
        del_z (float): gradient delay in z-direction in us
        nFrames (int): number of radial projections
        traj_type (int): trajectory type
            1: Spiral
            2. Halton
            3. Haltonized Spiral
            4. ArchimedianSeq
            5. Double Golden Mean

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: trajectory coodinates in the x, y,
            and z directions.
    """
    radial_distance_x = _generate_radial_1D_traj(
        decay_time=decay_time,
        sample_time=sample_time,
        n_points=n_points,
        grad_delay_time=del_x,
        plat_time=plat_time,
        ramp_time=ramp_time,
    )
    radial_distance_y = _generate_radial_1D_traj(
        decay_time=decay_time,
        sample_time=sample_time,
        n_points=n_points,
        grad_delay_time=del_y,
        plat_time=plat_time,
        ramp_time=ramp_time,
    )
    radial_distance_z = _generate_radial_1D_traj(
        decay_time=decay_time,
        sample_time=sample_time,
        n_points=n_points,
        grad_delay_time=del_z,
        plat_time=plat_time,
        ramp_time=ramp_time,
    )

    traj_angular = _gen_traj(n_frames, traj_type)
    x = traj_angular[:n_frames]
    y = traj_angular[n_frames : 2 * n_frames]
    z = traj_angular[2 * n_frames : 3 * n_frames]

    x = 0.5 * np.array([radial_distance_x]).transpose().dot(np.array([x])).transpose()
    y = 0.5 * np.array([radial_distance_y]).transpose().dot(np.array([y])).transpose()
    z = 0.5 * np.array([radial_distance_z]).transpose().dot(np.array([z])).transpose()

    return x, y, z


def get_scaling_factor(
    recon_size: float = 128,
    n_points: float = 64,
) -> float:
    """Get the scaling factor for the trajectory.

    The scaling factor is used to scale the trajectory to the reconstruction size.
    Otherwise, the image will be too small or too large.

    Args:
        recon_size (int): target reconstructed image size in number of voxels in each
            dimension.
        n_points (int): Number of points on each radial projection.
    Returns:
        (float) The scaling factor.

    NOTE: cincinnati (CCHMC) requires a unique scaling factor
    """
    return n_points / recon_size


def main(argv):
    """Generate trajectories for the given number of projections and trajectory type."""
    _gen_traj(FLAGS.n_proj, FLAGS.traj_type)


if __name__ == "__main__":
    app.run(main)
