"""Preprocessing util functions."""

import sys

sys.path.append("..")
from typing import Any, Dict, Tuple

import numpy as np

from spect import nmr_timefit
from utils import constants, recon_utils, signal_utils, spect_utils, traj_utils


def remove_contamination(
    dict_dis: Dict[str, Any],
    dict_bonus: Dict[str, Any],
) -> Dict:
    """Remove gas contamination from data."""
    _, fit_dict = spect_utils.fit_static_spectroscopy(
        fids=np.transpose(dict_bonus[constants.IOFields.FIDS_BONUS_DIS]),
        sample_time=dict_dis[constants.IOFields.SAMPLE_TIME],
        tr=dict_dis[constants.IOFields.TR] / 2,
        center_freq=dict_dis[constants.IOFields.XE_CENTER_FREQUENCY],
        rf_excitation=dict_dis[constants.IOFields.XE_DISSOLVED_OFFSET_FREQUENCY],
    )
    fit_obj = nmr_timefit.NMR_TimeFit(
        ydata=fit_dict[constants.SpectIOFields.DATA_DIS_AVG],
        tdata=fit_dict[constants.SpectIOFields.T_SPECTRA],
        area=fit_dict[constants.SpectIOFields.FIT_PARAMS][0:3],
        freq=fit_dict[constants.SpectIOFields.FIT_PARAMS][3:6],
        fwhmL=fit_dict[constants.SpectIOFields.FIT_PARAMS][6:9],
        fwhmG=fit_dict[constants.SpectIOFields.FIT_PARAMS][9:12],
        phase=fit_dict[constants.SpectIOFields.FIT_PARAMS][12:15],
        line_broadening=0,
        zeropad_size=np.size(fit_dict[constants.SpectIOFields.T_SPECTRA]),
        method="voigt",
    )
    dict_dis[constants.IOFields.FIDS_DIS] = signal_utils.remove_gasphase_contamination(
        data_dissolved=dict_dis[constants.IOFields.FIDS_DIS],
        data_gas=dict_dis[constants.IOFields.FIDS_DIS],
        sample_time=dict_dis[constants.IOFields.SAMPLE_TIME],
        freq_gas_acq_diss=fit_obj.freq[2],
        phase_gas_acq_diss=fit_obj.phase[2],
        area_gas_acq_diss=fit_obj.area[2],
        fa_gas=0.5,
    )
    return dict_dis


def prepare_traj(data_dict: Dict[str, Any]) -> np.ndarray:
    """Prepare k space trajectory for use in reconstruction.

    Args:
        data_dict: dictionary containing data and metadata extracted from the twix file.

    Returns:
        traj (np.array): trajectory array of shape (n_projections, n_points, 3)
    """
    data = data_dict[constants.IOFields.FIDS]
    traj_x, traj_y, traj_z = traj_utils.generate_trajectory(
        sample_time=1e6 * data_dict[constants.IOFields.SAMPLE_TIME],
        ramp_time=data_dict[constants.IOFields.RAMP_TIME],
        n_frames=data_dict[constants.IOFields.N_FRAMES],
        n_points=data.shape[1],
        del_x=data_dict[constants.IOFields.GRAD_DELAY_X],
        del_y=data_dict[constants.IOFields.GRAD_DELAY_Y],
        del_z=data_dict[constants.IOFields.GRAD_DELAY_Z],
    )
    # remove projections at the beginning and end of the trajectory
    shape_traj = traj_x.shape
    if constants.IOFields.N_SKIP_START in data_dict:
        nskip_start = int(data_dict[constants.IOFields.N_SKIP_START])
        nskip_end = int(data_dict[constants.IOFields.N_SKIP_END])
        traj_x = traj_x[nskip_start : shape_traj[0] - (nskip_end)]
        traj_y = traj_y[nskip_start : shape_traj[0] - (nskip_end)]
        traj_z = traj_z[nskip_start : shape_traj[0] - (nskip_end)]
    # stack trajectory
    traj = np.stack([traj_x, traj_y, traj_z], axis=-1)

    return traj


def normalize_data(data: np.ndarray, normalization: np.ndarray) -> np.ndarray:
    """Normalize data by a given normalization array.

    Args:
        data: data FIDs of shape (n_projections, n_points)
        normalization: normalization array of shape (n_projections,)
    """
    return np.divide(data, np.expand_dims(normalization, -1))


def truncate_data_and_traj(
    data: np.ndarray,
    traj: np.ndarray,
    n_skip_start: int = 200,
    n_skip_end: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Truncate data and trajectory to a specified number of points.

    Args:
        data_dis: data FIDs of shape (n_projections, n_points)
        traj_dis: trajectory of shape (n_projections, n_points, 3)
        n_skip_start: number of projections to skip at the start.
        n_skip_end: number of projections to skip at the end of the trajectory.

    Returns:
        A tuple of data and trajectory arrays with beginning and end projections
        removed.
    """
    shape_data = data.shape
    shape_traj = traj.shape
    return (
        data[n_skip_start : shape_data[0] - (n_skip_end)],
        traj[n_skip_start : shape_traj[0] - (n_skip_end)],
    )


def remove_noisy_projections(
    data: np.ndarray, traj: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove noisy projections from data and trajectory.

    Returns:
        Tuple of data, trajectory with noisy projections removed
    """
    # remove noisy radial projections
    indices = recon_utils.get_noisy_projections(
        data=data,
    )
    data, traj = recon_utils.apply_indices_mask(
        data=data,
        traj=traj,
        indices=np.array(indices),
    )
    return data, traj
