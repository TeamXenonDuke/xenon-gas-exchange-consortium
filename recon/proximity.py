"""Gridding kernels."""

import logging
import sys
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

sys.path.append("..")
from recon import kernel, sparse_gridding_distance


def _get_n_nonsparse_entries(n_points: int, kernel_width: float, n_dims: int) -> int:
    """Calculate maximum size of output indices.

    Args:
        n_points (int): number of points
        kernel_width (float): kernel width
        n_dims (int): number of dimensions
    Returns:
        int: number of non-sparse entries
    """
    max_nNeighbors = 1
    for _ in range(0, n_dims):
        max_nNeighbors = int(max_nNeighbors * (kernel_width + 1))
    return int(n_points * max_nNeighbors)


class Proximity(ABC):
    """Proximity super class.

    An abstract class defining a metric of proximity. There are different
        ways to measure "nearness" in k-space, so this serves as a generic
        object that can have a specific implementation that the user doesn't
        need to know about
    Attributes:
        verbosity (bool): Log output messages.
        kernel_obj (float): Kernel object.
        unique_string (str): Unique string describing class.
    """

    def __init__(self, kernel_obj: kernel.Kernel, verbosity: bool):
        """Initialize Proximity Superclass.

        Args:
            verbosity (bool): log output messages
            kernel_extent (float): kernel extent. The nonzero range of the
                kernel in units of pre-overgridded k-space voxels.
        """
        self.verbosity = verbosity
        self.kernel_obj = kernel_obj
        self.unique_string = "Proximity_" + self.kernel_obj.unique_string

    @abstractmethod
    def evaluate(
        self, traj: np.ndarray, overgrid_factor: int, matrix_size: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """Evaluate kernel function.

        Args:
            traj (np.ndarray): trajectory of shape (K, 3).
            overgrid_factor (int): overgridding factor. typically 3.
            matrix_size (np.ndarray): the gridding matrix size. This will be the
                reconstruction matrix size times the overgrid factor.
        """
        pass


class L2Proximity(Proximity):
    """An L2 proximity class defining distance in an L2 sense.

    Also known as  the Euclidean/pythagorean distance.
    Attributes:
        unique_string (str): unique string describing class
    """

    def __init__(self, kernel_obj: kernel.Kernel, verbosity: bool):
        """Initialize the L2 proximity class.

        Args:
            kernel_obj (kernel.Kernel): A kernel object for evaluating the kernel
            verbosity (bool): Log output messages.
        """
        super().__init__(kernel_obj=kernel_obj, verbosity=verbosity)
        self.unique_string = "L2_" + self.kernel_obj.unique_string

    def evaluate(
        self, traj: np.ndarray, overgrid_factor: int, matrix_size: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform sparse gridding.

        Args:
            traj (np.ndarray): trajectory of shape (K, n_dims)
            overgrid_factor (int): overgridding factor. typically 3
            matrix_size (np.ndarray): the gridding matrix size. This will be the
                reconstruction matrix size times the overgrid factor. Of shape (N,N,N)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: TODO
        """
        if self.verbosity:
            logging.info("Calculating L2 distances ...")

        assert traj.ndim == 2, "Trajectory must be of shape (K, n_dims)"

        n_points, n_dims = traj.shape[0], traj.shape[1]
        kernel_width = overgrid_factor * self.kernel_obj.extent
        (
            sample_idx,
            voxel_idx,
            pre_overgrid_distances,
        ) = sparse_gridding_distance.sparse_gridding_distance(
            coords=traj.flatten(),
            kernel_width=kernel_width,
            n_points=n_points,
            n_dims=n_dims,
            output_dims=matrix_size,
            n_nonsparse_entries=np.array([0]).astype(int),
            max_size=_get_n_nonsparse_entries(
                n_points=n_points, kernel_width=kernel_width, n_dims=n_dims
            ),
            force_dim=-1,
        )
        pre_overgrid_distances = pre_overgrid_distances / overgrid_factor
        if self.verbosity:
            logging.info("Finished Calculating L2 distances.")
            logging.info("Applying L2 bound ...")
        # Remove out of bound values
        keep_values = (sample_idx > 0) & (voxel_idx > 0)
        sample_idx = sample_idx[keep_values]
        voxel_idx = voxel_idx[keep_values]
        pre_overgrid_distances = pre_overgrid_distances[keep_values]

        if self.verbosity:
            logging.info("Applying kernel ...")
        kernel_vals = self.kernel_obj.evaluate(pre_overgrid_distances)

        return sample_idx, voxel_idx, kernel_vals
