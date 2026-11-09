"""Gridding kernels."""

import logging
import sys
from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sps

sys.path.append("..")
from recon import proximity


class SystemModel(ABC):
    """An abstract class defining a particular system model represenation.

    Attributes:
        verbosity (int): either 0 or 1 whether to log output messages
        proximity_obj (L2Proximity): a subclass that inherits from Proximity class.
        overgrid_factor (int): overgridding factor
        image_size (tuple): reconstructed image size.
    """

    def __init__(
        self,
        proximity_obj: proximity.Proximity,
        overgrid_factor: int,
        image_size: np.ndarray,
        verbosity: int,
    ):
        """Initialize abstract class.

        Args:
            proximity_obj (L2Proximity): a subclass that inherits from Proximity class.
            overgrid_factor (int): overgridding factor
            image_size (tuple): reconstructed image size
            verbosity (int): either 0 or 1 whether to log output messages
        """
        self.verbosity = verbosity
        self.proximity_obj = proximity_obj
        self.overgrid_factor = overgrid_factor
        self.crop_size = image_size
        self.full_size = np.ceil(self.overgrid_factor * self.crop_size).astype(int)
        self.unique_string = "sysmodel_" + proximity_obj.unique_string

    def crop(self, uncrop: np.ndarray) -> np.ndarray:
        """Crop the image if overgridding was used.

        Args:
            uncrop (np.ndarray): Uncropped image volume of shape (N, N, N)
        Returns:
            np.ndarray: Cropped image volume
        """
        s_lim = np.round(0.5 * (np.subtract(self.full_size, self.crop_size))).astype(
            int
        )
        l_lim = np.round(0.5 * np.add(self.full_size, self.crop_size)).astype(int)
        return uncrop[s_lim[0] : l_lim[0], s_lim[1] : l_lim[1], s_lim[2] : l_lim[2]]

    @abstractmethod
    def multiply(self, b) -> np.ndarray:
        """Multiply the system matrix by a vector."""
        pass

    @abstractmethod
    def transpose(self):
        """Change the transpose of the system matrix."""
        pass


class MatrixSystemModel(SystemModel):
    """A matrix system model class.

    A matrix system model class that stores all interpolation coefficients into a
    sparse matrix. Note that storage of the interpolation coefficients can take
    significant memory. If you are memory limited, consider creating a class which
    calculates interpolation coefficients on the fly, and does not require the memory
    overhead for the system matrix. The downside to on the fly calculations
    is that they compute slower in itterative applications, where interpolation
    coefficients are calculated twice each iteration (once togrid, and once to ungrid)

    Attributes:
        unique_string (str): a unique string describing the matrix system model.
        is_supersparse (bool): if A is a super sparse matrix.
        is_transpose (bool): if transpose of A is used.
        A: The sparse matrix storing interpolation coefficients.
        ATrans: The transpose of the sparse matrix storing interpolation coefficients.
    """

    def __init__(
        self,
        proximity_obj: proximity.Proximity,
        overgrid_factor: int,
        image_size: np.ndarray,
        traj: np.ndarray,
        verbosity: int,
    ):
        """Initialize the matrix system model class.

        Args:
            A: Sparse matrix.
            ATrans: Transpose of the sparse matrix.
            proximity_obj (L2Proximity): A subclass of the proximity class
            overgrid_factor (int): overgridding factor
            image_size (tuple): reconstructed image size
            traj (np.ndarray): trajectories of shape (K, 3)
            verbosity (int): either 0 or 1 whether to log output messages
        """
        super().__init__(
            proximity_obj=proximity_obj,
            overgrid_factor=overgrid_factor,
            image_size=image_size,
            verbosity=verbosity,
        )
        self.unique_string = "MatMod_" + proximity_obj.unique_string
        self.is_supersparse = False
        self.is_transpose = False

        if verbosity:
            logging.info("Calculating Matrix interpolation coefficients...")

        sample_idx, voxel_idx, kernel_vals = self.proximity_obj.evaluate(
            traj=traj, overgrid_factor=self.overgrid_factor, matrix_size=self.full_size
        )
        if verbosity:
            logging.info("Finished calculating Matrix interpolation coefficients)")

        self.A = sps.csr_matrix(
            (kernel_vals, (sample_idx - 1, voxel_idx - 1)),
            shape=(np.shape(traj)[0], np.prod(self.full_size)),
            dtype=np.float64,
        )
        self.A.eliminate_zeros()
        self.ATrans = self.A.transpose()

    def makeSuperSparse(self):
        """Return 1."""
        # achieved by eliminate zeros
        return None

    def revertSparseness(self, argv):
        """Return 1."""
        # function was used on the old code, not anymore
        return None

    def multiply(self, b) -> np.ndarray:
        """Multiply the system matrix by a vector."""
        return self.A.multiply(b) if not self.is_transpose else self.ATrans.multiply(b)

    def transpose(self):
        """Change the transpose of the system matrix."""
        self.is_transpose = not self.is_transpose
