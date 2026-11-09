"""Gridding kernels."""

from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm


class Kernel(ABC):
    """Gridding kernel abstract class.

    Attributes:
        verbosity (bool): Log output messages.
        extent (float): kernel extent. The nonzero range of the kernel in units
            of pre-overgridded k-space voxels.
        unique_string (str): Unique string defining object.
    """

    def __init__(self, kernel_extent: float, verbosity: bool = True):
        """Initialize Kernel Superclass.

        Args:
            verbosity (int): either 0 or 1 whether to log output messages
            kernel_extent (float): kernel extent. The nonzero range of the
                kernel in units of pre-overgridded k-space voxels.
        """
        self.verbosity = verbosity
        self.extent = kernel_extent
        self.unique_string = "Kernel_e" + str(self.extent)

    @abstractmethod
    def evaluate(self, distances: np.ndarray) -> np.ndarray:
        """Evaluate kernel function."""
        pass


class Gaussian(Kernel):
    """Gaussian kernel for gridding.

    Attributes:
        sigma (float): The sharpness of the gaussian function.
        unique_string (str): Unique string defining object.
    """

    def __init__(self, kernel_extent: float, kernel_sigma: float, verbosity: bool):
        """Initialize Gaussian Kernel subclass.

        Args:
            kernel_sigma (float): The sharpness of the gaussian function.
            verbosity (bool): Log output messages
            kernel_extent (float): kernel extent. The nonzero range of the
                kernel in units of pre-overgridded k-space voxels.
        """
        super().__init__(kernel_extent=kernel_extent, verbosity=verbosity)
        self.sigma = kernel_sigma
        self.unique_string = "Gaussian_e" + str(self.extent) + "_s" + str(self.sigma)

    def evaluate(self, distances: np.ndarray) -> np.ndarray:
        """Calculate Normalized Gaussian Function.

        Args:
            distances (np.ndarray): kernel distances before overgridding.

        Returns:
            np.ndarray: normalized gaussian function evaluated at kdistance_preovergrid
        """
        kernel_vals = np.divide(
            norm.pdf(distances, 0, self.sigma), norm.pdf(0, 0, self.sigma)
        )
        return kernel_vals
