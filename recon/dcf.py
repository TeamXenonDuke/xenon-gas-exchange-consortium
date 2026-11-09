"""Gridding kernels."""

import logging
import sys
import time
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm

sys.path.append("..")
from recon import system_model
from utils import constants


class DCF(ABC):
    """Density compensation filter abstract class.

    Attributes:
        verbosity (bool): Log output messages.
        unique_string (str): Unique string defining object.
        extent (float): kernel extent. The nonzero range of the kernel in units
            of pre-overgridded k-space voxels.
        space (str): a string
    """

    def __init__(self, kernel_extent: float, verbosity: bool = True):
        """Initialize Kernel Superclass.

        Args:
            verbosity (int): either 0 or 1 whether to log output messages
            kernel_extent (float): kernel extent. The nonzero range of the
                kernel in units of pre-overgridded k-space voxels.
            dcf (np.array): density compensation filter
        """
        self.verbosity = verbosity
        self.extent = kernel_extent
        self.unique_string = "Kernel_e" + str(self.extent)
        self.dcf = np.array([])
        self.space = constants.DCFSpace.DATASPACE

    def times(self, b: np.ndarray):
        """Multiple density compensation filter by array."""
        return np.multiply(self.dcf, b)


class IterativeDCF(DCF):
    """Calculate iterative DCF for reconstruction.

    An iterative DCF class based off:
    Pipe, J. G., & Menon, P. (1999). Sampling density compensation in MRI:
    rationale and an iterative numerical solution. Magnetic resonance in
    medicine : official journal of the Society of Magnetic Resonance in
    Medicine / Society of Magnetic Resonance in Medicine, 41(1), 179–86.
    Retrieved from http://www.ncbi.nlm.nih.gov/pubmed/10025627

    Attributes:
        system_obj (MatrixSystemModel): A subclass of the SystemModel
        dcf_iterations (int): number of iterations for density compensation.
        verbosity (bool): Log output messages.
        space (str): a string
        unique_string (str): unique string defining class.
    """

    def __init__(
        self,
        system_obj: system_model.MatrixSystemModel,
        dcf_iterations: int,
        verbosity: bool,
    ):
        """Initialize the iterative density compensation function class.

        Args:
            system_obj (MatrixSystemModel): A subclass of the SystemModel
            dcf_iterations (int): number of iterations for density compensation.
            verbosity (bool): Log output messages.
        """
        self.system_obj = system_obj
        self.dcf_iterations = dcf_iterations
        self.verbosity = verbosity
        self.unique_string = "iter" + str(dcf_iterations)
        self.space = constants.DCFSpace.DATASPACE
        # system_obj is a MatrixSystemModel
        idea_PSFdata = np.ones((system_obj.A._shape[1], 1))
        # reasonable first guess by summing all up
        dcf = np.divide(1, system_obj.A.dot(idea_PSFdata))
        # start timing
        time_start = time.time()
        # iteratively calculating dcf
        for kk in range(0, self.dcf_iterations):
            if self.verbosity:
                logging.info(" DCF iteration " + str(kk + 1))
            dcf = np.divide(dcf, system_obj.A.dot(system_obj.ATrans.dot(dcf)))

        time_end = time.time()
        if self.verbosity:
            logging.info("The runtime for iterative DCF: " + str(time_end - time_start))
        self.dcf = dcf
