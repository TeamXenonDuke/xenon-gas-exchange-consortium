"""Reconstruction models."""

import logging
import sys
import time
from abc import ABC, abstractmethod

import numpy as np

sys.path.append("..")

from recon import dcf, system_model
from utils import constants


class GriddedReconModel(ABC):
    """Reconstruction model after gridding.

    Attributes:
        system_obj (MatrixSystemModel): A subclass of the SystemModel
        verbosity (int): either 0 or 1 whether to log output messages
        crop (bool): crop image if used overgridding
        deapodize (bool): use deapodization
    """

    def __init__(self, system_obj: system_model.MatrixSystemModel, verbosity: int):
        """Initialize Gridded Reconstruction model.

        Args:
            system_obj (MatrixSystemModel): A subclass of the SystemModel
            verbosity (int): either 0 or 1 whether to log output messages
        """
        self.deapodize = False
        self.crop = True
        self.verbosity = verbosity
        self.system_obj = system_obj
        self.unique_string = "grid_" + system_obj.unique_string


class LSQgridded(GriddedReconModel):
    """LSQ gridding model.

    Attributes:
        dcf_obj (IterativeDCF): A density compensation function object.
        unique_string (str): A unique string defining this class
    """

    def __init__(
        self,
        system_obj: system_model.MatrixSystemModel,
        dcf_obj: dcf.DCF,
        verbosity: int,
    ):
        """Initialize the LSQ gridding model.

        Args:
            system_obj (MatrixSystemModel): A subclass of the System Object
            dcf_obj (IterativeDCF): A density compensation function object
            verbosity (int): either 0 or 1 whether to log output messages
        """
        super().__init__(system_obj=system_obj, verbosity=verbosity)
        self.dcf_obj = dcf_obj
        self.unique_string = (
            "grid_" + system_obj.unique_string + "_" + dcf_obj.unique_string
        )

    def grid(self, data: np.ndarray) -> np.ndarray:
        """Grid data.

        Currently supports only MatrixSystemModel
        Args:
            data (np.ndarray): complex kspace data of shape (K, 1)

        Raises:
            Exception: DCF string not recognized

        Returns:
            np.ndarray: gridded data.
        """
        if self.dcf_obj.space == constants.DCFSpace.GRIDSPACE:
            gridVol = np.multiply(self.system_obj.ATrans.dot(data), self.dcf_obj.dcf)
        elif self.dcf_obj.space == constants.DCFSpace.DATASPACE:
            gridVol = self.system_obj.ATrans.dot(np.multiply(self.dcf_obj.dcf, data))
        else:
            raise Exception("DCF space type not recognized")
        return gridVol

    def reconstruct(self, data: np.ndarray, traj: np.ndarray) -> np.ndarray:
        """Reconstruct the image given the kspace data and trajectory.

        Args:
            data (np.ndarray): kspace data of shape (K, 1)
            traj (np.ndarray): trajectories of shape (K, 3)

        Returns:
            np.ndarray: reconstructed image volume (complex datatype)
        """
        if self.verbosity:
            logging.info("Reconstructing ...")
            logging.info("-- Gridding Data ...")

        reconVol = self.grid(data)
        if self.verbosity:
            logging.info("-- Finished Gridding.")
        reconVol = np.reshape(reconVol, np.ceil(self.system_obj.full_size).astype(int))
        if self.verbosity:
            logging.info("-- Calculating IFFT ...")
        time_start = time.time()
        # reconVol = np.fft.fftshift(np.fft.ifftn(reconVol))
        reconVol = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(reconVol)))
        time_end = time.time()
        logging.info("The runtime for iFFT: " + str(time_end - time_start))
        if self.verbosity:
            logging.info("-- Finished IFFT.")
        if self.crop:
            reconVol = self.system_obj.crop(reconVol)
        if self.deapodize:
            if self.verbosity:
                logging.info("-- Calculating k-space deapodization function")
            deapVol = self.grid(
                np.expand_dims((~np.any(traj > 0, axis=1)).astype(np.float32), -1)
            )
            deapVol = np.reshape(
                deapVol, np.ceil(self.system_obj.full_size).astype(int)
            )
            if self.verbosity:
                logging.info("-- Calculating image-space deapodization function")
            deapVol = np.fft.ifftn(deapVol)
            deapVol = np.fft.ifftshift(deapVol)
            if self.crop:
                deapVol = self.system_obj.crop(deapVol)
            reconVol = np.divide(reconVol, deapVol)
            if self.verbosity:
                logging.info("-- Finished deapodization.")
        if self.verbosity:
            logging.info("-- Finished Reconstruction.")
        return reconVol
