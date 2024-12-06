"""Module for gas exchange imaging subject."""

import glob
import logging
import os
from typing import Any, Dict

import nibabel as nib
import numpy as np

import biasfield
import preprocessing as pp
import reconstruction
import registration
import segmentation
from config import base_config
from utils import (
    binning,
    constants,
    img_utils,
    io_utils,
    metrics,
    plot,
    recon_utils,
    report,
    signal_utils,
    spect_utils,
    traj_utils,
)


class Subject(object):
    """Module to for processing gas exchange imaging.

    Attributes:
        config (config_dict.ConfigDict): config dict
        data_dissolved (np.array): dissolved-phase data of shape (n_projections, n_points)
        data_gas (np.array): gas-phase data of shape (n_projections, n_points)
        data_ute (np.array): UTE proton data of shape (n_projections, n_points)
        dict_dis (dict): dictionary of dissolved-phase data and metadata
        dict_dyn (dict): dictionary of dynamic spectroscopy data and metadata
        dict_ute (dict): dictionary of UTE proton data and metadata
        dict_stats (dict): dictionary of statistics for reporting
        dict_info (dict): dictionary of information for reporting
        image_biasfield (np.array): bias field
        image_dissolved (np.array): dissolved-phase image
        image_gas_binned (np.array): binned gas-phase image
        image_gas_cor (np.array): gas-phase image after bias field correction
        image_gas_highreso (np.array): high-resolution gas-phase image
        image_gas_highsnr (np.array): high-SNR gas-phase image
        image_membrane (np.array): membrane image
        image_membrane2gas (np.array): membrane image normalized by gas-phase image
        image_membrane2gas_binned (np.array): binned image_membrane2gas
        image_proton (np.array): UTE proton image
        image_proton_reg (np.array): registered UTE proton image
        image_rbc (np.array): RBC image
        image_rbc2gas (np.array): RBC image normalized by gas-phase image
        image_rbc2gas_binned (np.array): binned image_rbc2gas
        mask (np.array): thoracic cavity mask
        mask_vent (np.ndarray): thoracic cavity mask without ventilation defects
        membrane_hb_correction_factor (float): membrane hb correction scaling factor
        rbc_hb_correction_factor (float): rbc hb correction scaling factor
        rbc_m_ratio (float): RBC to M ratio
        traj_dissolved (np.array): dissolved-phase trajectory of shape
            (n_projections, n_points, 3)
        traj_gas (np.array): gas-phase trajectory of shape (n_projections, n_points, 3)
        traj_scaling_factor (float): scaling factor for trajectory
        traj_ute (np.array): UTE proton trajectory of shape
    """

    def __init__(self, config: base_config.Config):
        """Init object."""
        logging.info("Initializing gas exchange imaging subject.")
        self.config = config
        self.data_dissolved = np.array([])
        self.data_gas = np.array([])
        self.dict_dis = {}
        self.dict_dyn = {}
        self.image_biasfield = np.array([0.0])
        self.image_dissolved = np.array([0.0])
        self.image_gas_binned = np.array([0.0])
        self.image_gas_cor = np.array([0.0])
        self.image_gas_highreso = np.array([0.0])
        self.image_gas_highsnr = np.array([0.0])
        self.image_membrane = np.array([0.0])
        self.image_membrane2gas = np.array([0.0])
        self.image_membrane2gas_binned = np.array([0.0])
        self.membrane_hb_correction_factor = 1.0
        self.image_proton = np.array([0.0])
        self.image_proton_reg = np.array([0.0])
        self.image_rbc = np.array([0.0])
        self.image_rbc2gas = np.array([0.0])
        self.image_rbc2gas_binned = np.array([0.0])
        self.rbc_hb_correction_factor = 1.0
        self.mask = np.array([0.0])
        self.mask_vent = np.array([0.0])
        self.rbc_m_ratio = 0.0
        self.dict_stats = {}
        self.dict_info = {}
        self.traj_scaling_factor = 1.0
        self.traj_dissolved = np.array([])
        self.traj_gas = np.array([])
        self.traj_ute = np.array([])
        self.reference_data_key = str()
        self.reference_data = {}

    def read_twix_files(self):
        """Read in twix files to dictionary.

        Read in the dynamic spectroscopy (if it exists) and the dissolved-phase image
        data.
        """

        self.dict_dis = io_utils.read_dis_twix(
            io_utils.get_dis_twix_files(str(self.config.data_dir)),
            config = self.config
        )
        try:
            self.dict_dyn = io_utils.read_dyn_twix(
                io_utils.get_dyn_twix_files(str(self.config.data_dir))
            )
        except ValueError:
            logging.info("No dynamic spectroscopy twix file found")
        if self.config.recon.recon_proton:
            self.dict_ute = io_utils.read_ute_twix(
                io_utils.get_ute_twix_files(str(self.config.data_dir))
            )

    def read_mrd_files(self):
        """Read in mrd files to dictionary.

        Read in the dynamic spectroscopy (if it exists) and the dissolved-phase image
        data.
        """
        self.dict_dis = io_utils.read_dis_mrd(
            io_utils.get_dis_mrd_files(str(self.config.data_dir)), 
            self.config.multi_echo
        )
        try:
            self.dict_dyn = io_utils.read_dyn_mrd(
                io_utils.get_dyn_mrd_files(str(self.config.data_dir))
            )
        except ValueError:
            logging.info("No dynamic spectroscopy MRD file found")
        if self.config.recon.recon_proton:
            self.dict_ute = io_utils.read_ute_mrd(
                io_utils.get_ute_mrd_files(str(self.config.data_dir))
            )

    def read_dicom_files(self):
        """Read in DICOM files for proton image."""
        self.image_proton = io_utils.read_dicom(
            self.config.dicom_proton_dir, self.image_gas_highreso.shape
        )

    def read_mat_file(self):
        """Read in mat file of reconstructed images.

        Note: The mat file variable names are matched to the instance variable names.
        Thus, if the variable names are changed in the mat file, they must be changed.
        """
        mdict = io_utils.import_mat(io_utils.get_mat_file(str(self.config.data_dir)))
        self.dict_dis = io_utils.import_matstruct_to_dict(mdict["dict_dis"])
        if "dict_dyn" in mdict.keys() and mdict["dict_dyn"].flatten()[0] is not None:
            self.dict_dyn = io_utils.import_matstruct_to_dict(mdict["dict_dyn"])
        if "dict_ute" in mdict.keys():
            logging.info("UTE proton data found.")
            self.dict_ute = io_utils.import_matstruct_to_dict(mdict["dict_ute"])
        self.data_dissolved = mdict["data_dissolved"]
        self.data_gas = mdict["data_gas"]
        self.image_dissolved = mdict["image_dissolved"]
        self.image_gas_highreso = mdict["image_gas_highreso"]
        self.image_gas_highsnr = mdict["image_gas_highsnr"]
        self.image_gas_cor = mdict["image_gas_cor"]
        self.image_proton = mdict["image_proton"]
        self.image_proton_reg = mdict["image_proton_reg"]
        self.image_biasfield = mdict["image_biasfield"]
        self.mask = mdict["mask"].astype(bool)
        self.mask_vent = mdict["mask_vent"].astype(bool)
        self.traj_dissolved = mdict["traj_dissolved"]
        self.traj_gas = mdict["traj_gas"]
        if self.config.rbc_m_ratio > 0:
            self.rbc_m_ratio = float(self.config.rbc_m_ratio)
        else:
            self.rbc_m_ratio = float(mdict["rbc_m_ratio"])

    def calculate_rbc_m_ratio(self):
        """Calculate RBC:M ratio using static spectroscopy.

        If a manual RBC:M ratio is specified, use that instead.
        """
        if self.config.rbc_m_ratio > 0:  # type: ignore
            self.rbc_m_ratio = float(self.config.rbc_m_ratio)  # type: ignore
            logging.info("Using manual RBC:M ratio of {}".format(self.rbc_m_ratio))
        else:
            logging.info("Calculating RBC:M ratio from static spectroscopy.")
            assert self.dict_dyn[constants.IOFields.FIDS_DIS] is not None
            self.rbc_m_ratio, _ = spect_utils.calculate_static_spectroscopy(
                fid=self.dict_dyn[constants.IOFields.FIDS_DIS],
                sample_time=self.dict_dyn[constants.IOFields.SAMPLE_TIME],
                tr=self.dict_dyn[constants.IOFields.TR],
                center_freq=self.dict_dyn[constants.IOFields.XE_CENTER_FREQUENCY],
                rf_excitation=self.dict_dyn[
                    constants.IOFields.XE_DISSOLVED_OFFSET_FREQUENCY
                ],
                plot=False,
            )

    def preprocess(self):
        """Prepare data and trajectory for reconstruction.

        NOTE: for standard 1pt Dixon sequence, gas and dissolved trajectories are the same.

        Also, calculates the scaling factor for the trajectory.
        """
        # remove contamination
        if self.config.recon.remove_contamination:
            self.dict_dis = pp.remove_contamination(self.dict_dyn, self.dict_dis)

        self.data_dissolved = self.dict_dis[constants.IOFields.FIDS_DIS]
        self.data_gas = self.dict_dis[constants.IOFields.FIDS_GAS]

        if (
            self.dict_dis[constants.IOFields.INSTITUTION]
            == constants.Institution.IOWA.value
            ):
                self.data_dissolved =  np.conjugate(self.data_dissolved);
                self.data_gas = np.conjugate(self.data_gas);

        # get or generate trajectories and trajectory scaling factors
        if constants.IOFields.TRAJ not in self.dict_dis.keys():
            self.traj_dissolved = pp.prepare_traj(self.dict_dis, config=self.config)
            self.traj_scaling_factor = traj_utils.get_scaling_factor(
                recon_size=int(self.config.recon.recon_size),
                n_points=self.data_gas.shape[1],
            )
            self.traj_gas = self.traj_dissolved
        else:
            self.traj_gas = self.dict_dis[constants.IOFields.TRAJ][0]
            self.traj_dissolved = self.dict_dis[constants.IOFields.TRAJ][1]

            if (
                self.dict_dis[constants.IOFields.INSTITUTION]
                == constants.Institution.CCHMC.value
            ):
                self.traj_scaling_factor = (
                    0.903  # cincinnati requires a unique scaling factor
                )
        

        # truncate gas and dissolved data and trajectories
        self.data_dissolved, self.traj_dissolved = pp.truncate_data_and_traj(
            self.data_dissolved,
            self.traj_dissolved,
            n_skip_start=int(self.config.recon.n_skip_start),
            n_skip_end=int(self.config.recon.n_skip_end),
        )
        self.data_gas, self.traj_gas = pp.truncate_data_and_traj(
            self.data_gas,
            self.traj_gas,
            n_skip_start=0,
            n_skip_end=int(self.config.recon.n_skip_end),
        )

        # remove noisy FIDs
        if self.config.recon.remove_noisy_projections:
            self.data_gas, self.traj_gas = pp.remove_noisy_projections(
                self.data_gas, self.traj_gas
            )
            self.data_dissolved, self.traj_dissolved = pp.remove_noisy_projections(
                self.data_dissolved, self.traj_dissolved
            )

        # rescale trajectories
        self.traj_dissolved *= self.traj_scaling_factor
        self.traj_gas *= self.traj_scaling_factor

        # prepare proton data and trajectories
        if self.config.recon.recon_proton:
            # get or generate trajectories
            if constants.IOFields.TRAJ not in self.dict_ute.keys():
                self.traj_ute = pp.prepare_traj(self.dict_ute)
            else:
                self.traj_ute = self.dict_ute[constants.IOFields.TRAJ]

            # get proton data
            self.data_ute = self.dict_ute[constants.IOFields.FIDS]

            # remove noisy FIDs
            if self.config.recon.remove_noisy_projections:
                self.data_ute, self.traj_ute = pp.remove_noisy_projections(
                    self.data_ute, self.traj_ute
                )

            # rescale trajectories
            self.traj_ute *= self.traj_scaling_factor

        
        # Choose appropriate reference distribution
        self.reference_data_key = self.config.reference_data_key

        if self.reference_data_key == constants.ReferenceDataKey.DUKE_REFERENCE.value:
            # Choose between 208 ppmm and 218 ppm. 
            # Default to 218 if other or no value for excitation found. 
            if 216 <= self.dict_dis[
                    constants.IOFields.XE_DISSOLVED_OFFSET_FREQUENCY
                ] <= 220:
                self.reference_data = constants.ReferenceDistribution.REFERENCE_218_PPM

            elif 206 <= self.dict_dis[
                    constants.IOFields.XE_DISSOLVED_OFFSET_FREQUENCY
                ] <= 210:    
                self.reference_data = constants.ReferenceDistribution.REFERENCE_208_PPM

            else: 
                self.reference_data = constants.ReferenceDistribution.REFERENCE_218_PPM
                logging.info("Warning: Unrecognized excitation frequency")

        elif self.reference_data_key == constants.ReferenceDataKey.MANUAL_REFERENCE.value:
            self.reference_data = constants.ReferenceDistribution.REFERENCE_MANUAL
            

    def reconstruction_ute(self):
        """Reconstruct the UTE image."""
        if self.config.recon.recon_key == constants.ReconKey.ROBERTSON.value:
            self.image_proton = reconstruction.reconstruct(
                data=(recon_utils.flatten_data(self.data_ute)),
                traj=recon_utils.flatten_traj(self.traj_ute),
                kernel_sharpness=float(self.config.recon.kernel_sharpness_hr),
                kernel_extent=9 * float(self.config.recon.kernel_sharpness_hr),
                image_size=int(self.config.recon.recon_size),
            )
            orientation = self.dict_ute[constants.IOFields.ORIENTATION]
            system_vendor = self.dict_ute[constants.IOFields.SYSTEM_VENDOR]
        elif self.config.recon.recon_key == constants.ReconKey.PLUMMER.value:
            raise NotImplementedError("Plummer CS reconstruction not implemented.")
        else:
            raise ValueError(f"Unknown reconstruction key")
        self.image_proton = img_utils.interp(
            self.image_proton,
            self.config.recon.matrix_size // self.config.recon.recon_size,
        )
        self.image_proton = img_utils.flip_and_rotate_image(
            self.image_proton,
            orientation=orientation,
            system_vendor=system_vendor,
        )
        io_utils.export_nii(np.abs(self.image_proton), "tmp/image_proton.nii")

    def reconstruction_gas(self):
        """Reconstruct the gas phase image."""
        if self.config.recon.recon_key == constants.ReconKey.ROBERTSON.value:
            self.image_gas_highsnr = reconstruction.reconstruct(
                data=(recon_utils.flatten_data(self.data_gas)),
                traj=recon_utils.flatten_traj(self.traj_gas),
                kernel_sharpness=float(self.config.recon.kernel_sharpness_lr),
                kernel_extent=9 * float(self.config.recon.kernel_sharpness_lr),
                image_size=int(self.config.recon.recon_size),
            )
            self.image_gas_highreso = reconstruction.reconstruct(
                data=(recon_utils.flatten_data(self.data_gas)),
                traj=recon_utils.flatten_traj(self.traj_gas),
                kernel_sharpness=float(self.config.recon.kernel_sharpness_hr),
                kernel_extent=9 * float(self.config.recon.kernel_sharpness_hr),
                image_size=int(self.config.recon.recon_size),
            )
            orientation = self.dict_dis[constants.IOFields.ORIENTATION]
            system_vendor = self.dict_dis[constants.IOFields.SYSTEM_VENDOR]
        elif self.config.recon.recon_key == constants.ReconKey.PLUMMER.value:
            raise NotImplementedError("Plummer CS reconstruction not implemented.")
        else:
            raise ValueError(
                f"Unknown reconstruction key: {self.config.recon.recon_key}"
            )

        self.image_gas_highreso = img_utils.interp(
            self.image_gas_highreso,
            self.config.recon.matrix_size // self.config.recon.recon_size,
        )
        self.image_gas_highsnr = img_utils.interp(
            self.image_gas_highsnr,
            self.config.recon.matrix_size // self.config.recon.recon_size,
        )
        self.image_gas_highsnr = img_utils.flip_and_rotate_image(
            self.image_gas_highsnr,
            orientation=orientation,
            system_vendor=system_vendor,
        )
        self.image_gas_highreso = img_utils.flip_and_rotate_image(
            self.image_gas_highreso,
            orientation=orientation,
            system_vendor=system_vendor,
        )
        io_utils.export_nii(np.abs(self.image_gas_highsnr), "tmp/image_gas_highsnr.nii")
        io_utils.export_nii(
            np.abs(self.image_gas_highreso), "tmp/image_gas_highreso.nii"
        )

    def reconstruction_dissolved(self):
        """Reconstruct the dissolved phase image."""
        if self.config.recon.recon_key == constants.ReconKey.ROBERTSON.value:
            self.image_dissolved = reconstruction.reconstruct(
                data=(recon_utils.flatten_data(self.data_dissolved)),
                traj=recon_utils.flatten_traj(self.traj_dissolved),
                kernel_sharpness=float(self.config.recon.kernel_sharpness_lr),
                kernel_extent=9 * float(self.config.recon.kernel_sharpness_lr),
                image_size=int(self.config.recon.recon_size),
            )
            orientation = self.dict_dis[constants.IOFields.ORIENTATION]
            system_vendor = self.dict_dis[constants.IOFields.SYSTEM_VENDOR]
        elif self.config.recon.recon_key == constants.ReconKey.PLUMMER.value:
            raise NotImplementedError("Plummer CS reconstruction not implemented.")
        else:
            raise ValueError(f"Unknown reconstruction key")
        self.image_dissolved = img_utils.interp(
            self.image_dissolved,
            self.config.recon.matrix_size // self.config.recon.recon_size,
        )
        self.image_dissolved = img_utils.flip_and_rotate_image(
            self.image_dissolved,
            orientation=orientation,
            system_vendor=system_vendor,
        )

    def segmentation(self):
        """Segment the thoracic cavity."""
        if self.config.segmentation_key == constants.SegmentationKey.CNN_VENT.value:
            logging.info("Performing neural network segmenation.")
            self.mask = segmentation.predict(self.image_gas_highreso)
        elif self.config.segmentation_key == constants.SegmentationKey.SKIP.value:
            self.mask = np.ones_like(self.image_gas_highreso)
        elif (
            self.config.segmentation_key == constants.SegmentationKey.MANUAL_VENT.value
        ):
            logging.info("Loading mask file specified by the user.")
            try:
                self.mask = np.squeeze(
                    np.array(nib.load(self.config.manual_seg_filepath).get_fdata())
                ).astype(bool)
            except ValueError:
                logging.error("Invalid mask nifti file.")
        else:
            raise ValueError("Invalid segmentation key.")

    def registration(self):
        """Register moving image to target image.

        Uses ANTs registration to register the proton image to the xenon image.
        """
        if self.config.registration_key == constants.RegistrationKey.MASK2GAS.value:
            logging.info("Run registration algorithm, vent is fixed, mask is moving")
            self.mask, self.image_proton_reg = np.abs(
                registration.register_ants(
                    abs(self.image_gas_highreso), self.mask.astype(np.int8), self.image_proton
                )
            )
        elif self.config.registration_key == constants.RegistrationKey.PROTON2GAS.value:
            logging.info("Run registration algorithm, vent is fixed, proton is moving")
            self.image_proton_reg, mask = np.abs(
                registration.register_ants(
                    abs(self.image_gas_highreso), self.image_proton, self.mask.astype(np.int8)
                )
            )
            if (
                self.config.segmentation_key
                == constants.SegmentationKey.CNN_PROTON.value
                or self.config.segmentation_key
                == constants.SegmentationKey.MANUAL_PROTON.value
            ):
                self.mask = mask
        elif self.config.registration_key == constants.RegistrationKey.MANUAL.value:
            # Load a file specified by the user
            try:
                proton_reg = glob.glob(self.config.manual_reg_filepath)[0]
                self.image_proton_reg = np.squeeze(
                    np.array(nib.load(proton_reg).get_fdata())
                )
            except ValueError:
                logging.error("Invalid proton nifti file.")
        elif self.config.registration_key == constants.RegistrationKey.SKIP.value:
            logging.info("No registration, setting registered proton to proton")
            self.image_proton_reg = self.image_proton
        else:
            raise ValueError("Invalid registration key.")

    def biasfield_correction(self):
        """Correct ventilation image for bias field."""
        if self.config.bias_key == constants.BiasfieldKey.SKIP.value:
            logging.info("Skipping bias field correction.")
            self.image_gas_cor = abs(self.image_gas_highreso)
            self.image_biasfield = np.ones(self.image_gas_highreso.shape)
        elif self.config.bias_key == constants.BiasfieldKey.N4ITK.value:
            logging.info("Performing N4ITK bias field correction.")
            (
                self.image_gas_cor,
                self.image_biasfield,
            ) = biasfield.correct_biasfield_n4itk(
                image=abs(self.image_gas_highreso),
                mask=self.mask.astype(bool),
            )
        else:
            raise ValueError("Invalid bias field correction key.")

    def gas_binning(self):
        """Bin gas images to colormap bins."""
        self.image_gas_binned = binning.linear_bin(
            image=img_utils.normalize(self.image_gas_cor, self.mask),
            mask=self.mask,
            thresholds=self.reference_data['threshold_vent'],
        )
        self.mask_vent = np.logical_and(self.image_gas_binned > 1, self.mask)

    def dixon_decomposition(self):
        """Perform Dixon decomposition on the dissolved-phase images."""
        self.image_rbc, self.image_membrane = img_utils.dixon_decomposition(
            image_gas=self.image_gas_highsnr,
            image_dissolved=self.image_dissolved,
            mask=self.mask_vent,
            rbc_m_ratio=self.rbc_m_ratio,
        )

    def hb_correction(self):
        """Apply hemoglobin correction."""
        if self.config.hb_correction_key != constants.HbCorrectionKey.NONE.value:
            if self.config.hb > 0:
                # get hb correction scaling factors
                (
                    self.rbc_hb_correction_factor,
                    self.membrane_hb_correction_factor,
                ) = signal_utils.get_hb_correction(self.config.hb)

                # if only applying correction to rbc signal, set membrane factor to 1
                if (
                    self.config.hb_correction_key
                    == constants.HbCorrectionKey.RBC_ONLY.value
                ):
                    logging.info("Applying hemoglobin correction to RBC signal only")
                    self.membrane_hb_correction_factor = 1.0
                else:
                    logging.info(
                        "Applying hemoglobin correction to RBC and membrane signal"
                    )

                # scale dissolved phase signals by hb correction scaling factors
                self.rbc_m_ratio *= (
                    self.rbc_hb_correction_factor / self.membrane_hb_correction_factor
                )
                self.image_rbc *= self.rbc_hb_correction_factor
                self.image_membrane *= self.membrane_hb_correction_factor
            else:
                raise ValueError("Invalid hemoglobin value")
        else:
            logging.info("Skipping hemoglobin correction")

    def dissolved_analysis(self):
        """Calculate the dissolved-phase images relative to gas image."""
        self.image_rbc2gas = img_utils.divide_images(
            image1=self.image_rbc,
            image2=np.abs(self.image_gas_highsnr),
            mask=self.mask_vent,
        )
        self.image_membrane2gas = img_utils.divide_images(
            image1=self.image_membrane,
            image2=np.abs(self.image_gas_highsnr),
            mask=self.mask_vent,
        )
        # scale by flip angle difference
        flip_angle_scale_factor = signal_utils.calculate_flipangle_correction(
            self.dict_dis[constants.IOFields.FA_GAS],
            self.dict_dis[constants.IOFields.FA_DIS],
        )
        t2star_scale_factor_rbc = signal_utils.calculate_t2star_correction(
            self.dict_dis[constants.IOFields.TE90],
            constants.T2STAR_RBC_3T,
            self.dict_dis[constants.IOFields.FIELD_STRENGTH],
        )
        t2star_scale_factor_membrane = signal_utils.calculate_t2star_correction(
            self.dict_dis[constants.IOFields.TE90],
            constants.T2STAR_MEMBRANE_3T,
            self.dict_dis[constants.IOFields.FIELD_STRENGTH],
        )
        self.image_rbc2gas = (
            flip_angle_scale_factor * t2star_scale_factor_rbc * self.image_rbc2gas
        )
        self.image_membrane2gas = (
            flip_angle_scale_factor
            * t2star_scale_factor_membrane
            * self.image_membrane2gas
        )

    def dissolved_binning(self):
        """Bin dissolved images to colormap bins."""
        self.image_rbc2gas_binned = binning.linear_bin(
            image=self.image_rbc2gas,
            mask=self.mask_vent,
            thresholds=self.reference_data["threshold_rbc"],
        )
        self.image_membrane2gas_binned = binning.linear_bin(
            image=self.image_membrane2gas,
            mask=self.mask_vent,
            thresholds=self.reference_data["threshold_membrane"],
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate image statistics.

        Returns:
            dict_stats: Dictionary of statistics for reporting
        """
        self.dict_stats = {
            constants.IOFields.SUBJECT_ID: self.config.subject_id,
            constants.IOFields.SCAN_DATE: self.dict_dis[constants.IOFields.SCAN_DATE],
            constants.IOFields.PROCESS_DATE: metrics.process_date(),
            constants.StatsIOFields.INFLATION: metrics.inflation_volume(
                self.mask, self.dict_dis[constants.IOFields.FOV]
            ),
            constants.StatsIOFields.RBC_M_RATIO: self.rbc_m_ratio,
            constants.StatsIOFields.VENT_SNR: metrics.snr(
                np.abs(self.image_gas_highreso), self.mask
            )[1],
            constants.StatsIOFields.VENT_DEFECT_PCT: metrics.bin_percentage(
                self.image_gas_binned, np.array([1]), self.mask
            ),
            constants.StatsIOFields.VENT_LOW_PCT: metrics.bin_percentage(
                self.image_gas_binned, np.array([2]), self.mask
            ),
            constants.StatsIOFields.VENT_HIGH_PCT: metrics.bin_percentage(
                self.image_gas_binned, np.array([5, 6]), self.mask
            ),
            constants.StatsIOFields.VENT_MEAN: metrics.mean(
                img_utils.normalize(np.abs(self.image_gas_cor), self.mask), self.mask
            ),
            constants.StatsIOFields.VENT_MEDIAN: metrics.median(
                img_utils.normalize(np.abs(self.image_gas_cor), self.mask), self.mask
            ),
            constants.StatsIOFields.VENT_STDDEV: metrics.std(
                img_utils.normalize(np.abs(self.image_gas_cor), self.mask), self.mask
            ),
            constants.StatsIOFields.RBC_SNR: metrics.snr(self.image_rbc, self.mask)[0],
            constants.StatsIOFields.RBC_DEFECT_PCT: metrics.bin_percentage(
                self.image_rbc2gas_binned, np.array([1]), self.mask
            ),
            constants.StatsIOFields.RBC_LOW_PCT: metrics.bin_percentage(
                self.image_rbc2gas_binned, np.array([2]), self.mask
            ),
            constants.StatsIOFields.RBC_HIGH_PCT: metrics.bin_percentage(
                self.image_rbc2gas_binned, np.array([5, 6]), self.mask
            ),
            constants.StatsIOFields.RBC_MEAN: metrics.mean(
                self.image_rbc2gas, self.mask_vent
            ),
            constants.StatsIOFields.RBC_MEDIAN: metrics.median(
                self.image_rbc2gas, self.mask_vent
            ),
            constants.StatsIOFields.RBC_STDDEV: metrics.std(
                self.image_rbc2gas, self.mask_vent
            ),
            constants.StatsIOFields.MEMBRANE_SNR: metrics.snr(
                self.image_membrane, self.mask
            )[0],
            constants.StatsIOFields.MEMBRANE_DEFECT_PCT: metrics.bin_percentage(
                self.image_membrane2gas_binned, np.array([1]), self.mask
            ),
            constants.StatsIOFields.MEMBRANE_LOW_PCT: metrics.bin_percentage(
                self.image_membrane2gas_binned, np.array([2]), self.mask
            ),
            constants.StatsIOFields.MEMBRANE_HIGH_PCT: metrics.bin_percentage(
                self.image_membrane2gas_binned, np.array([6, 7, 8]), self.mask
            ),
            constants.StatsIOFields.MEMBRANE_MEAN: metrics.mean(
                self.image_membrane2gas, self.mask_vent
            ),
            constants.StatsIOFields.MEMBRANE_MEDIAN: metrics.median(
                self.image_membrane2gas, self.mask_vent
            ),
            constants.StatsIOFields.MEMBRANE_STDDEV: metrics.std(
                self.image_membrane2gas, self.mask_vent
            ),
            constants.StatsIOFields.ALVEOLAR_VOLUME: metrics.alveolar_volume(
                self.image_gas_binned, self.mask, self.dict_dis[constants.IOFields.FOV]
            ),
            constants.StatsIOFields.KCO_EST: metrics.kco(
                self.image_membrane2gas,
                self.image_rbc2gas,
                self.mask_vent,
                self.reference_data['reference_fit_membrane'][1],
                self.reference_data['reference_fit_rbc'][1],
            ),
            constants.StatsIOFields.DLCO_EST: metrics.dlco(
                self.image_gas_binned,
                self.image_membrane2gas,
                self.image_rbc2gas,
                self.mask,
                self.mask_vent,
                self.dict_dis[constants.IOFields.FOV],
                self.reference_data['reference_fit_membrane'][1],
                self.reference_data['reference_fit_rbc'][1],
            ),
        }
        return self.dict_stats

    def get_info(self) -> Dict[str, Any]:
        """Gather information about the data and processing steps.

        Returns:
            dict_info: Dictionary of information.
        """
        self.dict_info = {
            constants.IOFields.SUBJECT_ID: self.config.subject_id,
            constants.IOFields.SCAN_DATE: self.dict_dis[constants.IOFields.SCAN_DATE],
            constants.IOFields.PROCESS_DATE: metrics.process_date(),
            constants.IOFields.SCAN_TYPE: self.config.recon.scan_type,
            constants.IOFields.PIPELINE_VERSION: constants.PipelineVersion.VERSION_NUMBER,
            constants.IOFields.SOFTWARE_VERSION: self.dict_dis[
                constants.IOFields.SOFTWARE_VERSION
            ],
            constants.IOFields.GIT_BRANCH: report.get_git_branch(),
            constants.IOFields.REFERENCE_DATA_KEY: self.reference_data['title'],
            constants.IOFields.BANDWIDTH: self.dict_dis[constants.IOFields.BANDWIDTH],
            constants.IOFields.SAMPLE_TIME: (
                1e6 * self.dict_dis[constants.IOFields.SAMPLE_TIME]
            ),
            constants.IOFields.FA_DIS: self.dict_dis[constants.IOFields.FA_DIS],
            constants.IOFields.FA_GAS: self.dict_dis[constants.IOFields.FA_GAS],
            constants.IOFields.FIELD_STRENGTH: self.dict_dis[
                constants.IOFields.FIELD_STRENGTH
            ],
            constants.IOFields.FLIP_ANGLE_FACTOR: signal_utils.calculate_flipangle_factor(
                self.dict_dis[constants.IOFields.FA_GAS],
                self.dict_dis[constants.IOFields.FA_DIS],
            ),
            constants.IOFields.FOV: self.dict_dis[constants.IOFields.FOV],
            constants.IOFields.XE_DISSOLVED_OFFSET_FREQUENCY: self.dict_dis[
                constants.IOFields.XE_DISSOLVED_OFFSET_FREQUENCY
            ],
            constants.IOFields.GRAD_DELAY_X: self.dict_dis[
                constants.IOFields.GRAD_DELAY_X
            ],
            constants.IOFields.GRAD_DELAY_Y: self.dict_dis[
                constants.IOFields.GRAD_DELAY_Y
            ],
            constants.IOFields.GRAD_DELAY_Z: self.dict_dis[
                constants.IOFields.GRAD_DELAY_Z
            ],
            constants.IOFields.HB_CORRECTION_KEY: self.config.hb_correction_key,
            constants.IOFields.HB: self.config.hb,
            constants.IOFields.RBC_HB_CORRECTION_FACTOR: self.rbc_hb_correction_factor,
            constants.IOFields.MEMBRANE_HB_CORRECTION_FACTOR: self.membrane_hb_correction_factor,
            constants.IOFields.KERNEL_SHARPNESS: self.config.recon.kernel_sharpness_hr,
            constants.IOFields.N_SKIP_START: self.config.recon.n_skip_start,
            constants.IOFields.N_DIS_REMOVED: len(
                self.dict_dis[constants.IOFields.FIDS_DIS]
            )
            - np.sum(
                recon_utils.get_noisy_projections(
                    data=self.dict_dis[constants.IOFields.FIDS_DIS]
                )
            ),
            constants.IOFields.N_GAS_REMOVED: len(
                self.dict_dis[constants.IOFields.FIDS_GAS]
            )
            - np.sum(
                recon_utils.get_noisy_projections(
                    data=self.dict_dis[constants.IOFields.FIDS_GAS]
                )
            ),
            constants.IOFields.REMOVE_NOISE: self.config.recon.remove_noisy_projections,
            constants.IOFields.SHAPE_FIDS: self.dict_dis[constants.IOFields.FIDS].shape,
            constants.IOFields.SHAPE_IMAGE: self.image_gas_highreso.shape,
            constants.IOFields.T2_CORRECTION_FACTOR_MEMBRANE: signal_utils.calculate_t2star_correction(
                self.dict_dis[constants.IOFields.TE90],
                constants.T2STAR_MEMBRANE_3T,
                self.dict_dis[constants.IOFields.FIELD_STRENGTH],
            ),
            constants.IOFields.T2_CORRECTION_FACTOR_RBC: signal_utils.calculate_t2star_correction(
                self.dict_dis[constants.IOFields.TE90],
                constants.T2STAR_RBC_3T,
                self.dict_dis[constants.IOFields.FIELD_STRENGTH],
            ),
            constants.IOFields.TE90: 1e6 * self.dict_dis[constants.IOFields.TE90],
            constants.IOFields.TR_DIS: 1e3 * self.dict_dis[constants.IOFields.TR],
        }
        return self.dict_info

    def generate_figures(self):
        """Export image figures."""
        #Issues with index_start, index_skip, index_end; all equal 0
        #everything fine up to this point as far as I can tell
        index_start, index_skip = plot.get_plot_indices(self.mask)
        proton_reg = img_utils.normalize(
            np.abs(self.image_proton),
            self.mask,
            method=constants.NormalizationMethods.PERCENTILE,
        )
        plot.plot_montage_grey(
            image=np.abs(self.image_gas_highreso),
            path="tmp/montage_vent.png",
            index_start=index_start,
            index_skip=index_skip,
        )
        plot.plot_montage_grey(
            image=np.abs(self.image_membrane),
            path="tmp/montage_membrane.png",
            index_start=index_start,
            index_skip=index_skip,
        )
        plot.plot_montage_grey(
            image=np.abs(self.image_rbc),
            path="tmp/montage_rbc.png",
            index_start=index_start,
            index_skip=index_skip,
        )
        plot.plot_montage_color(
            image=plot.map_and_overlay_to_rgb(
                self.image_gas_binned, proton_reg, constants.CMAP.VENT_BIN2COLOR
            ),
            path="tmp/montage_gas_binned.png",
            index_start=index_start,
            index_skip=index_skip,
        )
        plot.plot_montage_color(
            image=plot.map_and_overlay_to_rgb(
                self.image_rbc2gas_binned, proton_reg, constants.CMAP.RBC_BIN2COLOR
            ),
            path="tmp/montage_rbc_binned.png",
            index_start=index_start,
            index_skip=index_skip,
        )
        plot.plot_montage_color(
            image=plot.map_and_overlay_to_rgb(
                self.image_membrane2gas_binned,
                proton_reg,
                constants.CMAP.MEMBRANE_BIN2COLOR,
            ),
            path="tmp/montage_membrane_binned.png",
            index_start=index_start,
            index_skip=index_skip,
        )
        plot.plot_montage_color(
            image=plot.overlay_mask_on_image(proton_reg, self.mask.astype("uint8")),
            path="tmp/montage_proton_qa.png",
            index_start=index_start,
            index_skip=index_skip,
        )
        plot.plot_montage_color(
            image=plot.overlay_mask_on_image(
                np.abs(self.image_gas_highreso), self.mask.astype("uint8")
            ),
            path="tmp/montage_vent_qa.png",
            index_start=index_start,
            index_skip=index_skip,
        )
        plot.plot_montage_color(
            image=plot.overlay_mask_on_image(
                np.abs(self.image_dissolved), self.mask.astype("uint8")
            ),
            path="tmp/montage_dissolved_qa.png",
            index_start=index_start,
            index_skip=index_skip,
        )
        plot.plot_histogram(
            data=img_utils.normalize(self.image_gas_cor, self.mask)[
                np.array(self.mask, dtype=bool)
            ].flatten(),
            path="tmp/hist_vent.png",
            color=constants.VENTHISTOGRAMFields.COLOR,
            xlim=constants.VENTHISTOGRAMFields.XLIM,
            ylim=constants.VENTHISTOGRAMFields.YLIM,
            num_bins=constants.VENTHISTOGRAMFields.NUMBINS,
            refer_fit=self.reference_data['reference_fit_vent'],
            xticks=constants.VENTHISTOGRAMFields.XTICKS,
            yticks=constants.VENTHISTOGRAMFields.YTICKS,
            xticklabels=constants.VENTHISTOGRAMFields.XTICKLABELS,
            yticklabels=constants.VENTHISTOGRAMFields.YTICKLABELS,
            title=constants.VENTHISTOGRAMFields.TITLE,
        )
        plot.plot_histogram(
            data=np.abs(self.image_rbc2gas)[
                np.array(self.mask_vent, dtype=bool)
            ].flatten(),
            path="tmp/hist_rbc.png",
            color=constants.RBCHISTOGRAMFields.COLOR,
            xlim=constants.RBCHISTOGRAMFields.XLIM,
            ylim=constants.RBCHISTOGRAMFields.YLIM,
            num_bins=constants.RBCHISTOGRAMFields.NUMBINS,
            refer_fit=self.reference_data['reference_fit_rbc'],
            xticks=constants.RBCHISTOGRAMFields.XTICKS,
            yticks=constants.RBCHISTOGRAMFields.YTICKS,
            xticklabels=constants.RBCHISTOGRAMFields.XTICKLABELS,
            yticklabels=constants.RBCHISTOGRAMFields.YTICKLABELS,
            title=constants.RBCHISTOGRAMFields.TITLE,
        )
        plot.plot_histogram(
            data=np.abs(self.image_membrane2gas)[
                np.array(self.mask_vent, dtype=bool)
            ].flatten(),
            path="tmp/hist_membrane.png",
            color=constants.MEMBRANEHISTOGRAMFields.COLOR,
            xlim=constants.MEMBRANEHISTOGRAMFields.XLIM,
            ylim=constants.MEMBRANEHISTOGRAMFields.YLIM,
            num_bins=constants.MEMBRANEHISTOGRAMFields.NUMBINS,
            refer_fit=self.reference_data['reference_fit_membrane'],
            xticks=constants.MEMBRANEHISTOGRAMFields.XTICKS,
            yticks=constants.MEMBRANEHISTOGRAMFields.YTICKS,
            xticklabels=constants.MEMBRANEHISTOGRAMFields.XTICKLABELS,
            yticklabels=constants.MEMBRANEHISTOGRAMFields.YTICKLABELS,
            title=constants.MEMBRANEHISTOGRAMFields.TITLE,
        )

    def generate_pdf(self):
        """Generate HTML and PDF files."""
        # generate individual PDFs
        pdf_list = [
            os.path.join("tmp", pdf)
            for pdf in ["intro.pdf", "clinical.pdf", "grayscale.pdf", "qa"]
        ]
        report.intro(self.dict_info, path=pdf_list[0])
        report.clinical(
            {**self.dict_stats, **self.reference_data['reference_stats']},
            path=pdf_list[1],
        )
        report.grayscale(
            {**self.dict_stats, **self.reference_data['reference_stats']},
            path=pdf_list[2],
        )
        report.qa(
            {**self.dict_stats, **self.reference_data['reference_stats']},
            path=pdf_list[3],
        )

        # combine PDFs into one
        path = "tmp/{}_report.pdf".format(self.config.subject_id)
        report.combine_pdfs(pdf_list, path)

    def write_stats_to_csv(self):
        """Write statistics to file."""
        # write to combined csv of recently processed subjects
        io_utils.export_subject_csv(
            {**self.dict_info, **self.dict_stats}, path="data/stats_all.csv"
        )

        # write to individual subject csv
        io_utils.export_subject_csv(
            {**self.dict_info, **self.dict_stats},
            path="tmp/{}_stats.csv".format(self.config.subject_id),
            overwrite=True,
        )

    def save_subject_to_mat(self):
        """Save the instance variables into a mat file."""
        path = os.path.join("tmp", self.config.subject_id + ".mat")
        io_utils.export_subject_mat(self, path)

    def save_files(self):
        """Save select images to nifti files and instance variable to mat."""
        proton_reg = img_utils.normalize(
            np.abs(self.image_proton),
            self.mask,
            method=constants.NormalizationMethods.PERCENTILE,
        )
        io_utils.export_nii(
            self.image_rbc2gas_binned,
            "tmp/rbc_binned.nii",
            self.dict_dis[constants.IOFields.FOV],
        )
        io_utils.export_nii(
            np.abs(self.image_gas_highreso),
            "tmp/gas_highreso.nii",
            self.dict_dis[constants.IOFields.FOV],
        )
        io_utils.export_nii(
            np.abs(self.image_gas_highsnr),
            "tmp/gas_highsnr.nii",
            self.dict_dis[constants.IOFields.FOV],
        )
        io_utils.export_nii(
            np.abs(self.image_rbc),
            "tmp/rbc.nii",
            self.dict_dis[constants.IOFields.FOV],
        )
        io_utils.export_nii(
            np.abs(self.image_membrane),
            "tmp/membrane.nii",
            self.dict_dis[constants.IOFields.FOV],
        )
        io_utils.export_nii(
            np.abs(self.image_membrane2gas),
            "tmp/membrane2gas.nii",
            self.dict_dis[constants.IOFields.FOV],
        )
        io_utils.export_nii(
            self.mask.astype(float),
            "tmp/mask_reg.nii",
            self.dict_dis[constants.IOFields.FOV],
        )
        io_utils.export_nii(
            np.abs(self.image_dissolved),
            "tmp/dissolved.nii",
            self.dict_dis[constants.IOFields.FOV],
        )
        if self.config.recon.recon_proton:
            io_utils.export_nii(
                np.abs(self.image_proton),
                "tmp/proton.nii",
                self.dict_dis[constants.IOFields.FOV],
            )
            io_utils.export_nii(
                np.abs(self.image_proton_reg),
                "tmp/proton_reg.nii",
                self.dict_dis[constants.IOFields.FOV],
            ),
        io_utils.export_nii_4d(
            plot.map_and_overlay_to_rgb(
                self.image_rbc2gas_binned, proton_reg, constants.CMAP.RBC_BIN2COLOR
            ),
            "tmp/rbc2gas_rgb.nii",
        )
        io_utils.export_nii_4d(
            plot.map_and_overlay_to_rgb(
                self.image_membrane2gas_binned,
                proton_reg,
                constants.CMAP.MEMBRANE_BIN2COLOR,
            ),
            "tmp/membrane2gas_rgb.nii",
        )
        io_utils.export_nii_4d(
            plot.map_and_overlay_to_rgb(
                self.image_gas_binned,
                proton_reg,
                constants.CMAP.VENT_BIN2COLOR,
            ),
            "tmp/gas_rgb.nii",
        )

    def save_config_as_json(self):
        """Save subject config .py file as json."""
        io_utils.export_config_to_json(
            self.config,
            "tmp/{}_config_gx_imaging.json".format(self.config.subject_id),
        )

    def move_output_files(self):
        """Move output files into dedicated directory."""
        # define files to move
        output_files = (
            "tmp/{}_config_gx_imaging.json".format(self.config.subject_id),
            "tmp/{}.mat".format(self.config.subject_id),
            "tmp/{}_report.pdf".format(self.config.subject_id),
            "tmp/{}_stats.csv".format(self.config.subject_id),
            "tmp/gas_highreso.nii",
            "tmp/gas_rgb.nii",
            "tmp/mask_reg.nii",
            "tmp/membrane2gas_rgb.nii",
            "tmp/proton_reg.nii",
            "tmp/rbc2gas_rgb.nii",
        )

        # move files
        io_utils.move_files(output_files, self.config.data_dir)



