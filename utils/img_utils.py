"""Miscellaneous util functions mostly image processing."""

import os
import sys

import cv2

sys.path.append("..")
from typing import Any, List, Optional, Tuple

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import skimage
from scipy import ndimage

from utils import constants, io_utils
import nibabel as nb


def remove_small_objects(mask: np.ndarray, scale: float = 0.1):
    """Remove small unconnected voxels in the mask.

    Args:
        mask (np.ndarray): boolean mask
        scale (float, optional): scalaing factor to determin minimum size.
            Defaults to 0.015.

    Returns:
        Mask with the unconnected voxels removed
    """
    min_size = np.sum(mask) * scale
    return skimage.morphology.remove_small_objects(
        ar=mask, min_size=min_size, connectivity=1
    ).astype("bool")


def flip_image_complex(image: np.ndarray) -> np.ndarray:
    """Flip image of complex type along all axes.

    Args:
        image (np.ndarray): image to flip
    Returns:
        Flipped image.
    """
    return np.flip(np.flip(np.flip(np.transpose(image, (2, 1, 0)), 0), 1), 2)


def rotate_axial_to_coronal(image: np.ndarray) -> np.ndarray:
    """Rotate axial image to coronal.

    Image is assumed to be of complex datatype.

    Args:
        image (np.ndarray): image viewed in axial orientation.
    Returns:
        Rotated coronal image.
    """
    real = ndimage.rotate(ndimage.rotate(np.real(image), 90, (1, 2)), 270)
    imag = ndimage.rotate(ndimage.rotate(np.imag(image), 90, (1, 2)), 270)
    return real + 1j * imag


def flip_and_rotate_image(
    image: np.ndarray, orientation: str = constants.Orientation.CORONAL,
    system_vendor: str = constants.SystemVendor.SIEMENS
) -> np.ndarray:
    """Flip and rotate image based on orientation.

    Args:
        image (np.ndarray): image to flip and rotate.
        orientation (str, optional): orientation of the image. Defaults to coronal.
    Returns:
        Flipped and rotated image.
    """
    
    #if system_vendor == constants.SystemVendor.SIEMENS.value:
    if orientation == constants.Orientation.CORONAL:
        image = np.rot90(np.rot90(image, 3, axes=(1, 2)), 1, axes=(0, 2))
        image = np.rot90(image, 1, axes=(0, 1))
        image = np.flip(np.flip(image, axis=1), axis=2)
        return image
    elif orientation == constants.Orientation.TRANSVERSE:
        return rotate_axial_to_coronal(flip_image_complex(image))
    elif orientation == constants.Orientation.AXIAL:
        image = np.rot90(np.rot90(image, 1, axes=(1, 2)), 3, axes=(0, 2))
        image = np.rot90(image, 1, axes=(0, 1))
        image = np.flip(image, axis=2)
        return image
    elif orientation == constants.Orientation.NONE:
        return image
    elif system_vendor == constants.SystemVendor.PHILIPS.value:
        if orientation == constants.Orientation.CORONAL:
            image = np.rot90(np.rot90(image, 3, axes=(1, 2)), 1, axes=(0, 2))
            image = np.flip(image, axis=2)
            return image
        else:
            raise ValueError("Invalid orientation: {}.".format(orientation))
    elif system_vendor == constants.SystemVendor.GE.value:
        if orientation == constants.Orientation.CORONAL:
            def complex_rot_axial_iowa(x):
                from scipy.ndimage import rotate

                real = rotate(np.real(x), 180, (1, 2))
                imag = rotate(np.imag(x), 180, (1, 2))
                return real + 1j * imag
            def complex_align(x):
                return np.flip(np.flip(np.flip(np.transpose(x, (2, 1, 0)), 0), 1), 2)

            image = complex_rot_axial_iowa(complex_align(image))
            image= np.flip(image, axis=0)
            return image
        else:
            raise ValueError("Invalid orientation: {}.".format(orientation))
    else:
        raise ValueError("Invalid system_vendor: {}.".format(system_vendor))


def standardize_image(image: np.ndarray) -> np.ndarray:
    """Standardize image.

    Args:
        image (np.ndarray): image to standardize.
    Returns:
        Standardized image.
    """
    image = np.abs(image)
    image = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
    image = (image - np.mean(image)) / np.std(image)
    return image


def erode_image(image: np.ndarray, erosion: int) -> np.ndarray:
    """Erode image.

    Erodes image slice by slice.

    Args:
        image (np.ndarray): 3-D image to erode.
        erosion (int): size of erosion kernel.
    Returns:
        Eroded image.
    """
    kernel = np.ones((erosion, erosion), np.uint8)
    for i in range(image.shape[2]):
        image[:, :, i] = cv2.erode(image[:, :, i], kernel, iterations=1)
    return image


def divide_images(
    image1: np.ndarray, image2: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Divide image1 by image2 inside the masked region.

    Sets negative values to 0.

    Args:
        image1 (np.ndarray): image to divide.
        image2 (np.ndarray): image to divide by.
        mask (np.ndarray, optional): boolean mask to apply to the image. Defaults to None.
    Returns:
        Divided image.
    """
    out = np.zeros_like(image1)
    if isinstance(mask, np.ndarray):
        out[mask] = np.divide(image1[mask], image2[mask])
    else:
        out = np.divide(image1, image2)
    # set negative values to 0
    out[out < 0] = 0
    return out


def smooth_image(image: np.ndarray, kernel: int = 11) -> np.ndarray:
    """Smooth the image using a blurring kernel.

    Args:
        image (np.ndarray): 3D image to smooth.
        kernel (int, optional): size of the kernel. Defaults to 11.
    """
    kernel = np.ones((kernel, kernel, kernel)) / (kernel**3)  # type: ignore
    return ndimage.convolve(image, kernel, mode="constant")


def interp(img: np.ndarray, factor: int = 1):
    """Interpolate the image to be of size factor times the original size.

    Args:
        img (np.ndarray): image to interpolate
        factor (int): factor to interpolate by
    """
    img_real = ndimage.zoom(np.real(img), [factor, factor, factor])
    img_imag = ndimage.zoom(np.imag(img), [factor, factor, factor])
    return img_real + 1j * img_imag


def normalize(
    image: np.ndarray,
    mask: np.ndarray = np.array([0.0]),
    method: str = constants.NormalizationMethods.FRAC_VENT,  # default method = PERCENTILE_MASKED
    percentile: float = 99.0,
    bag_volume: float = None # Add bag_volume as a parameter with a default value
) -> np.ndarray:
    """Normalize the image to be between [0, 1.0].

    Args:
        image (np.ndarray): image matrix
        method (int): normalization method
        mask (np.ndarray): boolean mask
        percentile (np.ndarray): if normalization via max percentile, scale by
            percentile and set everything else to 1.0

    Returns:
        np.ndarray: normalized image
    """
    if bag_volume is None:
        raise ValueError("You must provide a numeric value for `bag_volume`.")
    elif method == constants.NormalizationMethods.MAX:
        return image * 1.0 / np.max(image)
    elif method == constants.NormalizationMethods.PERCENTILE:
        return image * 1.0 / np.percentile(image, percentile)
    elif method == constants.NormalizationMethods.PERCENTILE_MASKED:
        image_thre = np.percentile(image[mask], percentile)
        image_n = np.divide(np.multiply(image, mask), image_thre)
        image_n[image_n > 1] = 1
        return image_n
    elif method == constants.NormalizationMethods.MEAN:
        image[np.isnan(image)] = 0
        image[np.isinf(image)] = 0
        return image / np.mean(image[mask])
    # frac_vent_modifications
    elif method == constants.NormalizationMethods.FRAC_VENT:
        # Correction factor for upper airways and dead tube space. This is very much a guess.
        tcv_gas_vol = bag_volume - 10 #new estimate in mL
        # Have the 10 on the constants 
        

        # What is mask volume in ml?
        #in constants 
        voxel_side = .3125  # cm
        voxel_vol = voxel_side**3  # cm^3 = ml

        # What is the amount of signal within the lung?
        vent_img_mask = image.copy()  # Start with original vent image

        print("vent_img_mask shape:", vent_img_mask.shape)
        print("mask shape:", mask.shape)

        vent_img_mask[mask == 0] = 0.0  # Zero out the non-lung voxels
        

        print('image' + str(np.sum(image)))
        print('vent image mask' + str(np.sum(vent_img_mask)))

        #signal_total = np.sum(image)  # sum it all up

        #previous
        signal_total = np.sum(vent_img_mask)  # sum it all up

        # What is the conversion factor between signal and volume? semi-big assumption here
        # Framing it as volume/signal so we can multiply by a voxel signal and get a volume
        sig_vol_rat = tcv_gas_vol / signal_total

        #up until this part is using the big mask 

        # Fractional Vent in a voxel
        # frac_vent = New gas volume / (old volume + new volume -- but that's just the voxel volume?)
        frac_vent = (image * sig_vol_rat) / voxel_vol

        # Create a NIfTI image
        nifti_img = nb.Nifti1Image(frac_vent, affine=np.eye(4))

        # Save the image to a file
        nifti_img.to_filename('tmp/frac_vent_output.nii')

        # New volume is just proportional to signal
        # Flatten the 3D array to 1D for plotting the histogram
        # Create a masked image (zero elsewhere) for NIfTI
        frac_vent_masked = np.zeros_like(frac_vent)
        frac_vent_masked[mask == 1] = frac_vent[mask == 1]
        frac_vent_mask_nifti_img = nb.Nifti1Image(frac_vent_masked, affine=np.eye(4))
        frac_vent_mask_nifti_img.to_filename('tmp/frac_vent_mask.nii')

        # For histogram/stats:
        frac_vent_mask = frac_vent[mask == 1]
        flat_array = frac_vent_mask.flatten()

        # and the mean
        mean = np.mean(flat_array)

        return frac_vent

    else:
        raise ValueError("Invalid normalization method")


def correct_b0(
    image: np.ndarray, mask: np.ndarray, max_iterations: int = 100
) -> np.ndarray:
    """Correct B0 inhomogeneity.

    Args:
        image (np.ndarray): image to correct.
        mask (np.ndarray): mask of the image. must be same shape as image.
        max_iterations (int, optional): maximum number of iterations. Defaults to 20.
    Returns:
        Corrected phase of the corrected image.
    """
    index = 0
    meanphase = 1

    while abs(meanphase) > 1e-7:
        index = index + 1
        diffphase = np.angle(image)
        meanphase = np.mean(diffphase[mask])  # type: ignore
        image = np.multiply(image, np.exp(-1j * meanphase))
        if index > max_iterations:
            break
    return np.angle(image)  # type: ignore


def dixon_decomposition(
    image_gas: np.ndarray,
    image_dissolved: np.ndarray,
    mask: np.ndarray,
    rbc_m_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply 1-point dixon decomposition on images.

    Applies phase shift to the dissolved image such that the RBC and membrane are
    separated into the imaginary and real channel respectively. Also applies B0
    inhomogeneity correction by shifting the gas image to have zero mean phase and
    applying the same phase shift to the dissolved image.

    Args:
        image_gas (np.ndarray): gas image
        image_dissolved (np.ndarray): dissolved image
        mask (np.ndarray): boolean mask of the lung. must be the same size as the images.
        rbc_m_ratio (float): RBC:m ratio
    Returns:
        Tuple of decomposed RBC and membrane images respectively.
    """
    # correct for B0 inhomogeneity
    diffphase = correct_b0(image_gas, mask)
    # calculate phase shift to separate RBC and membrane
    desired_angle = np.arctan2(rbc_m_ratio, 1.0)
    current_angle = np.angle(np.sum(image_dissolved[mask > 0]))
    delta_angle = desired_angle - current_angle
    image_dixon = np.multiply(image_dissolved, np.exp(1j * (delta_angle)))
    image_dixon = np.multiply(image_dixon, np.exp(1j * (-diffphase)))
    # separate RBC and membrane components
    image_rbc = (
        np.imag(image_dixon)
        if np.mean(np.imag(image_dixon)[mask]) > 0
        else -1 * np.imag(image_dixon)  # type: ignore
    )
    image_membrane = (
        np.real(image_dixon)
        if np.mean(np.real(image_dixon)[mask]) > 0
        else -1 * np.real(image_dixon)  # type: ignore
    )
    return image_rbc, image_membrane


def calculate_rbc_oscillation(
    image_high: np.ndarray,
    image_low: np.ndarray,
    image_total: np.ndarray,
    mask: np.ndarray,
    method: str = constants.Methods.SMOOTH,
) -> np.ndarray:
    """Calculate RBC oscillation.

    Args:
        image_high (np.ndarray): high rbc image.
        image_low (np.ndarray): low rbc image.
        image_total (np.ndarray): total image.
        mask(np.ndarray): booleaan mask of the lung. Must be the same size as the images.
        method (str): method to use for calculating oscillation.
    Returns:
        RBC oscillation image in percentage.
    """
    image_total = image_total.copy()
    image_total[mask == 0] = np.max(image_total[mask > 0])
    if method == constants.Methods.ELEMENTWISE:
        return 100 * np.subtract(image_high, image_low) / image_total
    elif method == constants.Methods.MEAN:
        return (
            100
            * np.subtract(image_high, image_low)
            / np.abs(np.mean(image_total[mask > 0]))
        )
    elif method == constants.Methods.SMOOTH:
        return 100 * np.subtract(image_high, image_low) / smooth_image(image_total)
    else:
        raise ValueError("Invalid method: {}.".format(method))


def approximate_image_with_bspline(
    image: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Approximate image with B-spline.

    Args:
        image (np.ndarray): image to approximate.
        mask (np.ndarray, optional): mask of the image. Defaults to None.
    Returns:
        Approximated image
    """
    current_path = os.path.dirname(__file__)
    tmp_path = os.path.abspath(os.path.join(current_path, "..", "tmp"))
    bin_path = os.path.abspath(os.path.join(current_path, "..", "bin"))
    pathInput = os.path.join(tmp_path, "image.nii")
    pathOutput = os.path.join(tmp_path, "bspline.nii")
    pathMask = os.path.join(tmp_path, "mask.nii")
    pathExec = bin_path + "/ApproximateImageWithBSplines"
    # save the inputs into nii files so the executable can read in
    io_utils.export_nii(image, pathInput)
    if isinstance(mask, np.ndarray):
        io_utils.export_nii(mask.astype(float), pathMask)  # type: ignore
    # command string
    cmd = pathExec + " 3 " + pathInput + " " + pathOutput + " 3 4 1 "
    if isinstance(mask, np.ndarray):
        cmd = cmd + pathMask
    # registration command
    os.system(cmd)
    # read in the output
    return io_utils.import_nii(pathOutput)
