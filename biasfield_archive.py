"""Bias field correction.

Currently supports N4ITK bias field correction.
"""
import os
from typing import Tuple

import nibabel as nib
import numpy as np
from absl import app, flags
from scipy import signal

from utils import constants, img_utils

FLAGS = flags.FLAGS
flags.DEFINE_string("image_file", "", "nifti image file path.")
flags.DEFINE_string("mask_file", "", "nifti mask file path.")
flags.DEFINE_string("output_path", "", "output folder location")


def calculate_flip_angle(
    image1: np.ndarray,
    image2: np.ndarray,
    n_proj: int,
    T1: float = np.inf,
    TR: float = 4.5,
) -> np.ndarray:
    """Flip angle map calculation using RF-depolarization.

    Source: https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.29254

    Args:
        image1 (np.ndarray): magnitude image of the first temporal subdivision.
        image2 (np.ndarray): magnitude image of the second temporal subdivision.
        n_proj (int): number of radial projections
        T1 (float): T1 value milli-seconds. Defaults to np.inf.
        TR (float): TR value in milli-seconds. Defaults to 4.5ms.
    Returns:
        Flip angle map in degrees.
    """
    ratio = np.float_power(np.abs(image2 / image1), 2 / n_proj)
    # apply median filter
    ratio = signal.medfilt(ratio, kernel_size=3)
    # set values above 1 to 1
    ratio[ratio > 1] = 1
    flip_angle_map = (180 / np.pi) * np.arccos(ratio) * np.exp(TR / T1)
    return signal.medfilt(flip_angle_map, kernel_size=3)


def calculate_biasfield_rf(
    image1: np.ndarray,
    image2: np.ndarray,
    mask: np.ndarray,
    n_proj: int,
    T1: float = np.inf,
    TR: float = 4.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate bias field map using RF-depolarization.

    Source: https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.29254
    Args:
        image1 (np.ndarray): magnitude image of the first temporal subdivision.
        image2 (np.ndarray): magnitude image of the second temporal subdivision.
        mask (np.ndarray): mask of the image.
        n_proj (int): number of radial projections
        T1 (float): T1 value milli-seconds. Defaults to np.inf.
        TR (float): TR value in milli-seconds. Defaults to 4.5ms.
    Returns:
        Tuple of bias field map and smoothed bias field map.
    """
    image_alpha = calculate_flip_angle(image1, image2, n_proj, T1, TR)
    c1 = np.cos(image_alpha * np.pi / 180) * np.exp(-TR / T1)
    # calculate analytic bias field
    image_biasfield = (
        image_alpha
        * np.sin(image_alpha * np.pi / 180)
        * np.divide((1 - np.power(c1, n_proj)), 1 - c1)
    )
    # normalize to the mean
    image_biasfield = img_utils.normalize(
        image_biasfield, mask, method=constants.NormalizationMethods.MEAN
    )
    image_biasfield_smoothed = img_utils.approximate_image_with_bspline(
        image_biasfield, mask
    )
    return image_biasfield, image_biasfield_smoothed


def correct_biasfield_n4itk(
    image: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply N4ITK bias field correction.

    Args:
        image: np.ndarray 3D image to apply n4itk bias field correction.
        mask: np.ndarray 3D mask for n4itk bias field correcton.
    """
    current_path = os.path.dirname(__file__)
    tmp_path = os.path.join(current_path, "tmp")
    bin_path = os.path.join(current_path, "bin")

    pathInput = os.path.join(tmp_path, "image.nii")
    pathMask = os.path.join(tmp_path, "mask.nii")
    pathOutput = os.path.join(tmp_path, "image_cor.nii")
    pathBiasField = os.path.join(tmp_path, "biasfield.nii")

    pathN4 = os.path.join(bin_path, "N4BiasFieldCorrection")
    # save the inputs into nii files so the execute N4 can read in
    nii_imge = nib.Nifti1Image(np.abs(image), np.eye(4))
    nii_mask = nib.Nifti1Image(mask.astype(float), np.eye(4))
    nib.save(nii_imge, pathInput)
    nib.save(nii_mask, pathMask)
    cmd = (
        pathN4
        + " -d 3 -i "
        + pathInput
        + " -s 2 -x "
        + pathMask
        + " -o ["
        + pathOutput
        + ", "
        + pathBiasField
        + "]"
    )

    os.system(cmd)

    image_cor = np.array(nib.load(pathOutput).get_fdata())
    image_biasfield = np.array(nib.load(pathBiasField).get_fdata())

    # remove the generated nii files
    os.remove(pathInput)
    os.remove(pathMask)
    os.remove(pathOutput)
    os.remove(pathBiasField)

    return image_cor.astype("float64"), image_biasfield.astype("float64")


def correct_biasfield_rf(
    image: np.ndarray,
    image1: np.ndarray,
    image2: np.ndarray,
    mask: np.ndarray,
    n_proj: int,
    T1: float = np.inf,
    TR: float = 4.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply RF-depolarization bias field correction.

    Args:
        image (np.ndarray): magnitude image.
        image1 (np.ndarray): magnitude image of the first temporal subdivision.
        image2 (np.ndarray): magnitude image of the second temporal subdivision.
        mask (np.ndarray): mask of the image.
        n_proj (int): number of radial projections
        T1 (float): T1 value milli-seconds. Defaults to np.inf.
        TR (float): TR value in milli-seconds. Defaults to 4.5ms.
    """
    _, image_biasfield_smoothed = calculate_biasfield_rf(
        image1, image2, mask, n_proj, T1, TR
    )
    image_cor = np.divide(image, image_biasfield_smoothed)
    image_cor[np.isnan(image_cor)] = 0
    image_cor[np.isinf(image_cor)] = 0
    image_cor[mask == 0] = image[mask == 0]
    image_cor = img_utils.normalize(
        image_cor, mask, method=constants.NormalizationMethods.MAX
    )
    return image_cor, image_biasfield_smoothed


def main(argv):
    """Apply N4ITK bias field correction."""
    try:
        image = nib.load(FLAGS.image_file).get_fdata()
    except:
        raise ValueError("not a valid filename")
    try:
        mask = nib.load(FLAGS.mask_file).get_fdata()
    except:
        raise ValueError("not a valid filename")
    image_cor, biasfield = correct_biasfield_n4itk(image=image, mask=mask)
    image_cor_nii = nib.Nifti1Image(image_cor.astype(float), np.eye(4))

    if FLAGS.output_path:
        output_path = FLAGS.output_path
    else:
        output_path = os.path.dirname(FLAGS.image_file)

    nib.save(image_cor_nii, os.path.join(output_path, "image_cor.nii"))
    biasfield_nii = nib.Nifti1Image(biasfield.astype(float), np.eye(4))
    nib.save(biasfield_nii, os.path.join(output_path, "biasfield.nii"))


if __name__ == "__main__":
    app.run(main)
