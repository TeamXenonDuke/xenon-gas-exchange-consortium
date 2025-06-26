"""Metrics for evaluating images."""
import math
import sys
from datetime import datetime

sys.path.append("..")
import numpy as np
from scipy.ndimage.morphology import binary_dilation

from utils import constants

import pandas as pd
import logging
import numpy as np

def _get_dilation_kernel(x: int) -> int:
    """Get dilation kernel for binary dilation in 1-dimension."""
    return int((math.ceil(x * 0.025) * 2 + 1))


def snr(image: np.ndarray, mask: np.ndarray, window_size: int = 8):
    """Calculate SNR using sliding windows.

    Args:
        image (np.ndarray): 3-D array of image data.
        mask (np.ndarray): 3-D array of mask data.
        window_size (int): size of the sliding window for noise calculation.
            Defaults to 8.
    Returns:
        Tuple of SNR and Rayleigh SNR and image noise
    """
    shape = np.shape(image)
    # dilate the mask to analyze noise area away from the signal
    kernel_shape = (
        _get_dilation_kernel(shape[0]),
        _get_dilation_kernel(shape[1]),
        _get_dilation_kernel(shape[2]),
    )
    dilate_struct = np.ones((kernel_shape))
    noise_mask = binary_dilation(mask, dilate_struct).astype(bool)

    noise_temp = np.copy(image)
    noise_temp[noise_mask] = np.nan
    # set up for using mini noise cubes through the image and calculate std for noise
    n_noise_vox = window_size * window_size * window_size
    mini_vox_std = 0.75 * n_noise_vox  # minimul number of voxels to calculate std

    stepper = 0
    total = 0
    std_dev_mini_noise_vol = []

    for ii in range(0, int(shape[0] / window_size)):
        for jj in range(0, int(shape[1] / window_size)):
            for kk in range(0, int(shape[2] / window_size)):
                mini_cube_noise_dist = noise_temp[
                    ii * window_size : (ii + 1) * window_size,
                    jj * window_size : (jj + 1) * window_size,
                    kk * window_size : (kk + 1) * window_size,
                ]
                mini_cube_noise_dist = mini_cube_noise_dist[
                    ~np.isnan(mini_cube_noise_dist)
                ]
                # only calculate std for the noise when it is long enough
                if len(mini_cube_noise_dist) > mini_vox_std:
                    std_dev_mini_noise_vol.append(np.std(mini_cube_noise_dist, ddof=1))
                    stepper = stepper + 1
                total = total + 1

    image_noise = np.median(std_dev_mini_noise_vol)
    image_signal = np.average(image[mask])

    SNR = image_signal / image_noise
    return SNR, SNR * 0.66, image_noise


def inflation_volume(mask: np.ndarray, fov: float) -> float:
    """Calculate the inflation volume of isotropic 3D image.

    Args:
        mask: np.ndarray thoracic cavity mask.
        fov: float field of view in cm
    Returns:
        Inflation volume in L.
    """
    return (
        np.sum(mask) * fov**3 / np.shape(mask)[0] ** 3
    ) / constants.FOVINFLATIONSCALE3D


def GLI_volume(age: float, sex: str, height: float, volume_type: str = "frc") -> float:
    """
    Calculate the GLI-predicted lung volume for a given age, sex, and height.

    Args:
        age: float, subject age in years.
        sex: str, subject sex ("M" or "F").
        height: float, subject height in cm.
        volume_type: str, either "frc" (functional residual capacity) or "fvc" (forced vital capacity).

    Returns:
        Predicted lung volume (float) based on GLI lookup table. Returns np.nan if input is missing or match not found.
    """
    if pd.isna(age) or pd.isna(sex) or pd.isna(height):
        return np.nan
    lookup_df = pd.read_pickle('./assets/lut/GLI.pkl')

    # Ensure sex is upper case, and volume_type is lower case
    sex = sex.upper()
    volume_type = volume_type.lower()

    if volume_type == "frc":
        column_name = 'frc_predicted'
    elif volume_type == "fvc":
        column_name = 'fvc_predicted'
    else:
        raise ValueError("volume_type must be either 'frc' or 'fvc'")

    # Helper to get predicted value at specific age and height
    def get_predicted(a, h):
        row = lookup_df[(lookup_df['age'] == a) & (lookup_df['height'] == h) & (lookup_df['sex'] == sex)]
        if not row.empty:
            return row[column_name].values[0]
        else:
            return None

    age_int = int(age) == age
    height_int = int(height) == height

    if age_int and height_int:
        predicted_value = get_predicted(int(age), int(height))
        return predicted_value if predicted_value is not None else 0.0

    elif age_int or height_int:
        if age_int:
            h0 = int(np.floor(height))
            h1 = h0 + 1
            val0 = get_predicted(int(age), h0)
            val1 = get_predicted(int(age), h1)
            if val0 is not None and val1 is not None:
                predicted_value = val0 + (height - h0) * (val1 - val0)
                return predicted_value
        else:
            a0 = int(np.floor(age))
            a1 = a0 + 1
            val0 = get_predicted(a0, int(height))
            val1 = get_predicted(a1, int(height))
            if val0 is not None and val1 is not None:
                predicted_value = val0 + (age - a0) * (val1 - val0)
                return predicted_value

    else:
        a0 = int(np.floor(age))
        a1 = a0 + 1
        h0 = int(np.floor(height))
        h1 = h0 + 1
        val00 = get_predicted(a0, h0)
        val01 = get_predicted(a0, h1)
        val10 = get_predicted(a1, h0)
        val11 = get_predicted(a1, h1)

        if None not in [val00, val01, val10, val11]:
            wa1 = age - a0
            wa0 = 1 - wa1
            wh1 = height - h0
            wh0 = 1 - wh1

            predicted_value = (
                val00 * wa0 * wh0 +
                val01 * wa0 * wh1 +
                val10 * wa1 * wh0 +
                val11 * wa1 * wh1
            )
            return predicted_value

    return np.nan

def get_bag_volume(fvc_volume: float) -> float:
    """
    Given FVC volume, calculate the bag volume as 20% of FVC,
    rounded to the nearest 0.25 increment.
    
    Args:
        fvc_volume (float): The FVC volume in liters.

    Returns:
        float: Bag volume rounded to the nearest 0.25 increment.
    """
    bag_volume = 0.2 * fvc_volume
    # Round to the nearest 0.25
    bag_volume_rounded = round(bag_volume * 4) / 4.0
    return bag_volume_rounded

def process_date() -> str:
    """Return the current date in YYYY-MM-DD format."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d")


def bin_percentage(image: np.ndarray, bins: np.ndarray, mask: np.ndarray) -> float:
    """Get the percentage of voxels in the given bins.

    Args:
        image: np.ndarray binned image. Assumes that the values in the image are
            integers representing the bin number. Bin 0 is the region outside the mask
            and Bin 1 is the lowest bin, etc.
        bins: np.ndarray list of bins to include in the percentage calculation.
        mask: np.ndarray mask of the region of interest.
    Returns:
        Percentage of voxels in the given bins.
    """
    return 100 * np.sum(np.isin(image, bins)) / np.sum(mask > 0)


def mean(image: np.ndarray, mask: np.ndarray) -> float:
    """Get the mean of the image.

    Args:
        image: np.ndarray. The image.
        mask: np.ndarray. mask of the region of interest.()
    Returns:
        Mean of the image.
    """
    return np.mean(image[mask])


def negative_percentage(image: np.ndarray, mask: np.ndarray) -> float:
    """Get the percentage voxels of image inside mask that are negative.

    Args:
        image: np.ndarray. The image.
        mask: np.ndarray. mask of the region of interest.
    Returns:
        Percentage of voxels in the image that are negative.
    """
    return 100 * np.sum(image[mask] < 0) / np.sum(mask)


def median(image: np.ndarray, mask: np.ndarray) -> float:
    """Get the median of the image.

    Args:
        image: np.ndarray. The image.
        mask: np.ndarray. mask of the region of interest.
    Returns:
        Median of the image.
    """
    return np.median(image[mask])


def std(image: np.ndarray, mask: np.ndarray) -> float:
    """Get the standard deviation of the image.

    Args:
        image: np.ndarray. The image.
        mask: np.ndarray. mask of the region of interest.
    Returns:
        Standard deviation of the image.
    """
    return np.std(image[mask])


def dlco(
    image_gas: np.ndarray,
    image_membrane: np.ndarray,
    image_rbc: np.ndarray,
    mask: np.ndarray,
    mask_vent: np.ndarray,
    fov: float,
    membrane_mean: float = 0.736,
    rbc_mean: float = 0.471,
) -> float:
    """Get the DLCO of the image.

    Reference: https://journals.physiology.org/doi/epdf/10.1152/japplphysiol.00702.2020
    Args:
        image_gas: np.ndarray. The ventilation image.
        image_membrane: np.ndarray. The membrane image.
        img_rbc: np.ndarray. The RBC image.
        mask: np.ndarray. thoracic cavity mask.
        mask_vent: np.ndarray. mask of the non-VDP region.
        fov: float. field of view in cm.
        membrane_mean: float. The mean membrane in healthy subjects.
        rbc_mean: float. The mean RBC in healthy subjects.
    """
    return kco(
        image_membrane, image_rbc, mask_vent, membrane_mean, rbc_mean
    ) * alveolar_volume(image_gas, mask, fov)


def alveolar_volume(image: np.ndarray, mask: np.ndarray, fov: float) -> float:
    """Get the alveolar volume of the image.

    Reference: https://journals.physiology.org/doi/epdf/10.1152/japplphysiol.00702.2020
    Args:
        image: np.ndarray. The binned ventilation image.
        mask: np.ndarray. thoracic cavity mask.
        fov: float. field of view in cm.
    Returns:
        Alveolar volume in L.
    """
    return (
        constants.VA_ALPHA
        * inflation_volume(mask, fov)
        * (1.0 - bin_percentage(image, np.asarray([1]), mask) / 100)
    )


def kco(
    image_membrane: np.ndarray,
    image_rbc: np.ndarray,
    mask: np.ndarray,
    membrane_mean: float = 0.736,
    rbc_mean: float = 0.471,
) -> float:
    """Get the KCO of the image.

    Reference: https://journals.physiology.org/doi/epdf/10.1152/japplphysiol.00702.2020
    Args:
        image_membrane: np.ndarray. The membrane image.
        img_rbc: np.ndarray. The RBC image.
        mask: np.ndarray. mask of non-VDP region.
        membrane_mean: float. The mean membrane in healthy subjects.
        rbc_mean: float. The mean RBC in healthy subjects.
    """
    membrane_rel = mean(image_membrane, mask) / membrane_mean
    rbc_rel = mean(image_rbc, mask) / rbc_mean
    membrane_rel = 1.0 / membrane_rel if membrane_rel > 1 else membrane_rel
    return 1 / (
        1 / (constants.KCO_ALPHA * membrane_rel) + 1 / (constants.KCO_BETA * rbc_rel)
    )
