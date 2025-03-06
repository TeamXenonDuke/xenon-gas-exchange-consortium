"""Plotting functions for the project."""

import sys
from typing import Dict, List, Optional, Tuple

import skimage
from matplotlib.ticker import ScalarFormatter

sys.path.append("..")
import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import io_utils


def _merge_rgb_and_gray(gray_slice: np.ndarray, rgb_slice: np.ndarray) -> np.ndarray:
    """Combine the gray scale image with the RGB binning via HSV.

    Args:
        gray_slice (np.ndarray): 2D image slice of grayscale image.
        rgb_slice (_type_): 3D image slice of the RGB grayscale image of shape
            (H, W, C)

    Returns:
        (np.ndarray): merged image slice
    """
    # construct RGB version of gray-level ute
    gray_slice_color = np.dstack((gray_slice, gray_slice, gray_slice))
    # Convert the input image and color mask to HSV
    gray_slice_hsv = skimage.color.rgb2hsv(gray_slice_color)
    rgb_slice_hsv = skimage.color.rgb2hsv(rgb_slice)
    # Replace the hue and saturation of the original image
    # with that of the color mask
    gray_slice_hsv[..., 0] = rgb_slice_hsv[..., 0]
    gray_slice_hsv[..., 1] = rgb_slice_hsv[..., 1]
    mask = (
        (rgb_slice[:, :, 0] == 0)
        & (rgb_slice[:, :, 1] == 0)
        & (rgb_slice[:, :, 2] == 0)
    )
    mask = ~mask
    gray_slice_hsv[mask, :] = rgb_slice_hsv[mask, :]
    colormap = skimage.color.hsv2rgb(gray_slice_hsv)
    return colormap


def map_grey_to_rgb(image: np.ndarray, cmap: Dict[int, np.ndarray]) -> np.ndarray:
    """Map a greyscale image to a RGB image using a colormap.

    Args:
        image (np.ndarray): greyscale image of shape (x, y, z)
        cmap (Dict[int, np.ndarray]): colormap mapping integers to RGB values.
    Returns:
        RGB image of shape (x, y, z, 3)
    """
    rgb_image = np.zeros((image.shape[0], image.shape[1], image.shape[2], 3))
    for key in cmap.keys():
        rgb_image[image == key] = cmap[key]
    return rgb_image


def get_biggest_island_indices(arr: np.ndarray) -> Tuple[int, int]:
    """Get the start and stop indices of the biggest island in the array.

    Args:
        arr (np.ndarray): binary array of 0s and 1s.
    Returns:
        Tuple of start and stop indices of the biggest island.
    """
    # intitialize count
    cur_count = 0
    cur_start = 0

    max_count = 0
    pre_state = 0

    index_start = 0
    index_end = 0

    for i in range(0, np.size(arr)):
        if arr[i] == 0:
            cur_count = 0
            if (pre_state == 1) & (cur_start == index_start):
                index_end = i - 1
            pre_state = 0

        else:
            if pre_state == 0:
                cur_start = i
                pre_state = 1
            cur_count += 1
            if cur_count > max_count:
                max_count = cur_count
                index_start = cur_start

    return index_start, index_end


def map_and_overlay_to_rgb(
    image: np.ndarray, image_background: np.ndarray, cmap: Dict[int, np.ndarray]
) -> np.ndarray:
    """Map a greyscale image to a RGB image using a colormap and combine w/ background.

    Args:
        image (np.ndarray): greyscale image of shape (x, y, z)
        image_background (np.ndarray): greyscale image of shape (x, y, z)
        cmap (Dict[int, np.ndarray]): colormap mapping integers to RGB values.
    Returns:
        RGB image of shape (x, y, z, 3)
    """
    image_rgb = map_grey_to_rgb(image, cmap)
    image_out = np.zeros((image.shape[0], image.shape[1], image.shape[2], 3))
    for i in range(0, image.shape[2]):
        image_out[:, :, i, :] = _merge_rgb_and_gray(
            image_background[:, :, i], image_rgb[:, :, i, :]
        )
    return image_out


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay the border of a binary mask on a greyscale image in red.

    Args:
        image (np.ndarray): Greyscale image of shape (x, y, z)
        mask (np.ndarray): Binary mask of shape (x, y, z)

    Returns:
        np.ndarray: Overlaid image of shape (x, y, z, 3)
    """
    # divide by the maximum value to normalize to [0, 1]
    image = image / np.max(image)

    def border_mask(mask: np.ndarray) -> np.ndarray:
        mask_dilated = np.zeros_like(mask)
        for i in range(mask.shape[2]):
            mask_dilated[:, :, i] = cv2.dilate(
                mask[:, :, i].astype(np.uint8), np.ones((3, 3)), iterations=1
            )
        return mask_dilated - mask

    border = border_mask(mask)

    image_out = np.zeros((image.shape[0], image.shape[1], image.shape[2], 3))
    for i in range(image.shape[2]):
        image_slice = np.repeat(image[:, :, i][:, :, np.newaxis], 3, axis=2)
        border_slice = border[:, :, i][:, :, np.newaxis]
        image_slice[border_slice[..., 0] == 1] = [1, 0, 0]
        image_out[:, :, i, :] = image_slice

    return image_out


def get_plot_indices(image: np.ndarray, n_slices: int = 16) -> Tuple[int, int]:
    """Get the indices to plot the image.

    Args:
        image (np.ndarray): binary image.
        n_slices (int, optional): number of slices to plot. Defaults to 16.
    Returns:
        Tuple of start and interval indices.
    """
    sum_line = np.sum(np.sum(image, axis=0), axis=0)
    index_start, index_end = get_biggest_island_indices(sum_line > 300)
    flt_inter = (index_end - index_start) // n_slices
    # threshold to decide interval number
    if np.modf(flt_inter)[0] > 0.4:
        index_skip = np.ceil(flt_inter).astype(int)
    else:
        index_skip = np.floor(flt_inter).astype(int)
    return index_start, index_skip


def make_montage(image: np.ndarray, n_slices: int = 16) -> np.ndarray:
    """Make montage of the image.

    Makes montage of the image.
    Assumes the image is of shape (x, y, z, 3).

    Args:
        image (np.ndarray): image to make montage of.
        n_slices (int, optional): number of slices to plot. Defaults to 16.
    Returns:
        Montaged image array.
    """
    # get the shape of the image
    x, y, z, _ = image.shape
    # get the number of rows and columns
    n_rows = 1 if n_slices < 8 else 2
    n_cols = np.ceil(n_slices / n_rows).astype(int)
    # get the shape of the slices
    slice_shape = (x, y)
    # make the montage array
    montage = np.zeros((n_rows * slice_shape[0], n_cols * slice_shape[1], 3))
    # iterate over the slices
    for slice in range(n_slices):
        # get the row and column
        row = slice // n_cols
        col = slice % n_cols
        # get the slice
        slice = image[:, :, slice, :]
        # add to the montage
        montage[
            row * slice_shape[0] : (row + 1) * slice_shape[0],
            col * slice_shape[1] : (col + 1) * slice_shape[1],
            :,
        ] = slice
    return montage


def plot_montage_grey(
    image: np.ndarray, path: str, index_start: int, index_skip: int = 1
):
    """Plot a montage of the image in grey scale.

    Will make a montage of 2x8 of the image in grey scale and save it to the path.
    Assumes the image is of shape (x, y, z) where there are at least 16 slices.
    Otherwise, will plot all slices.

    Args:
        image (np.ndarray): gray scale image to plot of shape (x, y, z)
        path (str): path to save the image.
        index_start (int): index to start plotting from.
        index_skip (int, optional): indices to skip. Defaults to 1.
    """
    
    # divide by the maximum value
    image = image / np.max(image)
    # stack the image to make it 4D (x, y, z, 3)
    image = np.stack((image, image, image), axis=-1)
    # plot the montage
    index_end = index_start + index_skip * 16
    montage = make_montage(
        image[:, :, index_start:index_end:index_skip, :], n_slices=16
    )
    plt.figure()
    plt.imshow(montage, cmap="gray")
    plt.axis("off")
    plt.savefig(path, transparent=True, bbox_inches="tight", pad_inches=-0.05, dpi=300)
    plt.clf()
    plt.close()


def plot_montage_color(
    image: np.ndarray,
    path: str,
    index_start: int,
    index_skip: int = 1,
    n_slices: int = 16,
):
    """Plot a montage of the image in RGB.

    Will make a montage of default (2x8) of the image in RGB and save it to the path.
    Assumes the image is of shape (x, y, z) where there are at least n_slices.
    Otherwise, will plot all slices.

    Args:
        image (np.ndarray): RGB image to plot of shape (x, y, z, 3).
        path (str): path to save the image.
        index_start (int): index to start plotting from.
        index_skip (int, optional): indices to skip. Defaults to 1.
        n_slices (int, optional): number of slices to plot. Defaults to 16.
    """
    # plot the montage
    index_end = index_start + index_skip * n_slices
    montage = make_montage(
        image[:, :, index_start:index_end:index_skip, :], n_slices=n_slices
    )
    plt.figure()
    plt.imshow(montage, cmap="gray")
    plt.axis("off")
    plt.savefig(path, transparent=True, bbox_inches="tight", pad_inches=-0.05, dpi=300)
    plt.clf()
    plt.close()


def plot_histogram_rbc_osc(
    data: np.ndarray,
    path: str,
    fig_size: Tuple[int, int] = (9, 6),
    xlim: Tuple[float, float] = (-15, 35),
    ylim: Tuple[float, float] = (0, 0.2),
    xticks: List[float] = [-10, 0, 10, 20, 30, 50],
    yticks: List[float] = [0, 0.05, 0.1, 0.15],
    plot_ref: bool = True,
):
    """Plot histogram of RBC oscillation.

    Args:
        data (np.ndarray): data to plot histogram of.
        path (str): path to save the image.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    data = data.flatten()
    weights = np.ones_like(data) / float(len(data))
    # plot histogram
    _, bins, _ = ax.hist(
        data, bins=50, color=(0, 0.8, 0.8), weights=weights, edgecolor="black"
    )
    ax.set_ylabel("Fraction of Voxels", fontsize=35)
    # define and plot healthy reference line
    if plot_ref:
        data_ref = io_utils.import_np(path="data/reference_dist.npy")
        n, bins, _ = ax.hist(
            data_ref,
            bins=bins,
            color=(1, 1, 1),
            alpha=0.0,
            weights=np.ones_like(data_ref) / float(len(data_ref)),
        )
        ax.plot(0.5 * (bins[1:] + bins[:-1]), n, "--", color="k", linewidth=4)
    # set plot parameters
    plt.xlim(xlim)
    plt.ylim(ylim)
    # define ticks
    plt.xticks(xticks, ["{:.0f}".format(x) for x in xticks], fontsize=40)
    plt.yticks(yticks, ["{:.2f}".format(x) for x in yticks], fontsize=40)
    fig.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_histogram(
    data: np.ndarray,
    path: str,
    color: Tuple[float, float, float],
    xlim: float,
    ylim: float,
    num_bins: int,
    refer_fit: Tuple[float, float, float],
    xticks: Optional[List[float]] = None,
    yticks: Optional[List[float]] = None,
    xticklabels: Optional[List[str]] = None,
    yticklabels: Optional[List[str]] = None,
    xlabel: Optional[str] = None,
    title: Optional[str] = None,
):
    """Plot histogram of arbitrary data.

    Args:
        data (np.ndarray): data to plot histogram of.
        path (str): path to save the image.
        color (Tuple[float, float, float]): color of the histogram.
        xlim (float): x limit of the histogram.
        ylim (float): y limit of the histogram.
        num_bins (int): number of bins in the histogram.
        refer_fit (Tuple[float, float, float]): fit parameters of the healthy reference.
        xticks (Optional[List[float]], optional): x ticks. Defaults to None.
        yticks (Optional[List[float]], optional): y ticks. Defaults to None.
        xticklabels (Optional[List[str]], optional): x tick labels. Defaults to None.
        yticklabels (Optional[List[str]], optional): y tick labels. Defaults to None.
        xlabel (Optional[str], optional): x label. Defaults to None.
    """
    # make a thick frame
    plt.rc("axes", linewidth=4)
    fig, ax = plt.subplots(figsize=(9, 6))
    # the histogram of the data
    # limit the range of data
    data = data.flatten()
    data[data < 0] = 0
    data[data > xlim] = xlim
    data = np.append(data, xlim)
    weights = np.ones_like(data) / float(len(data))
    # plot histogram
    _, bins, _ = ax.hist(
        data, num_bins, color=color, weights=weights, edgecolor="black"
    )
    # define and plot healthy reference line
    normal = refer_fit[0] * np.exp(-(((bins - refer_fit[1]) / refer_fit[2]) ** 2))
    ax.plot(bins, normal, "--", color="k", linewidth=4)
    plt.xlim((0, xlim))
    plt.ylim((0, ylim))
    plt.locator_params(axis="x", nbins=4)
    try:
        plt.xticks(xticks, xticklabels, fontsize=35)
        plt.yticks(yticks, yticklabels, fontsize=35)
    except TypeError:
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=30)
    if title is not None:
        ax.set_title(title, fontsize=30)
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_data_rbc_k0(
    t: np.ndarray,
    data: np.ndarray,
    path: str,
    high: np.ndarray = np.array([]),
    low: np.ndarray = np.array([]),
):
    """Plot RBC k0 and binned indices."""
    fig, ax = plt.subplots(figsize=(9, 6))
    # plot healthy reference line
    ax.plot(t, data, "-", color="k", linewidth=5)
    ax.plot(t[high], data[high], ".", color="C2", markersize=10)
    ax.plot(t[low], data[low], ".", color="C1", markersize=10)
    ax.plot(t, np.zeros((len(t), 1)), ".", color="k", linewidth=2)
    ax.set_ylabel("Intensity (au)", fontsize=35)
    # set plot parameters
    plt.rc("axes", linewidth=4)
    plt.xticks([], [])
    plt.yticks(fontsize=40)
    # set ticks
    fig.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_histogram_with_thresholds(
    data: np.ndarray, thresholds: List[float], path: str
):
    """Generate the histogram for the healthy reference distribution.

    Plot histogram of the data with the thresholds each having a different color by
    setting the face color in matplotlib. All values below the first threshold are
    red, all values between the first and second threshold are orange, all values above
    last threshold are purple.

    Args:
        data (np.ndarray): data to plot
        thresholds (List[float]): list of thresholds to plot of length 7.
        path (str): path to save the figure.
    """
    _, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.hist(data, bins=500, density=True)

    # Plot the thresholds
    for threshold in thresholds:
        ax.axvline(threshold, color="k", linestyle="--", linewidth=1)
    # Set the face color for the thresholds
    i = 0
    while ax.patches[i].get_x() < thresholds[0]:
        ax.patches[i].set_facecolor((1, 0, 0))  # red
        i += 1

    while (
        i < len(ax.patches)
        and ax.patches[i].get_x() >= thresholds[0]
        and ax.patches[i].get_x() < thresholds[1]
    ):
        ax.patches[i].set_facecolor((1, 0.7143, 0))
        i += 1
    while (
        i < len(ax.patches)
        and ax.patches[i].get_x() >= thresholds[1]
        and ax.patches[i].get_x() < thresholds[2]
    ):
        ax.patches[i].set_facecolor((0.4, 0.7, 0.4))
        i += 1
    while (
        i < len(ax.patches)
        and ax.patches[i].get_x() >= thresholds[2]
        and ax.patches[i].get_x() < thresholds[3]
    ):
        ax.patches[i].set_facecolor((0, 1, 0))
        i += 1
    while (
        i < len(ax.patches)
        and ax.patches[i].get_x() >= thresholds[3]
        and ax.patches[i].get_x() < thresholds[4]
    ):
        ax.patches[i].set_facecolor((184.0 / 255.0, 226.0 / 255.0, 145.0 / 255.0))
        i += 1
    while (
        i < len(ax.patches)
        and ax.patches[i].get_x() >= thresholds[4]
        and ax.patches[i].get_x() < thresholds[5]
    ):
        ax.patches[i].set_facecolor((243.0 / 255.0, 205.0 / 255.0, 213.0 / 255.0))
        i += 1
    while (
        i < len(ax.patches)
        and ax.patches[i].get_x() >= thresholds[5]
        and ax.patches[i].get_x() < thresholds[6]
    ):
        ax.patches[i].set_facecolor((225.0 / 255.0, 129.0 / 255.0, 162.0 / 255.0))
        i += 1
    while i < len(ax.patches) and ax.patches[i].get_x() >= thresholds[6]:
        ax.patches[i].set_facecolor((197.0 / 255.0, 27.0 / 255.0, 125.0 / 255.0))
        i += 1
    # increase the size of the tick labels
    ax.set_xlabel("RBC Oscillation Amplitude (%)", fontsize=20)
    ax.set_ylabel("Density (a.u.)", fontsize=20)
    ax.set_yticks([])
    ax.tick_params(axis="x", which="major", labelsize=20)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(path)
