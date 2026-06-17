"""Sparse grid distance calculation.

An N-Dimmensional convolution based gridding algorithm. Motivated by
code written by Gary glover (http://www-mrsrl.stanford.edu/~brian/gridding/)
and also from code by Nick Zwart.
Background reading:
    1. A fast Sinc Function Gridding Algorithm for Fourier Inversion in
    Computer Tomography. O'Sullivan. 1985.
    2. Selection of a Convolution Function for Fourier Inversion using
    Gridding. Jackson et al. 1991.
    3. Rapid Gridding Reconstruction With a Minimal Oversampling Ratio.
    Beatty et al. 2005.

This code is based off the code written by Scott Robertson.

    Source: https://github.com/ScottHaileRobertson/Non-Cartesian-Reconstruction
"""

import logging
import math
from typing import Tuple

import numpy as np
from numba import njit

DEBUG = False
DEBUG_GRID = False


@njit
def grid_point(
    sample_loc: np.ndarray,
    idx_convert: np.ndarray,
    kernel_halfwidth_sqr: float,
    ndims: int,
    cur_dim: int,
    bounds: np.ndarray,
    seed_pt: np.ndarray,
    kern_dist_sq: float,
    output_dims: np.ndarray,
    sample_index: int,
    n_nonsparse_entries: np.ndarray,
    sparse_sample_indices: np.ndarray,
    sparse_voxel_indices: np.ndarray,
    sparse_distances: np.ndarray,
    force_dim: int,
):
    """Convolve ungridded data with a kernel.

    Recursive function that loops through a bounded section of the output grid,
    convolving the ungriddent point's data according to the convolution kernel,
    density compensation value, and the ungridded point's vlaue. The recursion allows
    for n-dimensional data reconstruction.

    Args:
        sample_loc (np.ndarray): The location of the ungridded point in the output
        idx_convert (np.ndarray): The conversion factor for converting the output
        kernel_halfwidth_sqr (float): The kernel halfwidth squared
        ndims (int): The number of dimensions
        cur_dim (int): The current dimension
        bounds (np.ndarray): The minimum and maximum bounds of the subarray.
        seed_pt (np.ndarray): The seed point
        kern_dist_sq (float): The kernel distance squared
        output_dims (np.ndarray): The output dimensions
        sample_index (int): The sample index
        n_nonsparse_entries (np.ndarray): The number of non-sparse entries
        sparse_sample_indices (np.ndarray): The sparse sample indices
        saprse_voxel_indices (np.ndarray): The sparse voxel indices
        sparse_distances (np.ndarray): The sparse distances
        force_dim (int): The force dimension. If -1, then no dimension is forced.
    """
    lower = int(bounds[2 * cur_dim])
    upper = int(bounds[2 * cur_dim + 1])

    for i in range(lower, upper + 1):
        seed_pt[cur_dim] = i
        if (cur_dim == force_dim) or force_dim == -1:
            new_kern_dist_sq = float(i - sample_loc[cur_dim])
            if DEBUG_GRID:
                logging.info(
                    "\tSaving distance for dim %u = %f (adding %f to %f)"
                    % (
                        cur_dim + 1,
                        new_kern_dist_sq,
                        new_kern_dist_sq * new_kern_dist_sq,
                        kern_dist_sq,
                    )
                )
            new_kern_dist_sq *= new_kern_dist_sq
            new_kern_dist_sq += kern_dist_sq
        else:
            if DEBUG_GRID:
                logging.info(
                    "\t\tWrong dimension dim %u, maintaining distance %f"
                    % (cur_dim + 1, kern_dist_sq)
                )

            new_kern_dist_sq = kern_dist_sq

        if cur_dim > 0:
            if DEBUG_GRID:
                debug_string = "\t Recursing dim {} - gridding [".format(cur_dim + 1)
                for j in range(ndims):
                    if j < cur_dim:
                        debug_string += "%u:%u" % (bounds[2 * j], bounds[2 * j + 1])
                    else:
                        debug_string += "%u" % seed_pt[j]
                    if j < ndims - 1:
                        debug_string += ","
                debug_string += "]"
                logging.info(debug_string)

            grid_point(
                sample_loc,
                idx_convert,
                kernel_halfwidth_sqr,
                ndims,
                cur_dim - 1,
                bounds,
                seed_pt,
                new_kern_dist_sq,
                output_dims,
                sample_index,
                n_nonsparse_entries,
                sparse_sample_indices,
                sparse_voxel_indices,
                sparse_distances,
                force_dim,
            )
        else:
            if new_kern_dist_sq <= kernel_halfwidth_sqr:
                idx_ = 0
                for j in range(ndims):
                    idx_ += seed_pt[j] * idx_convert[j]

                # logging.info(sample_index, n_nonsparse_entries)
                sparse_sample_indices[n_nonsparse_entries[0]] = sample_index + 1
                sparse_voxel_indices[n_nonsparse_entries[0]] = float(idx_ + 1)
                sparse_distances[n_nonsparse_entries[0]] = math.sqrt(new_kern_dist_sq)
                # print(sparse_distances[n_nonsparse_entries[0]], n_nonsparse_entries[0])
                n_nonsparse_entries[0] += 1

            else:
                if DEBUG_GRID:
                    debug_string = "\tVoxel [[{}, {}, {}] is too far ".format(
                        seed_pt[0],
                        seed_pt[1],
                        seed_pt[2],
                    )
                    debug_string += "({} > {}) from sample point ".format(
                        new_kern_dist_sq,
                        kernel_halfwidth_sqr,
                    )
                    debug_string += "[{}, {}, {}] (index {})!".format(
                        sample_loc[0],
                        sample_loc[1],
                        sample_loc[2],
                        sample_index,
                    )
                    logging.info(debug_string)

        seed_pt[cur_dim] = lower


@njit
def sparse_gridding_distance(
    coords: np.ndarray,
    kernel_width: float,
    n_points: int,
    n_dims: int,
    output_dims: np.ndarray,
    n_nonsparse_entries: np.ndarray,
    max_size: int,
    force_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform sparse gridding distance calculation.

    Uses convolution-based gridding. Loops through a set of n-dimensaional sample
    points and convolves them onto a grid.

    Args:
        coords: Array of sample coordinates.
        kernel_width: Kernel width.
        n_points: Number of sample points.
        n_dims: Number of dimensions.
        output_dims: Dimensions of output grid.
        n_nonsparse_entries: Number of non-sparse entries.
        max_size: Maximum size of output arrays.
        force_dim: Force a dimension to be gridded.

    Returns:
        nonsparse_sample_indices: Array of sample indices.
        nonsparse_voxel_indices: Array of voxel indices.
        nonsparse_distances: Array of distances.
    """
    # define constants
    kernel_halfwidth = kernel_width * 0.5
    kernel_halfwidth_sqr = kernel_halfwidth**2
    output_halfwidth = np.zeros(n_dims)

    # initialize output arrays
    nonsparse_sample_indices = np.zeros(max_size)
    nonsparse_voxel_indices = np.zeros(max_size)
    nonsparse_distances = np.zeros(max_size)

    # intialize bounds
    bounds = np.zeros(2 * n_dims)
    seed_pt = np.zeros(n_dims)
    sample_loc = np.zeros(n_dims)

    # calculate output halfwidth
    for dim in range(n_dims):
        output_halfwidth[dim] = int(np.ceil(float(output_dims[dim] * 0.5)))

    # calculate x, y, z to index conversions
    idx_convert = np.zeros(n_dims)
    for dim in range(n_dims):
        idx_convert[dim] = 1
        if dim > 0:
            for p in range(dim):
                idx_convert[dim] = idx_convert[dim] * output_dims[p]

    # loop through sample points
    for p in range(n_points):
        # calculate subarray boundaries
        for dim in range(n_dims):
            sample_loc[dim] = coords[n_dims * p + dim] * float(
                output_dims[dim]
            ) + float(output_halfwidth[dim])
            # calculate bounds
            bounds[2 * dim] = int(max(np.ceil(sample_loc[dim] - kernel_halfwidth), 0))
            bounds[2 * dim + 1] = int(
                min(np.floor(sample_loc[dim] + kernel_halfwidth), output_dims[dim] - 1)
            )
            # intialize recursive seed point as upper left corner of subarray
            seed_pt[dim] = bounds[2 * dim]

        if DEBUG:
            debug_string = "GRIDDING "

            debug_string += "Sample loc = [%f,%f,%f], " % (
                sample_loc[0],
                sample_loc[1],
                sample_loc[2],
            )
            debug_string += "Bounds = [%u:%u,%u:%u,%u:%u]" % (
                bounds[0],
                bounds[1],
                bounds[2],
                bounds[3],
                bounds[4],
                bounds[5],
            )
            logging.info(debug_string)

        grid_point(
            sample_loc,
            idx_convert,
            kernel_halfwidth_sqr,
            n_dims,
            n_dims - 1,
            bounds,
            seed_pt,
            0,
            output_dims,
            p,
            n_nonsparse_entries,
            nonsparse_sample_indices,
            nonsparse_voxel_indices,
            nonsparse_distances,
            force_dim,
        )

    return nonsparse_sample_indices, nonsparse_voxel_indices, nonsparse_distances
