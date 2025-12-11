"""
make_healthy_ref_hist_profile.py

Create a healthy-reference histogram profile from the cumulative
ventilation distribution saved in:

    data/FV_healthy_ref/8-28_reference_dist.npy

Output:

    data/FV_healthy_ref/8-28_reference_hist_profile.npy

Shape: (2, NBINS)
    row 0: bin centers
    row 1: probabilities for each bin (sum to 1)

This version:
- Uses counts/total (sum of probs = 1), like your gas/RBC/Mem code.
- Applies a small smoothing kernel and renormalizes, so the peak is not
  crazy tall compared to your other histograms.
"""

import numpy as np
from utils import io_utils


def save_healthy_ref_hist_profile(
    in_path="data/FV_healthy_ref/8-28_reference_dist.npy",
    out_path="data/FV_healthy_ref/8-28_reference_hist_profile.npy",
    nbins=50,
    xlim=1.0,
    smooth=True,
):
    """
    Load the cumulative healthy-reference distribution (1D array of voxel values),
    compute a histogram, (optionally) smooth, and save a (2, nbins) array:

        row 0: bin centers
        row 1: probabilities per bin (sum = 1)

    Args:
        in_path:   path to the 1D numpy file with all healthy voxel values.
        out_path:  path to save the (2, nbins) histogram profile.
        nbins:     number of histogram bins (50 for vent).
        xlim:      upper bound for FV; we mirror your VENT["XLIM"] = 1.0.
        smooth:    if True, apply a tiny smoothing kernel and renormalize.
    """
    # ----- Load data -----
    data_vent = io_utils.import_np(path=in_path)
    if data_vent is None or data_vent.size == 0:
        raise ValueError(f"Error: no data found in {in_path}")

    data_vent = np.asarray(data_vent, dtype=float)

    # STRICT filter to [0, xlim] like your gas/RBC/Mem code
    d = data_vent[(data_vent >= 0.0) & (data_vent <= xlim)]
    if d.size == 0:
        raise ValueError("No data in [0, xlim]. Check xlim or the input array.")

    # ----- Compute histogram (COUNTS, not density) -----
    # Edges like MATLAB linspace(0, xlim, NBINS+1)
    edges = np.linspace(0.0, xlim, nbins + 1)

    counts, _ = np.histogram(d, bins=edges)

    total = counts.sum()
    if total == 0:
        raise ValueError("Histogram has zero total counts; check data and xlim.")

    # Raw probabilities per bin (sum = 1)
    probs = counts.astype(float) / float(total)

    # Optional small smoothing to knock down a huge single-bin spike
    if smooth:
        # Simple [1, 2, 1] smoothing kernel
        kernel = np.array([1.0, 2.0, 1.0], dtype=float)
        kernel /= kernel.sum()
        probs_smooth = np.convolve(probs, kernel, mode="same")
        # Renormalize to keep sum = 1
        probs_smooth /= probs_smooth.sum()
        probs = probs_smooth

    # Bin centers
    centers = 0.5 * (edges[:-1] + edges[1:])

    if centers.shape[0] != nbins or probs.shape[0] != nbins:
        raise RuntimeError(
            f"Unexpected histogram shape: centers={centers.shape}, probs={probs.shape}"
        )

    # Stack into (2, nbins)
    hist_profile = np.vstack([centers, probs])

    # ----- Sanity checks -----
    print("hist_profile.shape =", hist_profile.shape)
    print("Sum of probabilities =", probs.sum())
    print("Max probability =", probs.max())

    # ----- Save -----
    io_utils.export_np(hist_profile, out_path)
    print(f"Saved healthy reference hist profile to: {out_path}")


if __name__ == "__main__":
    save_healthy_ref_hist_profile()
