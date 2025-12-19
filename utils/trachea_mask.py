import os
import nibabel as nib
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import remove_small_objects, binary_closing, ball
from typing import Optional


def _skew_standardized(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    z3 = ((x - m) / s) ** 3
    return float(np.mean(z3))


def otsu_hysteresis_mask_from_nifti(
    input_path: str,
    params: Optional[dict] = None
) -> np.ndarray:
    """Return boolean mask (same shape) from a NIfTI image on disk."""

    if params is None:
        params = {}

    base_low = float(params.get("BASE_LOW_FACTOR", 0.50))
    low_clip = float(params.get("LOW_CLIP", 0.50))
    high_clip = float(params.get("HIGH_CLIP", 0.70))
    alpha = float(params.get("ALPHA", 0.05))
    min_size = int(params.get("MIN_SIZE", 25))
    radius = int(params.get("RADIUS", 1))

    nii = nib.load(input_path)
    data = nii.get_fdata()

    vox = data[data > 0]
    if vox.size == 0:
        raise ValueError("No nonzero voxels found. Check input file.")

    skew = _skew_standardized(vox)
    t_high = threshold_otsu(vox)

    base_low = float(params.get("BASE_LOW_FACTOR", 0.50))
    low_clip = float(params.get("LOW_CLIP", 0.50))
    high_clip = float(params.get("HIGH_CLIP", 0.70))
    alpha = float(params.get("ALPHA", 0.05))
    min_size = int(params.get("MIN_SIZE", 25))
    radius = int(params.get("RADIUS", 1))

    low_factor = base_low - alpha * skew
    low_factor = float(np.clip(low_factor, low_clip, high_clip))
    t_low = t_high * low_factor

    seeds = (data >= t_high)
    region = (data >= t_low)

    lab = label(region, connectivity=1)
    seed_ids = np.unique(lab[seeds])
    seed_ids = seed_ids[seed_ids != 0]
    connected = np.isin(lab, seed_ids)

    connected = remove_small_objects(connected, min_size=min_size)
    closed = binary_closing(connected, ball(radius))

    return closed.astype(bool)


def save_mask_like(input_path: str, mask_bool: np.ndarray, output_path: str) -> None:
    """Save mask as uint8 NIfTI using affine/header from input_path."""
    nii = nib.load(input_path)
    out = nib.Nifti1Image(mask_bool.astype(np.uint8), affine=nii.affine, header=nii.header)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(out, output_path)


def union_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    """Boolean OR union; returns bool."""
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Mask shape mismatch: {mask_a.shape} vs {mask_b.shape}")
    return np.logical_or(mask_a.astype(bool), mask_b.astype(bool))
