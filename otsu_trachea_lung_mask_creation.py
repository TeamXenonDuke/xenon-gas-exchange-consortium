# Hysteresis with connectivity + adaptive LOW_FACTOR based on histogram skew
# plus morphological closing (im.close).

import nibabel as nib
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import remove_small_objects, binary_closing, ball

# default parameters (tweak as needed depedning on how well mass output matches gas.nii)
BASE_LOW_FACTOR = 0.50  # starting point for low threshold scaling
# This is the most important parameter to tweak if results are too noisy or too sparse
LOW_CLIP        = 0.50  # tweak this, don't go below this (too permissive invites noise)

HIGH_CLIP       = 0.7  # don't go above this (too strict might miss signal)
ALPHA           = 0.05  # how strongly skew affects LOW_FACTOR
MIN_SIZE        = 25    # remove tiny specks after connectivity
RADIUS          = 1    # im.close() closing size (increase for stronger smoothing)

def _skew_standardized(x):
    """Return simple standardized skewness (3rd moment); no scipy needed."""
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    z3 = ((x - m) / s) ** 3
    return float(np.mean(z3))

def otsu_hysteresis_segment(input_path, output_path):
    nii  = nib.load(input_path)
    data = nii.get_fdata()

    vox = data[data > 0]
    if vox.size == 0:
        raise ValueError("No nonzero voxels found. Check input file.")

    skew = _skew_standardized(vox)

    t_high = threshold_otsu(vox)

    low_factor = BASE_LOW_FACTOR - ALPHA * skew
    low_factor = float(np.clip(low_factor, LOW_CLIP, HIGH_CLIP))
    t_low = t_high * low_factor

    seeds  = (data >= t_high)
    region = (data >= t_low)

    lab = label(region, connectivity=1)
    seed_ids = np.unique(lab[seeds])
    seed_ids = seed_ids[seed_ids != 0]
    connected = np.isin(lab, seed_ids)

    connected = remove_small_objects(connected, min_size=MIN_SIZE)

    closed = binary_closing(connected, ball(RADIUS))

    mask = closed.astype(np.uint8)
    out = nib.Nifti1Image(mask, affine=nii.affine, header=nii.header)
    nib.save(out, output_path)

    kept = int(mask.sum())
    total = mask.size
    print(f"Skew: {skew:.3f} | Otsu high: {t_high:.6g}")
    print(f"LOW_FACTOR(adapted): {low_factor:.3f} -> Low: {t_low:.6g}")
    print(f"Mask voxels: {kept} / {total} ({100*kept/total:.2f}% kept)")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    SUBJECT = "TY-003"
    input_file = f"data/TY_subjects/{SUBJECT}/gas_highreso.nii"
    output_file = f"data/TY_subjects/{SUBJECT}/{SUBJECT}_mask_trachea_lung_corrected.nii"
    otsu_hysteresis_segment(input_file, output_file)