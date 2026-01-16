# utils/mask_include_trachea.py
from __future__ import annotations

import logging
import os
import numpy as np
import nibabel as nib

from utils import trachea_mask


def get_or_make_mask_include_trachea(
    *,
    config,
    base_lung_mask: np.ndarray,
    image_gas_highreso: np.ndarray,
) -> np.ndarray:
    """
    Return mask_include_trachea (lung + trachea).

    Priority:
    1) If user provided an existing filepath -> load it.
    2) Else if auto enabled -> generate trachea mask from gas image and union with base_lung_mask.
    3) Else -> just return base_lung_mask.

    Notes:
    - Writes output into <data_dir>/gx/mask_include_trachea.nii by default.
    """
    base_lung_mask = np.asarray(base_lung_mask).astype(bool)

    user_path = str(getattr(config, "trachea_plus_lung_mask_filepath", "") or "").strip()
    if user_path:
        try:
            if os.path.exists(user_path):
                logging.info(f"Loading mask_include_trachea from: {user_path}")
                loaded = np.squeeze(np.array(nib.load(user_path).get_fdata())).astype(bool)

                if loaded.shape != base_lung_mask.shape:
                    raise ValueError(
                        f"mask_include_trachea shape mismatch: loaded {loaded.shape} vs base {base_lung_mask.shape}"
                    )
                if loaded.sum() == 0:
                    raise ValueError("Loaded mask_include_trachea is empty.")

                return loaded
        except Exception as e:
            logging.warning(
                f"Failed to load provided trachea_plus_lung_mask_filepath='{user_path}'. "
                f"Falling back to auto-generation if enabled. Reason: {e}"
            )

    if not getattr(config, "auto_make_trachea_plus_lung_mask", False):
        logging.info("auto_make_trachea_plus_lung_mask is False; using base lung mask only.")
        return base_lung_mask

    # Internal reference nifti location
    ref_gas_nii_path = "tmp/image_gas_highreso.nii"
    if not os.path.exists(ref_gas_nii_path):
        raise FileNotFoundError(
            f"Expected {ref_gas_nii_path} to exist for auto trachea mask generation + saving."
        )

    logging.info("Auto-generating trachea mask (Otsu+hysteresis) from gas image array.")
    trach_mask = trachea_mask.otsu_hysteresis_mask_from_nifti(image_gas_highreso)
    trach_mask = np.asarray(trach_mask).astype(bool)

    if trach_mask.shape != base_lung_mask.shape:
        raise ValueError(
            f"Auto trachea mask shape mismatch: {trach_mask.shape} vs base {base_lung_mask.shape}"
        )

    combined = np.logical_or(base_lung_mask, trach_mask)

    # Default output dir: <data_dir>/gx
    data_dir = str(getattr(config, "data_dir", "") or "").strip() or "."
    out_dir = os.path.join(data_dir, "gx")
    os.makedirs(out_dir, exist_ok=True)

    # Default output name: mask_include_trachea.nii
    out_path = os.path.join(out_dir, "mask_include_trachea.nii")

    trachea_mask.save_mask_like(ref_gas_nii_path, combined, out_path)
    logging.info(f"Saved mask_include_trachea to: {out_path}")

    # Cache into config
    try:
        config.trachea_plus_lung_mask_filepath = out_path
    except Exception:
        pass

    return combined
