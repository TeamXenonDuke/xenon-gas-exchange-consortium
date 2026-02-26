"""Util functions for config files."""

import numpy as np

from utils import constants


def get_thresholds(recon_key: str) -> np.ndarray:
    if recon_key == constants.ReconKey.PLUMMER.value:
        return np.array([-0.38, 1.73, 4.41, 7.85, 12.31, 18.16, 25.93])
    elif recon_key == constants.ReconKey.ROBERTSON.value:
        return np.array([-2.02, 0.53, 3.66, 7.63, 12.99, 21.07, 35.56])
    else:
        raise ValueError(f"Invalid scan type: {recon_key}")
