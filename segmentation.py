"""Module for segmentation model inference.

@author: ZiyiW Now Sup
"""
import os

import numpy as np
from absl import app, flags
from scipy.ndimage import zoom

from models.model_vnet import vnet
from utils import constants, img_utils, io_utils

# define flags
FLAGS = flags.FLAGS

flags.DEFINE_string("image_type", "vent", "either ute or vent for segmentation")
flags.DEFINE_string("nii_filepath", "", "nii image file path")


def predict(
    image: np.ndarray,
    image_type: str = constants.ImageType.VENT.value,
    erosion: int = 0,
) -> np.ndarray:
    """Generate a segmentation mask from the proton or ventilation image.

    Args:
        image: np.nd array of the input image to be segmented.
        image_type: str of the image type ute or vent.
    Returns:
        mask: np.ndarray of type bool of the output mask.
    """
    # get shape of the image
    img_h, img_w, _ = np.shape(image)
    # reshaping image for segmentation
    if img_h == 64 and img_w == 64:
        print("Reshaping image for segmentation")
        image = zoom(abs(image), [2, 2, 2])
    elif img_h == 128 and img_w == 128:
        pass
    else:
        raise ValueError("Segmentation Image size should be 128 x 128 x n")

    model = vnet(input_size=(128, 128, 128, 1))

    if image_type == constants.ImageType.VENT.value:
        weights_dir_current = "./models/weights/model_ANATOMY_VEN.h5"
    elif image_type == constants.ImageType.UTE.value:
        weights_dir_current = "./models/weights/model_ANATOMY_UTE.h5"
    else:
        raise ValueError("image_type must be ute or vent")

    # Load model weights
    model.load_weights(weights_dir_current)

    # Normalize input to 0-255, then subtract the mean and divide by the standard deviation
    image = img_utils.standardize_image(image)

    # Model Prediction
    image = image[None, ...]
    image = image[..., None]
    mask = model.predict(image)
    # Making mask binary
    mask = mask[0, :, :, :, 0]
    mask[mask > 0.5] = 1
    mask[mask < 1] = 0
    # erode mask
    if erosion > 0:
        mask = img_utils.erode_image(mask, erosion)
    return mask.astype(bool)


def main(argv):
    """Run CNN model inference on ute or vent image."""
    image = io_utils.import_nii(FLAGS.nii_filepath)
    image_type = FLAGS.image_type
    mask = predict(image, image_type)
    export_path = os.path.join(os.path.dirname(FLAGS.nii_filepath), "mask.nii")
    io_utils.export_nii(image=mask.astype("float64"), path=export_path)


if __name__ == "__main__":
    app.run(main)
