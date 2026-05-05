"""Registration module."""
import logging
from typing import Tuple
import time
import ants # << needs antspyx version 0.6.2
import nibabel as nib
import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("image_static", "", "nii image file path of static image.")
flags.DEFINE_string("image_moving1", "", "nii image file path")
flags.DEFINE_string("image_moving2", "", "nii image file path")


def register_ants(
    image_static: np.ndarray, image_moving1: np.ndarray, image_moving2: np.ndarray, registration_key: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Register images using ANTsPy.

    Args:
        image_static: np.ndarray static image.
        image_moving1: np.ndarray moving image 1.
        image_moving2: np.ndarray moving image 2 using the calculated
            transform between image_static and image_moving1.

    Returns:
        Tuple of registered images (moving1_reg, moving2_reg).
    """

    def convert_to_float64(array: np.ndarray) -> np.ndarray:
        """Converts array to float64, computing magnitude if complex."""
        if np.iscomplexobj(array):
            return np.abs(array).astype(np.float64)  # magnitude of complex
        if array.dtype == bool:
            return array.astype(np.float64)
        return array.astype(np.float64)
    
    def numpy_to_ants(array: np.ndarray) -> ants.ANTsImage:
        """Convert numpy array to ANTsImage with neutral physical space metadata."""
        return ants.from_numpy(
            array.astype(np.float64),
            origin = (0.0, 0.0, 0.0),
            spacing = (1.0, 1.0, 1.0),
            direction = np.eye(3))

    start_time = time.time()
    
    # Ensure float dtype (ANTsPy requires numeric)
    image_static = convert_to_float64(image_static)
    image_moving1 = convert_to_float64(image_moving1)
    image_moving2 = convert_to_float64(image_moving2)

    # Convert numpy arrays --> ANTsImage
    ants_static = numpy_to_ants(np.abs(image_static))
    ants_moving1 = numpy_to_ants(np.abs(image_moving1))
    ants_moving2 = numpy_to_ants(np.abs(image_moving2))

    # Registration of moving1 --> static
    transform = ("TRSAA" if registration_key == "mask2gas" else "Rigid")
    logging.info(f"*** Using ANTsPy ({transform}) to register images ...")
    registration = ants.registration(fixed = ants_static, moving = ants_moving1,
                                     aff_metric = 'mattes', syn_metric = 'mattes',
                                     type_of_transform = transform, interpolator = "bSpline")

    moving1_reg = registration["warpedmovout"].numpy()

    # Apply the same transform to moving2
    moving2_ants_reg = ants.apply_transforms(fixed = ants_static, moving = ants_moving2,
        transformlist = registration["fwdtransforms"], interpolator = "bSpline",)
    moving2_reg = moving2_ants_reg.numpy()
    end_time = time.time()
    tot_time = end_time - start_time
    logging.info(f"Execution time: {tot_time:.2f} secs")
    return moving1_reg.astype("float64"), moving2_reg.astype("float64")


def main(argv):
    """ANTsPy registration command line."""
    register_ants(
        image_static = nib.load(FLAGS.image_static).get_fdata(),
        image_moving1 = nib.load(FLAGS.image_moving1).get_fdata(),
        image_moving2 = nib.load(FLAGS.image_moving2).get_fdata(),
    )


if __name__ == "__main__":
    app.run(main)