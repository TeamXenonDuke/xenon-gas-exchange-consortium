"""Registration module."""
import logging
import os
from typing import Tuple

import nibabel as nib
import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("image_static", "", "nii image file path of static image.")
flags.DEFINE_string("image_moving1", "", "nii image file path")
flags.DEFINE_string("image_moving2", "", "nii image file path")


def register_ants(
    image_static: np.ndarray, image_moving1: np.ndarray, image_moving2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Register images using Ants executables.

    Args:
        image_static: np.ndarray static image.
        image_moving1: np.ndarray moving image 1.
        image_moving2: np.ndarray moving image 2 using the calculated
            transform between image_static and image_moving1.

    Returns:
        Tuple of registered images
    """
    current_path = os.path.dirname(__file__)
    tmp_path = os.path.join(current_path, "tmp")
    bin_path = os.path.join(current_path, "bin")
    pathInputstatic = os.path.join(tmp_path, "image_static.nii")
    pathInputmoving2 = os.path.join(tmp_path, "image_moving2.nii")
    pathInputmoving1 = os.path.join(tmp_path, "image_moving1.nii")
    pathOutputprefix = os.path.join(tmp_path, "thisTransform_")
    pathOutputmoving2 = os.path.join(tmp_path, "transform_reg.nii.gz")
    pathOutputmoving1 = os.path.join(tmp_path, "moving_reg.nii.gz")

    pathReg = bin_path + "/antsRegistration"
    pathApply = bin_path + "/antsApplyTransforms"

    def convert_bool_to_float(array):
        """
        Converts a boolean array to a float32 array.

        Parameters:
            array (numpy.ndarray): Input array, can be of any dtype.

        Returns:
            numpy.ndarray: Converted array with dtype float32 if the input is boolean;
                           otherwise, the original array.
        """
        if array.dtype == bool:
            return array.astype(np.float32)
        return array

    # Converted the inputs with dtype float32 if the inputs are boolean
    image_static = convert_bool_to_float(image_static);
    image_moving1 = convert_bool_to_float(image_moving1);
    image_moving2 = convert_bool_to_float(image_moving2);

    # save the inputs into nii files so the execute N4 can read in
    nii_static = nib.Nifti1Image(abs(image_static), np.eye(4))
    nii_moving2 = nib.Nifti1Image(abs(image_moving2), np.eye(4))
    nii_moving1 = nib.Nifti1Image(abs(image_moving1), np.eye(4))
    nib.save(nii_static, pathInputstatic)
    nib.save(nii_moving2, pathInputmoving2)
    nib.save(nii_moving1, pathInputmoving1)

    # Rigid transformation
    logging.info("*** Using Ants Executable files to register images ...")
    output_prefix = "[" + pathOutputprefix + ", " + pathOutputmoving1 + "]"
    # command string
    cmd_register = (
        pathReg
        + " --dimensionality 3 \
        --float 0 \
        --interpolation BSpline \
        --metric MI["
        + pathInputstatic
        + ","
        + pathInputmoving1
        + ",1,32,Regular, 1] \
        --transform Rigid[0.1] \
        --convergence [20x20x20, 1e-6, 20] \
        --shrink-factors 4x2x1 \
        --smoothing-sigmas 0x0x0 \
        --output "
        + output_prefix
        + " \
        --verbose 1"
    )
    # registration command
    os.system(cmd_register)
    tdata = os.path.join(tmp_path, "thisTransform_0GenericAffine.mat")
    # command string
    cmd_applyTransform = (
        pathApply
        + " -d 3 -e 0 -i "
        + pathInputmoving2
        + " -r "
        + pathOutputmoving1
        + " -o "
        + pathOutputmoving2
        + " -t "
        + tdata
    )
    # call the command to apply transformation to image_transform
    os.system(cmd_applyTransform)
    try:
        moving2_reg = np.around(np.array(nib.load(pathOutputmoving2).get_fdata()))
    except FileNotFoundError:
        raise Exception(
            "registration failed, could not find antsRegistration executable"
        )
    moving1_reg = np.array(nib.load(pathOutputmoving1).get_fdata())
    # remove the generated nii files
    os.remove(pathInputstatic)
    os.remove(pathInputmoving1)
    os.remove(pathInputmoving2)
    os.remove(pathOutputprefix + "0GenericAffine.mat")
    os.remove(pathOutputmoving2)
    os.remove(pathOutputmoving1)

    return moving1_reg.astype("float64"), moving2_reg.astype("float64")


def main(argv):
    """Ants registration command line."""
    image_static_filepath = FLAGS.image_static
    image_moving1_filepath = FLAGS.image_moving1
    image_moving2_filepath = FLAGS.image_moving2

    register_ants(
        image_static=nib.load(image_static_filepath).get_fdata(),
        image_moving1=nib.load(image_moving1_filepath).get_fdata(),
        image_moving2=nib.load(image_moving2_filepath).get_fdata(),
    )


if __name__ == "__main__":
    app.run(main)
