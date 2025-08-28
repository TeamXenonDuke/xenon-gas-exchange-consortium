"""3 healthy REF - Script to get the healthy reference distribution. Create histograms
3 things for modify for each bias correction type
1) This file names when creating new
2) index in subject_classmap.py
3) This file names when plotting

"""
import glob
import importlib
import logging
import os
import pdb

# for histogram plotting
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags

from config import base_config
from subject_classmap import Subject
from utils import img_utils, io_utils, plot, signal_utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "cohort", "FV_healthy_ref", "cohort folder name in config folder"
)  # MODIFY
flags.DEFINE_bool("segmentation", False, "run segmentation again.")
CONFIG_PATH = "config/"


def ventilation_frac_vent_analysis(config: base_config.Config) -> np.ndarray:
    """Get the normalized image of subject image gas and subject mask reg.
    Args:
        config (config_dict.ConfigDict): config dict
    Returns:
        subject.image_gas_cor, subject.mask_reg: normalized images
    """
    subject = Subject(config)
    subject.read_mat_file()  # Check and Modify index

    '''mat_file_index (int, optional): index of the mat file to read in.
                Defaults to 0. 
                index 0 - n4itk, 
                index 1 - rf, 
                index 2 skip, 
                index 3 - template.'''
    if FLAGS.segmentation:
        subject.segmentation()

    image_gas_cor = subject.image_gas_cor  # Shape (128,128,128)
    big_mask = subject.big_mask  # Possible wrong shape
    
    # return the ventilation image that is normalized between 0 and 1, masked
    # this needs to be normalized by the 99th percentile
    # TODO: do not rescale by 99th percentile with fractional ventilation
    # Debugging print
    print(f"Before reshape: image shape = {image_gas_cor.shape}, mask shape = {big_mask.shape}")
    
    return img_utils.normalize(subject.image_gas_cor, subject.big_mask, bag_volume=config.bag_volume), subject.mask


def compile_distribution():
    """Generate the reference distribution for healthy subjects.
    
        Import the config file and load in the mat file for all
        subjects specified in by the cohort flag.
    """
    if FLAGS.cohort == "FV_healthy_ref":  # MODIFY NAME
        subjects = glob.glob(os.path.join(CONFIG_PATH, "FV_healthy_ref", "*py"))  # MODIFY NAME
    elif FLAGS.cohort == "cteph":
        subjects = glob.glob(os.path.join(CONFIG_PATH, "cteph", "*py"))
    elif FLAGS.cohort == "all":
        subjects = glob.glob(os.path.join(CONFIG_PATH, "FV_healthy_ref", "*py"))  # MODIFY NAME
        subjects += glob.glob(os.path.join(CONFIG_PATH, "cteph", "*py"))
    else:
        raise ValueError("Invalid cohort name")

    data_vent = np.array([])
    for subject in subjects:

        config_obj = importlib.import_module(
            name=subject[:-3].replace("/", "."), package=None
        )
        config = config_obj.get_config()
        logging.info("Processing subject: %s", config.subject_id)
        #breakpoint()

        image_ventilation, mask = ventilation_frac_vent_analysis(config)
        data_vent = np.append(data_vent, image_ventilation[mask > 0])
        io_utils.export_nii(np.float32(mask), "tmp/mask.nii")
        io_utils.export_nii(image_ventilation, "tmp/ventilation.nii")
        plt.hist(image_ventilation[mask > 0], bins=50)
        plt.savefig("data/FV_healthy_ref/{}/8-28_blue_histogram.png".format(config.subject_id))
        plt.show()
        testing_subject_mean = np.mean(data_vent)
        logging.info("mean: {}".format(testing_subject_mean))

    io_utils.export_np(
        data_vent, "data/FV_healthy_ref/8-28_reference_dist.npy"
    )  # MODIFY THIS!!!!



def get_thresholds():  # need box plots
    """Get the thresholds for the healthy reference distribution.
    
        Apply box-cox transformation to the healthy reference distribution.
    """

    data_vent = io_utils.import_np(
        path="data/FV_healthy_ref/8-28_reference_dist.npy"
    )  # MODIFY THIS!!!!
    scale_factor = 0
    assert data_vent is not None, "Error: No data_osc found."
    # Check if data_osc and scale_factor have enough values
    data_trans, lambda_ = signal_utils.boxcox(data=data_vent + scale_factor)
    mean_data_trans = np.mean(data_trans)
    std_data_trans = np.std(data_trans)
    logging.info("mean: {}".format(np.mean(data_vent)))
    logging.info("std: {}".format(np.std(data_vent)))
    plot.plot_histogram_ventilation(
        data_vent,
        "data/FV_healthy_ref/8-28_reference_dist.png"
        # MODIFY THIS TITLE!!!!
        # REMEBER TO CHANGE INDEX WITH EACH HISTOGRAM
    )

    # TODO: create function that matches intended purpose and does do purpose -
    logging.info("Lambda: %s", lambda_)
    for z in range(-2, 3):
        threshold = signal_utils.inverse_boxcox(
            lambda_, mean_data_trans + z * std_data_trans, scale_factor
        )
        logging.info("Box-cox threshold: %s", threshold)
    for z in range(-2, 3):
        threshold = np.mean(data_vent) + z * np.std(data_vent)
        logging.info("Gaussian threshold: %s", threshold)


def main(argv):
    """Run the main function.
    Compile the healthy reference distribution and get the thresholds.
    """
    compile_distribution()
    get_thresholds()
    # DONE-TODO: implement plot_histogram
    # plot_histogram()
    # DONE-TODO: plot reference distribution for no bias field correction, N4ITK, and
    # rf-depolarizationm and template --> 4 plots total


if __name__ == "__main__":
    """Run the main function."""
    app.run(main)


# NOTES
# what return, run
# go to declaration -->
# plot historgrams from code