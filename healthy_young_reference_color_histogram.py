"""Create healthy histogram figure for bias field correction
3 healthy subjects for Healthy Ref Cohort.

TODO: 
1) DONE: change osc --> vent
2) verify code file 2 (plot.py)
3) verify code file 3 (io_utils.py)
4) verify code file 4 (signal_utils.py)
"""

import logging
import pdb
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags

from utils import signal_utils

sys.path.append(".")
from utils import constants, io_utils, plot_color


def main(unused_argv):
    """Get the thresholds for the healthy reference distribution.

    Load the cumulative distribution from numpy file.
    Apply box-cox transformation to the healthy reference distribution.
    Plot the histogram with the thresholds.
    """
    bias_list = ["rf"]
    for bias_correction in bias_list:
        data_vent = io_utils.import_np(
            path=f"data/FV_healthy_ref/8-28_reference_dist.npy"
            # CHECK THIS
        )

        #Calculate VDPs
        print("data vent:", data_vent)
        VPD_red_data = (np.count_nonzero(data_vent < 0.06449900840620955)) / (len(data_vent))
        print("VDP = " + str(VPD_red_data * 100) + "%")

        scale_factor = 0
        data_trans, lambda_ = signal_utils.boxcox(data=data_vent + scale_factor)

        mean_data_trans = np.mean(data_trans)
        std_data_trans = np.std(data_trans)
        logging.info("Mean: {}".format(np.mean(data_vent)))
        logging.info("Std: {}".format(np.std(data_vent)))
        logging.info("Lambda: %s", lambda_)
        thresholds = []

        for z in range(-2, 3):  # changed from 5 to 3
            threshold = signal_utils.inverse_boxcox(
                lambda_, mean_data_trans + z * std_data_trans, scale_factor
            )
            thresholds.append(threshold)
            logging.info("Box-cox threshold: %s", threshold)

        for z in range(-2, 3):  # changed from 5 to 3
            threshold = np.mean(data_vent) + z * np.std(data_vent)
            logging.info("Gaussian threshold: %s", threshold)

        plot_color.plot_histogram_with_thresholds(
            data_vent,
            thresholds,
            f"data/FV_healthy_ref/8-28_color_reference_dist_rf.png",
            index=bias_list.index(bias_correction)
            # bins=6,  # Set the number of bins to 6
        )

        logging.info("Finished processing bias correction type: %s", bias_correction)


if __name__ == "__main__":
    app.run(main)
