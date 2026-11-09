"""Scripts to run gas exchange mapping pipeline in batches."""
import glob
import importlib
import logging
import os
import pdb

from absl import app, flags

from main import gx_mapping_readin, gx_mapping_reconstruction

FLAGS = flags.FLAGS

flags.DEFINE_string("cohort", "healthy", "cohort folder name in config folder")

CONFIG_PATH = "config/"


def main(argv):
    """Run the gas exchange imaging pipeline in multiple subjects.

    Import the config file and run the gas exchange imaging pipeline on all
    subjects specified in by the cohort flag.
    """
    if FLAGS.cohort == "healthy":
        subjects = glob.glob(os.path.join(CONFIG_PATH, "healthy", "*py"))
    elif FLAGS.cohort == "cteph":
        subjects = glob.glob(os.path.join(CONFIG_PATH, "cteph", "*py"))
    elif FLAGS.cohort == "ild":
        subjects = glob.glob(os.path.join(CONFIG_PATH, "ild", "*py"))
    elif FLAGS.cohort == "tyvaso":
        subjects = glob.glob(os.path.join(CONFIG_PATH, "tyvaso", "*py"))
    elif FLAGS.cohort == "jupiter":
        subjects = glob.glob(os.path.join(CONFIG_PATH, "jupiter", "*py"))
    elif FLAGS.cohort == "all":
        subjects = glob.glob(os.path.join(CONFIG_PATH, "healthy", "*py"))
        subjects += glob.glob(os.path.join(CONFIG_PATH, "cteph", "*py"))
        subjects += glob.glob(os.path.join(CONFIG_PATH, "ild", "*py"))
        subjects += glob.glob(os.path.join(CONFIG_PATH, "tyvaso", "*py"))
    elif "-" in FLAGS.cohort:
        cohorts = FLAGS.cohort.split("-")
        subjects = []
        for cohort in cohorts:
            if cohort == "healthy":
                subjects += glob.glob(os.path.join(CONFIG_PATH, "healthy", "*py"))
            elif cohort == "cteph":
                subjects += glob.glob(os.path.join(CONFIG_PATH, "cteph", "*py"))
            elif cohort == "ild":
                subjects += glob.glob(os.path.join(CONFIG_PATH, "ild", "*py"))
            elif cohort == "tyvaso":
                subjects += glob.glob(os.path.join(CONFIG_PATH, "tyvaso", "*py"))
            else:
                raise ValueError("Invalid cohort name")
    else:
        raise ValueError("Invalid cohort name")

    for subject in subjects:
        config_obj = importlib.import_module(
            name=subject[:-3].replace("/", "."), package=None
        )
        config = config_obj.get_config()
        logging.info("Processing subject: %s", config.subject_id)
        if FLAGS.force_recon:
            gx_mapping_reconstruction(config)
        elif FLAGS.force_readin:
            gx_mapping_readin(config)
        elif config.processes.gx_mapping_recon:
            gx_mapping_reconstruction(config)
        elif config.processes.gx_mapping_readin:
            gx_mapping_readin(config)
        else:
            pass


if __name__ == "__main__":
    app.run(main)
