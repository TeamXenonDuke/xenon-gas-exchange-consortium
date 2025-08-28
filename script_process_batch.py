"""Run the gas exchange imaging pipeline across subjects (batch).

Example usage:
 python script_process_batch.py --cohort=FV_healthy_ref
 """

import glob
import importlib
import logging
import os
from pathlib import Path

from absl import app, flags

import main as gx_main
from main import gx_mapping_readin, gx_mapping_reconstruction

FLAGS = flags.FLAGS

flags.adopt_module_key_flags(gx_main)

flags.DEFINE_string(
    "cohort",
    "FV_healthy_ref",
    "Cohort(s): single like 'FV_healthy_ref', hyphen list like 'cteph-ild', or 'all'.",
)

CONFIG_ROOT = "config"

def _subjects_for(cohort: str):
    pat = os.path.join(CONFIG_ROOT, cohort, "*.py")
    return sorted(p for p in glob.glob(pat) if not os.path.basename(p).startswith("_"))

def main(_):
    if FLAGS.cohort == "all":
        cohorts = ["FV_healthy_ref", "cteph", "ild", "tyvaso", "jupiter"]
    elif "-" in FLAGS.cohort:
        cohorts = FLAGS.cohort.split("-")
    else:
        cohorts = [FLAGS.cohort]

    subjects = []
    for c in cohorts:
        if c not in {"FV_healthy_ref", "cteph", "ild", "tyvaso", "jupiter"}:
            raise ValueError(f"Invalid cohort name: {c}")
        subjects += _subjects_for(c)

    if not subjects:
        raise FileNotFoundError(f"No subject configs found for: {cohorts}")

    for cfg_path in subjects:
        module_name = Path(os.path.splitext(cfg_path)[0]).as_posix().replace("/", ".")
        cfg_mod = importlib.import_module(module_name)
        config = cfg_mod.get_config()

        logging.info("Processing subject: %s", getattr(config, "subject_id", cfg_path))

        if getattr(FLAGS, "force_recon", False):
            gx_mapping_reconstruction(config)
        elif getattr(FLAGS, "force_readin", False):
            gx_mapping_readin(config)
        elif getattr(config.processes, "gx_mapping_recon", False):
            gx_mapping_reconstruction(config)
        elif getattr(config.processes, "gx_mapping_readin", False):
            gx_mapping_readin(config)
        else:
            logging.info("No processes enabled for %s; skipping.", cfg_path)

if __name__ == "__main__":
    app.run(main)
