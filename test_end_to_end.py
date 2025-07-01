""" End-to-end test file. Tests '009-028B_test' """

import pytest
import subprocess
import os
import pandas as pd
from datetime import date
from git import Repo

""" Static dataframe

Models the expected csv for 009-028B_test.py
This will be compared to the generated csv.
"""
expected = pd.DataFrame([{
    "subject_id": "009-028B",
    "scan_date": "2025-06-02",
    "process_date": str(date.today()),
    "scan_type": "medium",
    "pipeline_version": 4, #FIX
    "software_version": "syngo MR XA30", #FIX
    "git_branch": Repo(".").active_branch.name,
    "reference_data_key": "REFERENCE_208_PPM",
    "bandwidth": 781.2500,
    "sample_time": 10.0,
    "fa_dis": 15.0,
    "fa_gas": 0.5,
    "field_strength": 2.8936,
    "flip_angle_factor": 29.66,
    "fov": 40.0,
    "xe_dissolved_offset_frequency": 208,
    "grad_delay_x": -5,
    "grad_delay_y": -5,
    "grad_delay_z": -5,
    "hb_correction_key": "none",
    "hb": 0.0,
    "rbc_hb_correction_factor": 1.0,
    "membrane_hb_correction_factor": 1.0,
    "kernel_sharpness": 0.32,
    "n_skip_start": 60,
    "n_dis_removed": 26,
    "n_gas_removed": 10,
    "remove_noise": True,
    "shape_fids": "(2487, 64)",
    "shape_image": "(128, 128, 128)",
    "t2_correction_factor_membrane": 1.4839,
    "t2_correction_factor_rbc": 1.5395,
    "te90": 500.0,
    "tr_dis": 8.4,
    "user_lung_volume_value": "NA/ NA",
    "inflation": 2.2,
    "rbc_m_ratio": 0.1553,
    "vent_snr": 31.0332,
    "vent_defect_pct": 8.5092,
    "vent_low_pct": 38.8379,
    "vent_high_pct": 4.0063,
    "vent_mean": 0.5820,
    "vent_median": 0.5819,
    "vent_stddev": 0.1409,
    "rbc_snr": 4.2971,
    "rbc_defect_pct": 27.6886,
    "rbc_low_pct": 32.9060,
    "rbc_high_pct": 0.4395,
    "rbc_mean": 0.0021,
    "rbc_median": 0.0018,
    "rbc_stddev": 0.0018,
    "membrane_snr": 31.7665,
    "membrane_defect_pct": 0.0,
    "membrane_low_pct": 0.0993,
    "membrane_high_pct": 1.8453,
    "membrane_mean": 0.0110,
    "membrane_median": 0.0108,
    "membrane_stddev": 0.0021,
    "alveolar_volume": 2.9340,
    "kco_est": 3.6786,
    "dlco_est": 10.7930
}])

@pytest.fixture(scope="session")
def output():
    """ Setup function
    
    Ran once when the test file is called.
    Runs the program with '009-028B_test.py' and generates the actual csv.
    The csv will be converted to a dataframe to be compared to the expected csv.
    """

    config_path = os.path.join("config", "tests", "009-028B_test.py")
    subprocess.run(["python", "main.py", "--config", config_path])

    csv_path = os.path.join("config", "tests", "009-028B_test", "gx", "009-028B_stats.csv")
    return pd.read_csv(csv_path)


@pytest.mark.parametrize("header", expected.columns.tolist())
def test_full_csv(output, header):
    """
    Each generated value in the output csv will be compared to the expected values in the expected csv.
    Any discrepancy will be reported to the terminal.
    This function is parametrized so that it will run for each variable (column header) in the csv.

    Args:
        output (DataFrame): The generated output csv as a dataframe.
        header (str): The variable names (column headers) in the csv.
    """

    output_val = output.loc[0, header]
    expected_val = expected.loc[0, header]
    if isinstance(output_val, float):
        assert round(output_val, 4) == round(expected_val, 4), f"Expected value '{expected_val}', but program generated value '{round(output_val, 4)}' (rounded to 4 decimal spaces) for variable '{header}'"
    else:
        assert output_val == expected_val, f"Expected value '{expected_val}', but program generated value '{output_val}' for variable '{header}'"

