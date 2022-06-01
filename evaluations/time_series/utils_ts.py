import sys

sys.path.append("")

import numpy as np
import pandas as pd
import utils as utils
import csv
import os
from pathlib import Path
import matplotlib.pyplot as plt

from config import Config

NORMAL_WINDOW_LENGTH, ANOMALY_WINDOW_LENGTH = 30, 30

def collect_simulations(sims_path):
    sims = []
    for sim_path in sims_path.iterdir():
        if sim_path.is_dir() and sim_path.name.endswith(
                "-uncertainty-evaluated"
        ):
            sims.append(sim_path.name)
    return sims


def get_frame_ids(np_array):
    print("{}, {}".format("frame_id", "uncertainty"))
    for index, val in np.ndenumerate(np_array):
        print("{}, {}".format(index[0], val))


def driving_log_to_np(csv_file):
    columns_to_read = ["uncertainty"]
    df = pd.read_csv(csv_file, usecols=columns_to_read)
    driving_log_2d = df.to_numpy()

    assert driving_log_2d.ndim == 2
    assert len(driving_log_2d) == len(df)

    return driving_log_2d

def load_config():
    root_dir = utils.get_project_root()
    cfg = Config()
    cfg.from_pyfile(root_dir / "config_my.py")
    return root_dir, cfg

###############################################################################
# CHECKS
###############################################################################

def windows_check(len_uncertainties, len_uncertainties_windows):
    actual_len = len_uncertainties_windows
    expected_len = len_uncertainties / NORMAL_WINDOW_LENGTH
    return int(expected_len) == int(actual_len)