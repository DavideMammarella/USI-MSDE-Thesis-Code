# !!! Simulations must be in the project folder under: simulations/<name_of_simulation>
# !!! <name_of_simulation> folder must contain: IMG folder (with .jpg) and driving_log.csv

# This script must be used only with UWIZ models

import sys

sys.path.append("..")

import base64
import csv
import logging

# Standard library import ----------------------------------------------------------------------------------------------
import os
import pathlib
from io import BytesIO
from pathlib import Path

import numpy as np
import utils_logging
from PIL import Image

from evaluations.utils_threshold import calc_and_store_thresholds

logger = logging.Logger("utils_thresholds")
utils_logging.log_info(logger)

import utils

# Local libraries import -----------------------------------------------------------------------------------------------
from config import Config
from utils import resize


def visit_nominal_simulation(sim_path):
    csv_file = sim_path / "driving_log_normalized.csv"
    uncertainties = []

    with open(csv_file, mode="r") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            uncertainties.append(float(row.get("uncertainty")))

    return uncertainties


def get_nominal_simulation(sims_path):
    for sim_path in sims_path.iterdir():
        if (
            sim_path.is_dir()
            and "normal" in str(sim_path).casefold()
            and sim_path.name.endswith("-uncertainty-evaluated")
        ):
            return sim_path.name


def main():
    curr_project_path = Path(
        os.path.normpath(os.getcwd() + os.sep + os.pardir)
    )  # overcome OS issues

    cfg = Config()
    cfg_pyfile_path = curr_project_path / "config_my.py"
    cfg.from_pyfile(cfg_pyfile_path)

    # Analyse all simulations ------------------------------------------------------------------------------------------
    sims_path = Path(curr_project_path, cfg.SIMULATIONS_DIR)
    nominal_sim = get_nominal_simulation(sims_path)
    uncertainties = visit_nominal_simulation(sims_path / nominal_sim)
    calc_and_store_thresholds(
        uncertainties, Path(curr_project_path, "data", "thresholds")
    )


if __name__ == "__main__":
    main()
