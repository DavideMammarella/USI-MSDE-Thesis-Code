# !!! Simulations must be in the project folder under: simulations/<name_of_simulation>
# !!! <name_of_simulation> folder must contain: IMG folder (with .jpg) and driving_log.csv
# ALL SIMULATIONS MUST BE NORMALIZED using simulations_normalizer.py

# This script must be used only with UWIZ models

# Standard library import ----------------------------------------------------------------------------------------------
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings

import sys

sys.path.append("..")

import base64
import csv
import pathlib
import statistics
from io import BytesIO
from pathlib import Path

import numpy as np

# Tensorflow library import --------------------------------------------------------------------------------------------
import tensorflow
import uncertainty_wizard as uwiz
from PIL import Image
from tensorflow import keras
from tensorflow.keras.models import load_model
from tqdm import tqdm

import utils

# Local libraries import -----------------------------------------------------------------------------------------------
from config import Config
from data.simulations_normalizer import (
    check_driving_log,
    normalize_img_path,
    write_driving_log,
)
from selforacle.vae import VAE, normalize_and_reshape
from utils import resize


def get_avg_unc(sim_path):
    csv_file = sim_path / "driving_log_normalized.csv"

    uncertainties = []
    with open(csv_file) as f:
        driving_log_normalized = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]

    for d in driving_log_normalized:
        uncertainties.append(float(d.get("uncertainty")))
    f.close()

    avg_unc = statistics.mean(uncertainties)

    return avg_unc


def collect_simulations(sims_path):
    sims = []
    for sim_path in sims_path.iterdir():
        if sim_path.is_dir() and sim_path.name.endswith(
            "-uncertainty-evaluated"
        ):
            sims.append(sim_path.name)
    print(">> Collected simulations: " + str(len(sims)) + "\n")
    return sims


def main():
    root_dir = utils.get_project_root()
    cfg = Config()
    cfg_pyfile_path = root_dir / "config_my.py"
    cfg.from_pyfile(cfg_pyfile_path)

    sims_path = Path(root_dir, cfg.SIMULATIONS_DIR)
    simulations = collect_simulations(sims_path)

    print(
        "{:<30} {:<15}\n----------------------------------------------------".format(
            "Model", "Average Uncertainty"
        )
    )
    avgs = []
    for sim in simulations:
        sim_path = Path(root_dir, cfg.SIMULATIONS_DIR, sim)
        avg_unc = get_avg_unc(sim_path)
        avgs.append(avg_unc)
        print(
            "{:<30} {:<15}".format(
                str(sim).replace("-uncertainty-evaluated", ""), avg_unc
            )
        )
    print("----------------------------------------------------\n")

    avgs_sorted = sorted(avgs)
    nominal_unc = avgs_sorted[0]
    min_unc = avgs_sorted[1]  # second min, because first is nominal
    max_unc = avgs_sorted[-1]
    avg_unc = statistics.mean(avgs_sorted)
    print(
        ">> Nominal Unc: {} \n>> Min Unc: {} \n>> Max Unc: {} \n>> Avg Unc: {}".format(
            nominal_unc, min_unc, max_unc, avg_unc
        )
    )


if __name__ == "__main__":
    main()
