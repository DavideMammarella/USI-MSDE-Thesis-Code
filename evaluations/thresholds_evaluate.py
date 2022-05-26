# !!! Simulations must be in the project folder under: simulations/<name_of_simulation>
# !!! <name_of_simulation> folder must contain: IMG folder (with .jpg) and driving_log.csv

# This script must be used only with UWIZ models

import sys
sys.path.append("..")

# Standard library import ----------------------------------------------------------------------------------------------
import os
import pathlib
from pathlib import Path
import numpy as np
import csv
from PIL import Image
import base64
from io import BytesIO
import utils_logging
from evaluations.utils_threshold import calc_and_store_thresholds

import logging
logger = logging.Logger("utils_thresholds")
utils_logging.log_info(logger)

# Local libraries import -----------------------------------------------------------------------------------------------
from config import Config
import utils
from utils import resize

# Tensorflow library import --------------------------------------------------------------------------------------------
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import load_model
from selforacle.vae import VAE, normalize_and_reshape
import uncertainty_wizard as uwiz

# Model setup ----------------------------------------------------------------------------------------------------------
model = None
prev_image_array = None
anomaly_detection = None
autoencoder_model = None
frame_id = 0
batch_size = 1
uncertainty = -1

# TODO: add tqdm

def visit_simulation(sim_path):
    """
    Visit driving_log of a given simulation and extract uncertainties in a numpy array.
    """
    csv_file = sim_path / "driving_log.csv"
    uncertainties = []

    with open(csv_file, mode='r') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            uncertainties.append(float(row['uncertainty']))

    return uncertainties


def collect_simulations(curr_project_path):
    """
    Visit all simulation folders and collect only names that contain ("-uncertainty-evaluated").
    :return: list of simulations
    """
    sims_path = Path(curr_project_path, "simulations")
    sims = []

    for sim_path in sims_path.iterdir():
        if sim_path.is_dir() and sim_path.name.endswith("-uncertainty-evaluated"):
            sims.append(sim_path.name)

    return sims


def main():
    global model

    curr_project_path = Path(os.path.normpath(os.getcwd() + os.sep + os.pardir))  # overcome OS issues

    cfg = Config()
    cfg_pyfile_path = curr_project_path / "config_my.py"
    cfg.from_pyfile(cfg_pyfile_path)

    # Analyse nominal simulation ---------------------------------------------------------------------------------------
    extracted_data = []
    driving_log = []

    sims = collect_simulations(curr_project_path)
    for sim in sims:
        if "Normal" in sim:
            logger.info("Nominal simulation: " + sim + "\n")
            sim_path = Path(curr_project_path, "simulations", sim)
            uncertainties = visit_simulation(sim_path)
            print(uncertainties)
            calc_and_store_thresholds(uncertainties, sim_path)
            driving_log = []
            images_path = []


if __name__ == "__main__":
    main()
