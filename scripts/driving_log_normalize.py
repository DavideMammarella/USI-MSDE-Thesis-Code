# !!! Simulations must be in the project folder under: simulations/<name_of_simulation>
# !!! <name_of_simulation> folder must contain: IMG folder (with .jpg) and driving_log.csv

# Standard library import ----------------------------------------------------------------------------------------------
import os
import pathlib
from pathlib import Path
import numpy as np
import csv
from PIL import Image
import base64
from io import BytesIO

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


def visit_simulation(sim_path):
    """
    Visit driving_log of a given simulation and extract steering angle and images.
    :param sim_path:
    :return:
    """
    header = ["frameId", "model", "anomaly_detector", "threshold", "sim_name", "lap", "waypoint", "loss",
     "steering_angle", "throttle", "speed", "crashed", "center", "tot_OBEs", "tot_crashes"]
    csv_file_in = sim_path / "driving_log.csv"
    csv_file_out = sim_path / "driving_log_normalized.csv"

    with csv_file_in.open('r') as fp:
        reader = csv.DictReader(fp, fieldnames=header)

        # use newline='' to avoid adding new CR at end of line
        with csv_file_out.open('w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=reader.fieldnames)
            writer.writeheader()
            header_mapping = next(reader)
            writer.writerows(reader)


def collect_simulations(curr_project_path):
    """
    Visit all simulation folders and collect only the names of simulations not previously analysed ("-uncertainty-evaluated").
    :return: list of simulations
    """
    sims_path = Path(curr_project_path, "simulations")

    # First Iteration: collect all simulations -------------------------------------------------------------------------
    _, dirs, _ = next(os.walk(sims_path))  # list all folders in simulations_path (only top level)

    # Second iteration: collect all simulations to exclude -------------------------------------------------------------
    exclude = []
    for d in dirs:
        if "-uncertainty-evaluated" in d:
            exclude.append(d)
            exclude.append(d[:-len("-uncertainty-evaluated")])

    sims_evaluated = int(len(exclude) / 2)
    print("Summary...")
    print(">> Total simulations:\t", len(dirs) - sims_evaluated)
    print(">> Simulations already evaluated:\t", sims_evaluated)

    # Third iteration: collect all simulations to evaluate (excluding those already evaluated) -------------------------
    sims = [d for d in dirs if d not in exclude]
    print(">> Simulations to evaluate:\t", len(sims))

    return sims


def main():
    global model

    curr_project_path = Path(os.path.normpath(os.getcwd() + os.sep + os.pardir))  # overcome OS issues

    cfg = Config()
    cfg_pyfile_path = curr_project_path / "config_my.py"
    cfg.from_pyfile(cfg_pyfile_path)

    # Analyse all simulations ------------------------------------------------------------------------------------------

    sims = collect_simulations(curr_project_path)
    for sim in sims:
        sim_path = Path(curr_project_path, "simulations", sim)
        visit_simulation(sim_path)


if __name__ == "__main__":
    main()