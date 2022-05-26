# !!! Simulations must be in the project folder under: simulations/<name_of_simulation>
# !!! <name_of_simulation> folder must contain: IMG folder (with .jpg) and driving_log.csv

# This script must be used only with UWIZ models

# Standard library import ----------------------------------------------------------------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import sys
sys.path.append("..")

import pathlib
from pathlib import Path
import numpy as np
import csv
from PIL import Image
import base64
from io import BytesIO
from tqdm import tqdm

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


def uwiz_prediction(image):
    """
    Predict steering angle and uncertainty of an image using the uncertainty wizard.
    :param model: Uncertainty wizard model
    :param image: Image from simulation
    :return: steering angle and uncertainty
    """
    global model

    # Process image for prediction -------------------------------------------------------------------------------------
    image = np.asarray(image)  # from PIL image to numpy array
    image = utils.preprocess(image)  # apply pre-processing
    image = np.array([image])  # model expects 4D array

    # Predict steering angle and uncertainty ---------------------------------------------------------------------------
    outputs, unc = model.predict_quantified(image, quantifier="std_dev", sample_size=20, batch_size=20)
    steering_angle = outputs[0][0]
    uncertainty = unc[0][0]

    return steering_angle, uncertainty


def predict_on_IMG(images_path):
    """
    Use IMG previously extracted and make predictions.
    :param sim_path: Path of simulations folder inside the project
    :param images_path: list of IMGs title
    :return: dictionary with uncertainty, steering angle and IMG path
    """
    intermediate_output = []  # list of dictionaries with IMG path, steering angle and uncertainty
    for image in tqdm(images_path, position=0, leave=False):
        image_to__process = Image.open(str(image))
        steering_angle, uncertainty = uwiz_prediction(image_to__process)
        intermediate_output.append(
            {'uncertainty': uncertainty, 'steering_angle': steering_angle, 'center': image})
    return intermediate_output


def visit_simulation(sim_path):
    csv_file = sim_path / "driving_log_normalized.csv"
    print("\nReading simulation:\t", str(sim_path))
    with open(csv_file) as f:
        driving_log_normalized = [{k: v for k, v in row.items()}
                                  for row in csv.DictReader(f, skipinitialspace=True)]
    images_path = [Path(sim_path, d.get("center")) for d in driving_log_normalized]

    return driving_log_normalized, images_path


def collect_simulations(sims_path):
    # First Iteration: collect all simulations -------------------------------------------------------------------------
    _, dirs, _ = next(os.walk(sims_path))  # list all folders in simulations_path (only top level)

    # Second iteration: collect all simulations to exclude -------------------------------------------------------------
    exclude = []
    for d in dirs:
        if "-uncertainty-evaluated" in d:
            exclude.append(d)
            exclude.append(d[:-len("-uncertainty-evaluated")])

    sims_evaluated = int(len(exclude) / 2)
    print(">> Total simulations:\t", len(dirs) - sims_evaluated)
    print(">> Simulations already evaluated:\t", sims_evaluated)

    # Third iteration: collect all simulations to evaluate (excluding those already evaluated) -------------------------
    sims = [d for d in dirs if d not in exclude]
    print(">> Simulations to evaluate:\t", len(sims))

    return sims


def write_csv(sim_path, driving_log, intermediate_output):
    final_output = []

    for d in driving_log:
        for d_out in intermediate_output:
            if d.get("center") == d_out.get("center"):  # making csv robust to missing data
                final_output.append(
                    {'frame_id': d.get("frame_id"),
                     'model': d.get("model"),
                     'anomaly_detector': d.get("anomaly_detector"),
                     'threshold': d.get("threshold"),
                     'sim_name': d.get("sim_name"),
                     'lap': d.get("lap"),
                     'waypoint': d.get("waypoint"),
                     'loss': d.get("loss"),
                     'uncertainty': d_out.get("uncertainty"),
                     'cte': d.get("cte"),
                     'steering_angle': d_out.get("steering_angle"),
                     'throttle': d.get("throttle"),
                     'speed': d.get("speed"),
                     'brake': d.get("brake"),
                     'crashed': d.get("crashed"),
                     'distance': d.get("distance"),
                     'time': d.get("time"),
                     'ang_diff': d.get("ang_diff"),
                     'center': d_out.get("center"),
                     'tot_OBEs': d.get("tot_obes"),
                     'tot_crashes': d.get("tot_crashes")
                     })

    folder = Path(str(sim_path) + "-uncertainty-evaluated")
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "driving_log_normalized.csv"

    with csv_path.open(mode="w") as csv_file:
        headers = ["frame_id", "model", "anomaly_detector", "threshold", "sim_name", "lap", "waypoint", "loss",
                   "uncertainty", "cte", "steering_angle", "throttle", "speed", "brake", "crashed", "distance", "time",
                   "ang_diff", "center", "tot_OBEs", "tot_crashes"]
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        for data in final_output:
            writer.writerow(data)


def main():
    curr_project_path = Path(os.path.normpath(os.getcwd() + os.sep + os.pardir))  # overcome OS issues
    cfg = Config()
    cfg_pyfile_path = curr_project_path / "config_my.py"
    cfg.from_pyfile(cfg_pyfile_path)

    # Load the self-driving car model ----------------------------------------------------------------------------------
    global model
    model_path = os.path.join(curr_project_path, cfg.SDC_MODELS_DIR, cfg.SDC_MODEL_NAME)
    model = uwiz.models.load_model(model_path)

    # Analyse all simulations ------------------------------------------------------------------------------------------
    sims_path = Path(curr_project_path, cfg.SIMULATIONS_DIR)
    simulations = collect_simulations(sims_path)

    for sim in simulations:
        sim_path = Path(curr_project_path, cfg.SIMULATIONS_DIR, sim)
        driving_log, images_path = visit_simulation(sim_path)
        print("Calculating uncertainties using UWIZ on IMGs...")
        predictions = predict_on_IMG(images_path)
        print(">> Predictions done:", len(predictions))
        print("Writing CSV...")
        write_csv(sim_path, driving_log, predictions)
        print(">> CSV written to:\t" + str(sim_path) + "-uncertainty-evaluated")


if __name__ == "__main__":
    main()
