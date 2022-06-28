# !!! Simulations must be in the project folder under: simulations/<name_of_simulation>
# !!! <name_of_simulation> folder must contain: IMG folder (with .jpg) and driving_log.csv
# ALL SIMULATIONS MUST BE NORMALIZED using simulations_normalizer.py

# This script must be used only with UWIZ models

import os

from utils.augmentation import preprocess
from utils.custom_csv import visit_simulation, create_driving_log_norm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings

from pathlib import Path

import numpy as np
import uncertainty_wizard as uwiz
from PIL import Image
from tqdm import tqdm

from utils import navigate, custom_csv


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
    image = preprocess(image)  # apply pre-processing
    image = np.array([image])  # model expects 4D array

    # Predict steering angle and uncertainty ---------------------------------------------------------------------------
    outputs, unc = model.predict_quantified(
        image, quantifier="std_dev", sample_size=20, batch_size=20
    )
    steering_angle = outputs[0][0]
    uncertainty = unc[0][0]

    return steering_angle, uncertainty


def predict_on_IMG(images_dict):
    """
    Use IMG previously extracted and make predictions.
    :param sim_path: Path of simulations folder inside the project
    :param images_path: list of IMGs title
    :return: dictionary with uncertainty, steering angle and IMG path
    """
    predictions_dict = (
        []
    )  # list of dictionaries with IMG path, steering angle and uncertainty
    for d in tqdm(images_dict, position=0, leave=False):
        image_to__process = Image.open(str(d.get("center")))
        steering_angle, uncertainty = uwiz_prediction(image_to__process)
        predictions_dict.append(
            {
                "frame_id": d.get("frame_id"),
                "uncertainty": uncertainty,
                "steering_angle": steering_angle,
                "center": str(d.get("center")).rsplit("/", 1)[-1],
            }
        )
    return predictions_dict


def main():
    cfg = navigate.config()
    model_path = navigate.model_path()
    sims_path = navigate.simulations_dir()
    simulations = navigate.collect_simulations_to_evaluate(sims_path)

    # Load the self-driving car model ----------------------------------------------------------------------------------
    global model
    model = uwiz.models.load_model(str(model_path))

    for sim in simulations:
        sim_path = Path(sims_path, sim)
        driving_log, images_dict = visit_simulation(sim_path)

        print("Calculating white_box using UWIZ on IMGs...")
        predictions_dict = predict_on_IMG(images_dict)
        print(">> Predictions done:", len(predictions_dict))

        print("Writing CSV...")
        create_driving_log_norm(sim_path, driving_log, predictions_dict)
        print(">> CSV written to:\t" + str(sim_path) + "-uncertainty-evaluated")

        # print("Check CSV integrity (Original Normalized vs Predicted)...")
        # check_driving_log(
        #     Path(sim_path / "driving_log_normalized.csv"),
        #     Path(
        #         str(sim_path)
        #         + "-uncertainty-evaluated/driving_log_normalized.csv"
        #     ),
        # )
        # print(">> CSV is OK!")


if __name__ == "__main__":
    main()
