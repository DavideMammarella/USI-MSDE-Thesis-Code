# !!! Simulations must be in the project folder under: simulations/<name_of_simulation>
# !!! <name_of_simulation> folder must contain: IMG folder (with .jpg) and driving_log.csv
# ALL SIMULATIONS MUST BE NORMALIZED using simulations_normalizer.py

# This script must be used only with UWIZ models

import os

from utils.augmentation import preprocess, resize
from utils.custom_csv import visit_simulation, create_driving_log_norm
from utils.sdc import load_sdc_model
from utils.vae import load_vae, normalize_and_reshape

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings

from pathlib import Path

import numpy as np
import uncertainty_wizard as uwiz
from PIL import Image
from tqdm import tqdm

from utils import navigate, custom_csv


def sao_prediction(image):
    global anomaly_detector
    global sdc_model

    batch_size = 128
    image = np.asarray(image)  # from PIL image to numpy array

    # Calculate loss -------------------------------------------------------------------------------------------
    image_copy = np.copy(image)
    image_copy = resize(image_copy)
    image_copy = normalize_and_reshape(image_copy)
    loss = anomaly_detector.test_on_batch(image_copy)[2]

    # Process image for prediction -------------------------------------------------------------------------------------
    image = preprocess(image)  # apply pre-processing
    image = np.array([image])  # model expects 4D array

    # Predict steering angle and uncertainty ---------------------------------------------------------------------------
    x = np.concatenate(
        [image for idx in range(batch_size)]
    )  # take batch of data_nominal
    outputs = sdc_model.predict_on_batch(
        x
    )  # save predictions from a sample pass
    steering_angle = outputs.mean(axis=0)[
        0
    ]  # average over all passes is the final steering angle

    return steering_angle, loss


def predict_on_IMG(images_dict):
    predictions_dict = (
        []
    )  # list of dictionaries with IMG path, steering angle and uncertainty
    for d in tqdm(images_dict, position=0, leave=False):
        image_to__process = Image.open(str(d.get("center")))
        steering_angle, loss = sao_prediction(image_to__process)
        predictions_dict.append(
            {
                "frame_id": d.get("frame_id"),
                "loss": loss,
                "steering_angle": steering_angle,
                "center": str(d.get("center")).rsplit("/", 1)[-1],
            }
        )
    return predictions_dict


def main():
    cfg = navigate.config()
    sims_path = navigate.simulations_dir()
    simulations = navigate.collect_simulations_to_evaluate_loss(sims_path)
    print(simulations)

    # Load the self-driving car model ----------------------------------------------------------------------------------
    global anomaly_detector
    global sdc_model
    sdc_model = load_sdc_model()  # load CAR model
    anomaly_detector, _ = load_vae(cfg.ANOMALY_DETECTOR_NAME)  # load AUTOENCODER model

    for sim in simulations:
        sim_path = Path(sims_path, sim)
        driving_log, images_dict = visit_simulation(sim_path)

        print("Calculating LOSS using SAO on IMGs...")
        predictions_dict = predict_on_IMG(images_dict)
        print(">> Predictions done:", len(predictions_dict))

        print("Writing CSV...")
        create_driving_log_norm(sim_path, driving_log, predictions_dict, "-loss-evaluated")
        print(">> CSV written to:\t" + str(sim_path) + "-loss-evaluated")


if __name__ == "__main__":
    main()
