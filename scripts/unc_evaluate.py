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
    print("Predicting on IMG...")
    intermediate_output = [] # list of dictionaries with IMG path, steering angle and uncertainty
    for image in images_path:
        image_to__process = Image.open(str(image))
        image_path_normalize = "/".join(str(image).rsplit('/', 5)[2:]) # normalize path
        steering_angle, uncertainty = uwiz_prediction(image_to__process)
        intermediate_output.append({'uncertainty': uncertainty, 'steering_angle': steering_angle, 'center': image_path_normalize})
    print(">> Predictions done:", len(intermediate_output))
    return intermediate_output


def visit_simulation(sim_path):
    """
    Visit driving_log of a given simulation and extract steering angle and images.
    :param sim_path:
    :return:
    """
    csv_file = sim_path / "driving_log.csv"

    with open(csv_file) as f:
        driving_log = [{k: v for k, v in row.items()}
             for row in csv.DictReader(f, skipinitialspace=True)]
    print("\nReading simulation:\t", driving_log[0]["sim_name"])
    print(">> Row read:\t", len(driving_log))

    # Extract and normalize images paths -------------------------------------------------------------------------------
    images_path = [Path(sim_path, "IMG", d["center"].rsplit("\\",1)[1]) for d in driving_log]
    print(">> Images processed:\t", len(images_path))
    return driving_log, images_path


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

    return sims[0] #TODO: only one for testing, remove [0]


def write_csv(sim_path, driving_log, intermediate_output):
    final_output = []

    print("Writing CSV...")
    for d in driving_log:
        for d_out in intermediate_output:
            if d["center"].rsplit("\\", 1)[1] == d_out["center"].rsplit("/", 1)[1]:  # making csv robust to missing data
                final_output.append(
                    {'frameId': d["frameId"],
                     'model': d["model"],
                     'anomaly_detector': d["anomaly_detector"],
                     'threshold': d["threshold"],
                     'sim_name': d["sim_name"],
                     'lap': d["lap"],
                     'waypoint': d["waypoint"],
                     'loss': d["loss"],
                     'uncertainty': d_out["uncertainty"],
                     'cte': d["cte"],
                     'steering_angle': d_out["steering_angle"],
                     'throttle': d["throttle"],
                     'brake': d["brake"],
                     'crashed': d["crashed"],
                     'distance': d["distance"],
                     'time': d["time"],
                     'ang_diff': d["ang_diff"],
                     'center': d_out["center"],
                     'tot_OBEs': d["tot_OBEs"],
                     'tot_crashes': d["tot_crashes"]
                     })

    folder = Path(str(sim_path) + "-uncertainty-evaluated")
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "driving_log.csv"

    with csv_path.open(mode = "w") as csv_file:
        headers = ["frameId","model","anomaly_detector","threshold","sim_name","lap","waypoint","loss","uncertainty","cte","steering_angle","throttle","brake","crashed","distance","time","ang_diff","center","tot_OBEs","tot_crashes"]
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        for data in final_output:
            writer.writerow(data)
    print(">> CSV file (", len(final_output), "rows) written to:\t", folder)

def main():
    global model

    curr_project_path = Path(os.path.normpath(os.getcwd() + os.sep + os.pardir))  # overcome OS issues

    cfg = Config()
    cfg_pyfile_path = curr_project_path / "config_my.py"
    cfg.from_pyfile(cfg_pyfile_path)

    # Load the self-driving car model ----------------------------------------------------------------------------------
    model_path = os.path.join(curr_project_path, cfg.SDC_MODELS_DIR, cfg.SDC_MODEL_NAME)
    model = uwiz.models.load_model(model_path)

    # Load self-assessment oracle model --------------------------------------------------------------------------------
    sao_path = curr_project_path / cfg.SAO_MODELS_DIR
    encoder_name = "encoder-" + cfg.ANOMALY_DETECTOR_NAME
    encoder_path = sao_path / encoder_name
    encoder = tensorflow.keras.models.load_model(encoder_path)

    decoder_name = "decoder-" + cfg.ANOMALY_DETECTOR_NAME
    decoder_path = sao_path / decoder_name
    decoder = tensorflow.keras.models.load_model(decoder_path)

    anomaly_detection = VAE(
        model_name=cfg.ANOMALY_DETECTOR_NAME,
        loss=cfg.LOSS_SAO_MODEL,
        latent_dim=cfg.SAO_LATENT_DIM,
        encoder=encoder,
        decoder=decoder,
    )
    anomaly_detection.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.SAO_LEARNING_RATE)
    )

    # Analyse all simulations ------------------------------------------------------------------------------------------
    extracted_data = []
    driving_log = []


    sims = collect_simulations(curr_project_path)
    sim = sims # TODO: transform to a for loop
    sim_path = Path(curr_project_path, "simulations", sim) # TODO: transform to a for loop
    driving_log, images_path = visit_simulation(sim_path) # TODO: transform to a for loop (remember to reset those 2 after every analysis)
    intermediate_output = predict_on_IMG(images_path) # TODO: transform to a for loop
    write_csv(sim_path, driving_log, intermediate_output) # TODO: transform to a for loop



if __name__ == "__main__":
    main()
