# !!! Simulations must be in the project folder under: simulations/<name_of_simulation>
# !!! <name_of_simulation> folder must contain: IMG folder (with .jpg) and driving_log.csv

# Standard library import ----------------------------------------------------------------------------------------------
import os
import sys
from sys import exit
from warnings import simplefilter
from datetime import datetime
from pathlib import Path
from PIL import Image
import sched, time
import numpy as np
import logging
import signal
import base64
from io import BytesIO
import run

# Local libraries import -----------------------------------------------------------------------------------------------
from config import Config
import utils
from utils import rmse, crop, resize

# Tensorflow library import --------------------------------------------------------------------------------------------
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import load_model
from selforacle.vae import VAE, normalize_and_reshape
import uncertainty_wizard as uwiz


def visit_and_collect_simulations(curr_project_path):
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
    print("Total simulations:\t", len(dirs) - sims_evaluated)
    print("Simulations already evaluated:\t", sims_evaluated)

    # Third iteration: collect all simulations to evaluate (excluding those already evaluated) -------------------------
    sims = [d for d in dirs if d not in exclude]
    print("Simulations to evaluate:\t", len(sims))

    return sims


def main():
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
    sims = visit_and_collect_simulations(curr_project_path)

    # for sim in sims:


if __name__ == "__main__":
    main()
