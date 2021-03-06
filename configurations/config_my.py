# Copyright 2021 by Andrea Stocco, the Software Institute at USI.
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME.
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

# FOLDERS PATH -> DO NO TOUCH
DATA_DIR = "data"
TRAINING_DATA_DIR = "datasets"  # root folder for all driving training sets)
SIMULATIONS_DIR = "simulations"  # folder where to store the simulations
SDC_MODELS_DIR = "models"  # self-driving car models
SAO_MODELS_DIR = "sao"  # trained autoencoder-based self-assessment oracle models
TRAINING_SET_DIR = "dataset5"  # the driving training set to use
SIMULATOR_DIR = "server"  # name of the server to use
RESULTS_DIR = "results"  # folder where to store the performance metrics

# simulations settings
TRACK = "track1"  # ["track1"|"track2"|"track3"|"track1","track2","track3"] the race track to use
TRACK1_DRIVING_STYLES = ["normal", "recovery", "reverse"]
TRACK2_DRIVING_STYLES = [
    "normal",
    "recovery",
    "recovery2",
    "recovery3",
    "reverse",
    "sport_normal",
    "sport_reverse",
]
TRACK3_DRIVING_STYLES = [
    "normal",
    "recovery",
    "recovery2",
    "reverse",
    "sport_normal",
]
TRACK1_IMG_PER_LAP = 1140
TRACK2_IMG_PER_LAP = 1870
TRACK3_IMG_PER_LAP = 1375

# self-driving car model settings
TEST_SIZE = 0.2  # split of training data used for the validation set (keep it low)
SDC_MODEL_NAME = "track1-uwiz-final"
NUM_EPOCHS_SDC_MODEL = 500  # training epochs for the self-driving car model
BATCH_SIZE = 128  # number of samples per gradient update
SAVE_BEST_ONLY = True  # only saves when the model is considered the "best" according to the quantity monitored
LEARNING_RATE = 1.0e-4  # amount that the weights are updated during training
USE_PREDICTIVE_UNCERTAINTY = True  # use MC-Dropout model

# Udacity simulation settings
ANOMALY_DETECTOR_NAME = "track1-MSE-latent2"
SIMULATION_NAME = "track1-sunny"
TESTING_DATA_DIR = (
    None  # Udacity simulations logs (write simulations if you want to record them)
)
MAX_SPEED = 35  # car's max speed, capped at 35 mph (default)
MIN_SPEED = 10  # car's min speed, capped at 10 mph (default)
SAO_THRESHOLD = 180  # the SAO threshold
MAX_LAPS = 1  # max laps before sim stops
FPS = 15

# autoencoder-based self-assessment oracle settings
NUM_EPOCHS_SAO_MODEL = (
    10  # training epochs for the autoencoder-based self-assessment oracle
)
SAO_LATENT_DIM = 2  # dimension of the latent space
LOSS_SAO_MODEL = "MSE"  # "VAE"|"MSE" objective function for the autoencoder-based self-assessment oracle
# DO NOT TOUCH THESE
SAO_BATCH_SIZE = 128
SAO_LEARNING_RATE = 0.0001

# adaptive anomaly detection settings
UNCERTAINTY_TOLERANCE_LEVEL = 0.00328  # from Michelmore et al.
CTE_TOLERANCE_LEVEL = 2.5  # from Stocco et al.
IMPROVEMENT_RATIO = 1
