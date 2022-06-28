import datetime
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import navigate
from utils.sdc import *

np.random.seed(0)


def get_driving_styles():
    """
    Retrieves the driving styles to compose the training set
    """
    cfg = navigate.config()

    if cfg.TRACK == "track1":
        return cfg.TRACK1_DRIVING_STYLES
    elif cfg.TRACK == "track2":
        return cfg.TRACK2_DRIVING_STYLES
    elif cfg.TRACK == "track3":
        return cfg.TRACK3_DRIVING_STYLES
    else:
        print("Invalid TRACK option within the config file")
        exit(1)


def load_training_data(model_type, sampling=None):
    """
    Load training data_nominal and split it into training and validation set
    """
    cfg = navigate.config()
    drive = get_driving_styles()

    print("Loading training set " + str(cfg.TRACK) + str(drive))

    start = time.time()

    x = None
    y = None
    path = None
    x_train = None
    y_train = None
    x_test = None
    y_test = None

    for drive_style in drive:
        try:
            training_path = navigate.training_set_dir()
            path = Path(
                training_path,
                cfg.TRACK,
                drive_style,
                "driving_log.csv",
            )
            data_df = pd.read_csv(path)
            data_df["center"] = str(training_path) + data_df["center"].astype(str)
            data_df["left"] = str(training_path) + data_df["left"].astype(str)
            data_df["right"] = str(training_path) + data_df["right"].astype(str)
            if model_type=="autoencoder" and sampling is not None:
                print("sampling every " + str(sampling) + "th frame")
                data_df = data_df[data_df.index % sampling == 0]
            if x is None:
                x = data_df[["center", "left", "right"]].values
                y = data_df["steering"].values
            else:
                x = np.concatenate(
                    (x, data_df[["center", "left", "right"]].values), axis=0
                )
                y = np.concatenate((y, data_df["steering"].values), axis=0)
        except FileNotFoundError:
            print("Unable to read file %s" % path)
            continue

    if x is None:
        print(
            "No driving data_nominal were provided for training. Provide correct paths to the driving_log.csv files"
        )
        exit()

    if model_type=="autoencoder":
        if cfg.TRACK == "track1":
            print(
                "For %s, we use only the first %d images (~1 lap)"
                % (cfg.TRACK, cfg.TRACK1_IMG_PER_LAP)
            )
            x = x[: cfg.TRACK1_IMG_PER_LAP]
        elif cfg.TRACK == "track2":
            print(
                "For %s, we use only the first %d images (~1 lap)"
                % (cfg.TRACK, cfg.TRACK2_IMG_PER_LAP)
            )
            x = x[: cfg.TRACK2_IMG_PER_LAP]
        elif cfg.TRACK == "track3":
            print(
                "For %s, we use only the first %d images (~1 lap)"
                % (cfg.TRACK, cfg.TRACK3_IMG_PER_LAP)
            )
            x = x[: cfg.TRACK3_IMG_PER_LAP]
        else:
            print("Incorrect cfg.TRACK option provided")
            exit()

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=cfg.TEST_SIZE, random_state=0
        )
    except TypeError:
        print("Missing header to csv files")
        exit()

    duration_train = time.time() - start
    print(
        "Loading training set completed in %s."
        % str(datetime.timedelta(seconds=round(duration_train)))
    )

    print("Data set: " + str(len(x)) + " elements")
    print("Training set: " + str(len(x_train)) + " elements")
    print("Test set: " + str(len(x_test)) + " elements")
    return x_train, x_test, y_train, y_test