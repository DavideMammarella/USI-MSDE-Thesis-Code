import datetime
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from batch_generator import Generator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from utils import navigate
from utils.model import *

np.random.seed(0)


def get_driving_styles(cfg):
    """
    Retrieves the driving styles to compose the training set
    """
    if cfg.TRACK == "track1":
        return cfg.TRACK1_DRIVING_STYLES
    elif cfg.TRACK == "track2":
        return cfg.TRACK2_DRIVING_STYLES
    elif cfg.TRACK == "track3":
        return cfg.TRACK3_DRIVING_STYLES
    else:
        print("Invalid TRACK option within the config file")
        exit(1)


# TODO: a bit redundant w/ load_data_for_vae but this one loads y as well
def load_data(cfg):
    """
    Load training data_nominal and split it into training and validation set
    """
    drive = get_driving_styles(cfg)

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


def train_model(model, cfg, x_train, x_test, y_train, y_test):
    models_path = navigate.models_dir()
    """
    Train the self-driving car model
    """
    if cfg.USE_PREDICTIVE_UNCERTAINTY:
        model_name = cfg.TRACK + "-" + cfg.SDC_MODEL_TYPE + "-{epoch:03d}"
        model_path = Path(models_path, model_name)
    else:
        model_name = cfg.TRACK + "-" + cfg.SDC_MODEL_TYPE.replace(".h5", "") + "-{epoch:03d}.h5"
        model_path = Path(models_path, model_name)

    checkpoint = ModelCheckpoint(
        str(model_path), monitor="val_loss", verbose=0, save_best_only=True, mode="auto"
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="loss", min_delta=0.0005, patience=10, mode="auto"
    )

    model.compile(loss="mean_squared_error", optimizer=Adam(lr=cfg.LEARNING_RATE))

    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    x_test, y_test = shuffle(x_test, y_test, random_state=0)

    train_generator = Generator(x_train, y_train, True, cfg)
    val_generator = Generator(x_test, y_test, False, cfg)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=cfg.NUM_EPOCHS_SDC_MODEL,
        callbacks=[checkpoint, early_stop],
        verbose=1,
    )

    # summarize history for loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()

    if cfg.USE_PREDICTIVE_UNCERTAINTY:
        model_name = cfg.TRACK + "-" + cfg.SDC_MODEL_TYPE + "-final"
        model_path = Path(models_path, model_name)
    else:
        model_name = cfg.TRACK + "-" + cfg.SDC_MODEL_TYPE.replace(".h5", "") + "-final.h5"
        model_path = Path(models_path, model_name)

    # save the last model anyway (might not be the best)
    model.save(str(model_path))


def main():
    """
    Load train/validation data_nominal set and train the model
    """
    cfg = navigate.config()

    x_train, x_test, y_train, y_test = load_data(cfg)

    model = build_model(cfg.SDC_MODEL_TYPE, cfg.USE_PREDICTIVE_UNCERTAINTY)

    train_model(model, cfg, x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
