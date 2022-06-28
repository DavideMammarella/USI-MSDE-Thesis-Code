import datetime
import os
import time

import numpy as np
import pandas as pd
from matplotlib import image as mpimg
from matplotlib import pyplot as plt

from utils import navigate
from utils.model import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH

import datetime
import os
import time

import numpy as np
import pandas as pd
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow import keras

from client.train import get_driving_styles
from monitors.selforacle.vae import VAE, Decoder, Encoder
from utils.model import IMAGE_CHANNELS, RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH


def load_vae(load_vae_from_disk):
    """
    Load a trained VAE from disk and compile it, or creates a new one to be trained.
    """
    cfg = navigate.config()
    name = cfg.TRACK + "-" + cfg.LOSS_SAO_MODEL + "-latent" + str(cfg.SAO_LATENT_DIM)

    if load_vae_from_disk:
        encoder = tensorflow.keras.models.load_model(
            cfg.SAO_MODELS_DIR + os.path.sep + "encoder-" + name
        )
        decoder = tensorflow.keras.models.load_model(
            cfg.SAO_MODELS_DIR + os.path.sep + "decoder-" + name
        )
        print("loaded trained VAE from disk")
    else:
        encoder = Encoder().call(
            cfg.SAO_LATENT_DIM,
            RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH * IMAGE_CHANNELS,
        )
        decoder = Decoder().call(cfg.SAO_LATENT_DIM, (cfg.SAO_LATENT_DIM,))
        print("created new VAE model to be trained")

    vae = VAE(
        model_name=name,
        loss=cfg.LOSS_SAO_MODEL,
        latent_dim=cfg.SAO_LATENT_DIM,
        encoder=encoder,
        decoder=decoder,
    )
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.SAO_LEARNING_RATE))

    return vae, name


def load_data_for_vae_training(sampling=None):
    """
    Load training data_nominal and split it into training and validation set
    Load only the first lap for each track
    """
    cfg = navigate.config()
    drive = get_driving_styles(cfg)

    print("Loading training set " + str(cfg.TRACK) + str(drive))

    start = time.time()

    x = None
    path = None
    x_train = None
    x_test = None

    for drive_style in drive:
        try:
            path = os.path.join(
                cfg.TRAINING_DATA_DIR,
                cfg.TRAINING_SET_DIR,
                cfg.TRACK,
                drive_style,
                "driving_log.csv",
            )
            data_df = pd.read_csv(path)

            if sampling is not None:
                print("sampling every " + str(sampling) + "th frame")
                data_df = data_df[data_df.index % sampling == 0]

            if x is None:
                x = data_df[["center"]].values
            else:
                x = np.concatenate((x, data_df[["center"]].values), axis=0)
        except FileNotFoundError:
            print("Unable to read file %s" % path)
            continue

    if x is None:
        print(
            "No driving data_nominal were provided for training. Provide correct paths to the driving_log.csv files"
        )
        exit()

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
        x_train, x_test = train_test_split(x, test_size=cfg.TEST_SIZE, random_state=0)
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
    return x_train, x_test


def load_vae_by_name(name):
    """
    Load a trained VAE from disk by name
    """
    cfg = navigate.config()

    encoder = tensorflow.keras.models.load_model(
        cfg.SAO_MODELS_DIR + os.path.sep + "encoder-" + name
    )
    decoder = tensorflow.keras.models.load_model(
        cfg.SAO_MODELS_DIR + os.path.sep + "decoder-" + name
    )

    vae = VAE(
        model_name=name,
        loss=cfg.LOSS_SAO_MODEL,
        latent_dim=cfg.SAO_LATENT_DIM,
        encoder=encoder,
        decoder=decoder,
    )
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.SAO_LEARNING_RATE))

    return vae


def load_improvement_set(ids):
    """
    Load the paths to the images in the cfg.SIMULATION_NAME directory.
    Filters those having a frame id in the set ids.
    """
    cfg = navigate.config()
    start = time.time()

    x = None
    path = None

    try:
        path = os.path.join(
            cfg.TESTING_DATA_DIR, cfg.SIMULATION_NAME, "driving_log.csv"
        )
        data_df = pd.read_csv(path)

        print("Filtering only false positives")
        data_df = data_df[data_df["frameId"].isin(ids)]

        if x is None:
            x = data_df[["center"]].values
        else:
            x = np.concatenate((x, data_df[["center"]].values), axis=0)

    except FileNotFoundError:
        print("Unable to read file %s" % path)

    if x is None:
        print(
            "No driving data_nominal were provided for training. Provide correct paths to the driving_log.csv files"
        )
        exit()

    duration_train = time.time() - start
    print(
        "Loading improvement data_nominal set completed in %s."
        % str(datetime.timedelta(seconds=round(duration_train)))
    )

    print("False positive data_nominal set: " + str(len(x)) + " elements")

    return x


def load_all_images():
    """
    Load the actual images (not the paths!) in the cfg.SIMULATION_NAME directory.
    """
    cfg = navigate.config()
    path = os.path.join(cfg.TESTING_DATA_DIR, cfg.SIMULATION_NAME, "driving_log.csv")
    data_df = pd.read_csv(path)

    x = data_df["center"]
    print("read %d images from directory %s" % (len(x), cfg.SIMULATION_NAME))

    start = time.time()

    # load the images
    images = np.empty([len(x), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])

    for i, path in enumerate(x):
        image = mpimg.imread(path)  # load center images

        # visualize whether the input_image image as expected
        # import matplotlib.pyplot as plt
        # plt.imshow(image)
        # plt.show()

        images[i] = image

    duration_train = time.time() - start
    print(
        "Loading data_nominal set completed in %s."
        % str(datetime.timedelta(seconds=round(duration_train)))
    )

    print("Data set: " + str(len(images)) + " elements")

    return images


def plot_reconstruction_losses(
    losses, new_losses, name, threshold, new_threshold, data_df
):
    """
    Plots the reconstruction errors for one or two sets of losses, along with given thresholds.
    Crashes are visualized in red.
    """
    plt.figure(figsize=(20, 4))
    x_losses = np.arange(len(losses))

    x_threshold = np.arange(len(x_losses))
    y_threshold = [threshold] * len(x_threshold)
    plt.plot(
        x_threshold,
        y_threshold,
        "--",
        color="black",
        alpha=0.4,
        label="threshold",
    )

    # visualize crashes
    crashes = data_df[data_df["crashed"] == 1]
    is_crash = (crashes.crashed - 1) + threshold
    plt.plot(is_crash, "x:r", markersize=4)

    if new_threshold is not None:
        plt.plot(
            x_threshold,
            [new_threshold] * len(x_threshold),
            color="red",
            alpha=0.4,
            label="new threshold",
        )

    plt.plot(x_losses, losses, "-.", color="blue", alpha=0.7, label="original")
    if new_losses is not None:
        plt.plot(x_losses, new_losses, color="green", alpha=0.7, label="retrained")

    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Number of Instances")
    plt.title("Reconstruction error for " + name)

    plt.savefig("plots/reconstruction-plot-" + name + ".png")

    plt.show()
