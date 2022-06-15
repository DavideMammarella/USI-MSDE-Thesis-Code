import datetime
import os
import time

import numpy as np
import pandas as pd
from matplotlib import image as mpimg
from matplotlib import pyplot as plt

from utils.model import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH


def load_improvement_set(cfg, ids):
    """
    Load the paths to the images in the cfg.SIMULATION_NAME directory.
    Filters those having a frame id in the set ids.
    """
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


def load_all_images(cfg):
    """
    Load the actual images (not the paths!) in the cfg.SIMULATION_NAME directory.
    """
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
