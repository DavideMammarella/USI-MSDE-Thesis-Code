# Copyright 2021 Testing Automated @ Università della Svizzera italiana (USI)
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import datetime
import os
import time
from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow
import tensorflow as tf
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense

from utils import navigate
from utils.models_train import get_driving_styles
from utils.sdc import (
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    RESIZED_IMAGE_HEIGHT,
    RESIZED_IMAGE_WIDTH,
)

original_dim = RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH * IMAGE_CHANNELS


class Sampling(layers.Layer):
    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(
            shape=(batch, dim), mean=0.0, stddev=1.0
        )
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    def call(self, latent_dim, inputs, **kwargs):
        inputs = keras.Input(shape=(original_dim,))
        x = Dense(512, activation="relu")(inputs)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(
            inputs=inputs, outputs=[z_mean, z_log_var, z], name="encoder"
        )
        # encoder.summary()

        return encoder


class Decoder(layers.Layer):
    def call(self, latent_dim, latent_inputs, **kwargs):
        latent_inputs = keras.Input(shape=(latent_dim,), name="z_sampling")
        x = Dense(
            512,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l1(0.001),
        )(latent_inputs)
        decoder_outputs = Dense(original_dim, activation="sigmoid")(x)

        decoder = keras.Model(
            inputs=latent_inputs, outputs=decoder_outputs, name="decoder"
        )
        # decoder.summary()

        return decoder


class VAE(keras.Model, ABC):
    """
    Define the VAE as a `Model` with a custom `train_step`
    """

    def __init__(self, model_name, loss, latent_dim, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.model_name = model_name
        self.intermediate_dim = 512
        self.latent_dim = latent_dim
        self.lossFunc = loss
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.mean_squared_error(data, reconstruction)
            )
            reconstruction_loss *= RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT

            if self.lossFunc == "VAE":
                kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                kl_loss = tf.reduce_mean(kl_loss)
                kl_loss *= -0.5
                total_loss = reconstruction_loss + kl_loss
                grads = tape.gradient(total_loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                return {
                    "loss": total_loss,
                    "reconstruction_loss": reconstruction_loss,
                    "kl_loss": kl_loss,
                }
            else:
                total_loss = reconstruction_loss
                grads = tape.gradient(total_loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                return {
                    "loss": total_loss,
                    "reconstruction_loss": reconstruction_loss,
                }

    def call(self, inputs, **kwargs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.mean_squared_error(inputs, reconstruction)
        )
        reconstruction_loss *= RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT

        if self.lossFunc[0] == "VAE":
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
            self.add_metric(kl_loss, name="kl_loss", aggregation="mean")
            self.add_metric(total_loss, name="total_loss", aggregation="mean")
            self.add_metric(
                reconstruction_loss,
                name="reconstruction_loss",
                aggregation="mean",
            )
            return reconstruction
        else:
            total_loss = reconstruction_loss
            self.add_metric(total_loss, name="total_loss", aggregation="mean")
            self.add_metric(
                reconstruction_loss,
                name="reconstruction_loss",
                aggregation="mean",
            )
            return reconstruction


def get_input_shape():
    return (original_dim,)


def get_image_dim():
    return RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH * IMAGE_CHANNELS


def normalize_and_reshape(x):
    x = x.astype("float32") / 255.0
    x = x.reshape(-1, RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH * IMAGE_CHANNELS)
    return x


def reshape(x):
    x = x.reshape(-1, RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH * IMAGE_CHANNELS)
    return x


def load_vae(name=None):
    """
    Load a trained VAE from disk and compile it, or creates a new one to be trained.
    """
    cfg = navigate.config()
    sao_path = str(navigate.sao_dir())

    if name:
        encoder = tensorflow.keras.models.load_model(
            str(Path(sao_path, "encoder-" + name))
        )
        decoder = tensorflow.keras.models.load_model(
            str(Path(sao_path, "decoder-" + name))
        )
        print(">> Loaded trained VAE from disk")
    else:
        name = (
            cfg.TRACK + "-" + cfg.LOSS_SAO_MODEL + "-latent" + str(cfg.SAO_LATENT_DIM)
        )
        encoder = Encoder().call(
            cfg.SAO_LATENT_DIM,
            RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH * IMAGE_CHANNELS,
        )
        decoder = Decoder().call(cfg.SAO_LATENT_DIM, (cfg.SAO_LATENT_DIM,))
        print("\n>> Created new VAE model to be trained")

    vae = VAE(
        model_name=name,
        loss=cfg.LOSS_SAO_MODEL,
        latent_dim=cfg.SAO_LATENT_DIM,
        encoder=encoder,
        decoder=decoder,
    )
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.SAO_LEARNING_RATE))

    return vae, name


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
