# Copyright 2021 Testing Automated @ Università della Svizzera italiana (USI)
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.
import datetime
import gc
import os
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import backend as K
from sklearn.utils import shuffle

from utils.vae_batch_generator import Generator
from utils import navigate
from utils.models_train import load_training_data
from utils.vae import load_vae
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings

def train_vae_model(
    cfg, vae, name, x_train, x_test, delete_model, retraining, sample_weights
):
    """
    Train the VAE model
    """

    sao_path = str(navigate.sao_dir())
    my_encoder = Path(sao_path, "encoder-" + name)
    my_decoder = Path(sao_path, "decoder-" + name)

    if delete_model or "RETRAINED" in name:
        shutil.rmtree(my_encoder, ignore_errors=True)
        print(">> Model %s deleted" % ("encoder-" + name))
        shutil.rmtree(my_decoder, ignore_errors=True)
        print(">> Model %s deleted" % ("decoder-" + name))

    if my_encoder.exists() and my_decoder.exists():
        if retraining:
            print(
                "\nModel %s already exists and retraining=true. Keep training."
                % str(name)
            )
            my_encoder = Path(sao_path, "encoder-" + name)
            my_decoder = Path(sao_path, "decoder-" + name)
        else:
            print(
                "\nModel %s already exists and retraining=false. Quit training."
                % str(name)
            )
            return
    else:
        print("\nModel %s does not exist. Training..." % str(name))

    start = time.time()

    x_train = shuffle(x_train, random_state=0)
    x_test = shuffle(x_test, random_state=0)

    # set uniform weights to all samples
    weights = np.ones(shape=(len(x_train),))

    # weighted retraining
    if retraining:
        if sample_weights is not None:
            weights = sample_weights

    train_generator = Generator(x_train, True, cfg, weights)
    val_generator = Generator(x_test, True, cfg, weights)

    history = vae.fit(
        train_generator,
        validation_data=val_generator,
        shuffle=True,
        epochs=cfg.NUM_EPOCHS_SAO_MODEL,
        verbose=0,
    )

    duration_train = time.time() - start
    print(
        ">> Training completed in %s."
        % str(datetime.timedelta(seconds=round(duration_train)))
    )

    # Plot the autoencoder training history
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_total_loss"])
    plt.ylabel("reconstruction loss (" + str(cfg.LOSS_SAO_MODEL) + ")")
    plt.xlabel("epoch")
    plt.title("training-" + str(vae.model_name))
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig(sao_path + "/history-training-" + str(vae.model_name) + ".png")
    plt.show()

    # save the last model
    vae.encoder.save(my_encoder.__str__(), save_format="tf", include_optimizer=True)
    vae.decoder.save(my_decoder.__str__(), save_format="tf", include_optimizer=True)

    del vae
    K.clear_session()
    gc.collect()


def main():
    cfg = navigate.config()
    model_type = "autoencoder"

    x_train, x_test, _, _ = load_training_data(model_type)
    vae, name = load_vae()

    train_vae_model(
        cfg,
        vae,
        name,
        x_train,
        x_test,
        delete_model=True,
        retraining=False,
        sample_weights=None,
    )


if __name__ == "__main__":
    main()
