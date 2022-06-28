from pathlib import Path

import tensorflow
import uncertainty_wizard as uwiz
from tensorflow import keras
from tensorflow.keras import backend as K

from monitors.selforacle.vae import VAE


def rmse(y_true, y_pred):
    """
    Calculates RMSE
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def load_sdc_model(cfg, model_path):
    if cfg.SDC_MODEL_TYPE == "uwiz":
        model = uwiz.models.load_model(model_path)
    elif cfg.SDC_MODEL_TYPE == "chauffeur":
        model = tensorflow.keras.models.load_model(
            model_path, custom_objects={"rmse": rmse}
        )
    elif (
        cfg.SDC_MODEL_TYPE == "dave2"
        or cfg.SDC_MODEL_TYPE == "epoch"
        or cfg.SDC_MODEL_TYPE == "commaai"
    ):
        model = tensorflow.keras.models.load_model(model_path)
    else:
        print("cfg.SDC_MODEL_TYPE option unknown. Exiting...")
        exit()
    return model


def load_sao_models(cfg, sao_path):
    encoder = tensorflow.keras.models.load_model(
        str(Path(sao_path, "encoder-" + cfg.ANOMALY_DETECTOR_NAME))
    )
    decoder = tensorflow.keras.models.load_model(
        str(Path(sao_path, "decoder-" + cfg.ANOMALY_DETECTOR_NAME))
    )

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
    return anomaly_detection
