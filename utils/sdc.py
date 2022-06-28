from pathlib import Path

import tensorflow
import uncertainty_wizard as uwiz
from tensorflow import keras
from tensorflow.keras import backend as K

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda
from tensorflow.keras.regularizers import l2

from utils import navigate

RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH = 80, 160
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS)

def rmse(y_true, y_pred):
    """
    Calculates RMSE
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def load_sdc_model():
    cfg = navigate.config()
    model_path = str(Path(navigate.models_dir(), cfg.SDC_MODEL_NAME))

    if "uwiz" in cfg.SDC_MODEL_NAME:
        model = uwiz.models.load_model(model_path)
    elif "chauffeur" in cfg.SDC_MODEL_NAME:
        model = tensorflow.keras.models.load_model(
            model_path, custom_objects={"rmse": rmse}
        )
    elif (
            "dave2" in cfg.SDC_MODEL_NAME
            or "epoch" in cfg.SDC_MODEL_NAME
            or "commaai" in cfg.SDC_MODEL_NAME
    ):
        model = tensorflow.keras.models.load_model(model_path)
    else:
        print("sdc_model_type option unknown. Exiting...")
        exit()
    return model


def build_sdc_model(model_type, use_dropout):
    """
    Retrieve the DAVE-2 NVIDIA model
    """
    model = None

    if "uwiz" or "dave2" in model_type:
        if use_dropout:
            """
            Uwiz model w Dropout layers
            """
            model = uwiz.models.StochasticSequential()
            model.add(
                Lambda(
                    lambda x: x / 127.5 - 1.0,
                    input_shape=INPUT_SHAPE,
                    name="lambda_layer",
                )
            )
            model.add(
                Conv2D(
                    24,
                    (5, 5),
                    activation="relu",
                    strides=(2, 2),
                    kernel_regularizer=l2(1.0e-6),
                )
            )
            model.add(Dropout(rate=0.05))
            model.add(
                Conv2D(
                    36,
                    (5, 5),
                    activation="relu",
                    strides=(2, 2),
                    kernel_regularizer=l2(1.0e-6),
                )
            )
            model.add(Dropout(rate=0.05))
            model.add(
                Conv2D(
                    48,
                    (5, 5),
                    activation="relu",
                    strides=(2, 2),
                    kernel_regularizer=l2(1.0e-6),
                )
            )
            model.add(Dropout(rate=0.05))
            model.add(Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2(1.0e-6)))
            model.add(Dropout(rate=0.05))
            model.add(Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2(1.0e-6)))
            model.add(Dropout(rate=0.05))
            model.add(Flatten())
            model.add(Dense(100, activation="relu", kernel_regularizer=l2(1.0e-6)))
            model.add(Dropout(rate=0.05))
            model.add(Dense(50, activation="relu", kernel_regularizer=l2(1.0e-6)))
            model.add(Dropout(rate=0.05))
            model.add(Dense(10, activation="relu", kernel_regularizer=l2(1.0e-6)))
            model.add(Dropout(rate=0.05))
            model.add(Dense(1))
        else:
            """
            original NVIDIA model w/out Dropout layers
            """
            model = Sequential()
            model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
            model.add(Conv2D(24, (5, 5), activation="elu", strides=(2, 2)))
            model.add(Conv2D(36, (5, 5), activation="elu", strides=(2, 2)))
            model.add(Conv2D(48, (5, 5), activation="elu", strides=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation="elu"))
            model.add(Conv2D(64, (3, 3), activation="elu"))
            model.add(Dropout(rate=0.05))
            model.add(Flatten())
            model.add(Dense(100, activation="elu"))
            model.add(Dense(50, activation="elu"))
            model.add(Dense(10, activation="elu"))
            model.add(Dense(1))
    else:
        print("Incorrect model name provided")
        exit()

    assert model is not None
    print(model.inner.summary())

    return model
