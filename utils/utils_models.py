# Copyright 2021 Testing Automated @ Università della Svizzera italiana (USI)
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.
import uncertainty_wizard as uwiz
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda
from tensorflow.keras.regularizers import l2

from utils.utils import INPUT_SHAPE


def build_model(model_name, use_dropout=False):
    """
    Retrieve the DAVE-2 NVIDIA model
    """
    model = None

    if "uwiz" or "dave2" in model_name:
        model = create_model(use_dropout)
    else:
        print("Incorrect model name provided")
        exit()

    assert model is not None
    print(model.inner.summary())

    return model


def create_model(use_dropout):
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
        model.add(
            Conv2D(
                64, (3, 3), activation="relu", kernel_regularizer=l2(1.0e-6)
            )
        )
        model.add(Dropout(rate=0.05))
        model.add(
            Conv2D(
                64, (3, 3), activation="relu", kernel_regularizer=l2(1.0e-6)
            )
        )
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

    return model
