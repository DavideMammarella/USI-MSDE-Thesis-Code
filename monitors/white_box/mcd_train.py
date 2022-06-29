import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from utils.models_train import load_training_data
from utils.sdc import *
from utils.sdc_batch_generator import Generator

np.random.seed(0)


def train_sdc_model(model_type, model, cfg, x_train, x_test, y_train, y_test):
    models_path = navigate.models_dir()
    model_name = cfg.TRACK + "-" + model_type + "-{epoch:03d}"
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
        model_name = cfg.TRACK + "-" + model_type + "-final"
        model_path = Path(models_path, model_name)

    # save the last model anyway (might not be the best)
    model.save(str(model_path))


def main():
    """
    Load train/validation data_nominal set and train the model
    """
    cfg = navigate.config()

    model_type = "uwiz"

    x_train, x_test, y_train, y_test = load_training_data(model_type)

    model = build_sdc_model(model_type, cfg.USE_PREDICTIVE_UNCERTAINTY)

    train_sdc_model(model_type, model, cfg, x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
