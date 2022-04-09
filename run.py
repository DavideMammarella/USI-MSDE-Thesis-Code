# Copyright 2021 Testing Automated @ Università della Svizzera italiana (USI)
# Code adapted from https://github.com/naokishibuya/car-behavioral-cloning
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

# Standard library import
import os
import sys
from sys import exit
from warnings import simplefilter
from datetime import datetime
from pathlib import Path
from PIL import Image
import sched, time
import numpy as np
import logging
import signal
import base64
from io import BytesIO
import run

# Local libraries import
from config import Config
import utils
from utils import rmse, crop, resize

# Tensorflow library import
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import load_model
from selforacle.vae import VAE, normalize_and_reshape
import uncertainty_wizard as uwiz

# Flask, eventlet, socketio library import
import socketio
from flask import Flask
import eventlet.wsgi
eventlet.patcher.monkey_patch()

# Multithreading library import
import threading
import subprocess

# None chose the best option (Threading, Eventlet, Gevent) based on installed packages
sio = socketio.Server(async_mode=None, logger=True)
app = Flask(__name__)

model = None
prev_image_array = None
anomaly_detection = None
autoencoder_model = None
frame_id = 0
batch_size = 120
uncertainty = -1


def quantify_uncertainty(image):
    outputs, unc = model.predict_quantified(image, quantifier="std_dev", sample_size=15)
    steering_angle = outputs[0][0]
    uncertainty = unc[0][0]
    print("Unc quantified!")
    return steering_angle, uncertainty


@sio.on('telemetry')
def telemetry(sid, data):
    if data:

        if int(data["lapNumber"]) > cfg.MAX_LAPS:
            sio.emit('shutdown', data={}, skip_sid=True)

        # The current speed of the car
        speed = float(data["speed"])

        # the current way point and lap
        wayPoint = int(data["currentWayPoint"])
        lapNumber = int(data["lapNumber"])

        # Cross-Track Error: distance from the center of the lane
        cte = float(data["cte"])

        # brake
        brake = float(data["brake"])
        # print("brake: %.2f" % brake)

        # the distance driven by the car
        distance = float(data["distance"])

        # the time driven by the car
        sim_time = int(data["sim_time"])
        # print(sim_time)

        # the angular difference
        ang_diff = float(data["ang_diff"])

        # whether an OBE or crash occurred
        isCrash = int(data["crash"])

        # the total number of OBEs and crashes so far
        number_obe = int(data["tot_obes"])
        number_crashes = int(data["tot_crashes"])

        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        # save frame
        image_path = ""
        if cfg.TESTING_DATA_DIR != "":
            timestamp = datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
            image_filename = os.path.join(
                cfg.TESTING_DATA_DIR, cfg.SIMULATION_NAME, "IMG", timestamp
            )
            image_path = "{}.jpg".format(image_filename)
            image.save(image_path)

        try:
            # from PIL image to numpy array
            image = np.asarray(image)

            # get the loss
            image_copy = np.copy(image)
            image_copy = resize(image_copy)
            image_copy = normalize_and_reshape(image_copy)
            loss = anomaly_detection.test_on_batch(image_copy)[2]

            # apply the pre-processing
            image = utils.preprocess(image)

            # the model expects 4D array
            image = np.array([image])

            global steering_angle
            global uncertainty
            global batch_size

            if cfg.USE_PREDICTIVE_UNCERTAINTY:

                if cfg.USE_UWIZ:
                    quantify_uncertainty(image)
                else:
                    # save predictions from every image
                    outputs = model.predict(image)
                    steering_angle = outputs[0][0]

            else:
                steering_angle = float(model.predict(image, batch_size=1))

            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit

            if speed > speed_limit:
                speed_limit = cfg.MIN_SPEED  # slow down
            else:
                speed_limit = cfg.MAX_SPEED

            if loss > cfg.SAO_THRESHOLD * 1.1:
                confidence = -1
            elif cfg.SAO_THRESHOLD < loss <= cfg.SAO_THRESHOLD * 1.1:
                confidence = 0
            else:
                confidence = 1

            throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2

            global frame_id

            send_control(
                steering_angle, throttle, confidence, loss, cfg.MAX_LAPS, uncertainty
            )

            if cfg.TESTING_DATA_DIR:
                csv_path = os.path.join(cfg.TESTING_DATA_DIR, cfg.SIMULATION_NAME)
                utils.write_csv_line(
                    csv_path,
                    [
                        frame_id,
                        cfg.SDC_MODEL_NAME,
                        cfg.ANOMALY_DETECTOR_NAME,
                        cfg.SAO_THRESHOLD,
                        cfg.SIMULATION_NAME,
                        lapNumber,
                        wayPoint,
                        loss,
                        uncertainty,  # new metrics
                        cte,
                        steering_angle,
                        throttle,
                        speed,
                        brake,
                        isCrash,
                        distance,
                        sim_time,
                        ang_diff,  # new metrics
                        image_path,
                        number_obe,
                        number_crashes,
                    ],
                )

                frame_id = frame_id + 1

        except Exception as e:
            print(e)

    else:
        sio.emit("manual", data={}, skip_sid=True)  # DO NOT CHANGE THIS


@sio.on('connect')  # DO NOT CHANGE THIS
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0, 1, 0, 1, 1)

@sio.on('disconnect')
def disconnect(sid):
    print("disconnect ", sid)
    sio.disconnect(sid)
    os.kill(os.getpid(), signal.SIGTERM)

# DO NOT CHANGE THIS
def send_control(steering_angle, throttle, confidence, loss, max_laps, uncertainty):
    sio.emit(
        "steer",
        data={
            "steering_angle": steering_angle.__str__(),
            "throttle": throttle.__str__(),
            "confidence": confidence.__str__(),
            "loss": loss.__str__(),
            "max_laps": max_laps.__str__(),
            "uncertainty": uncertainty.__str__(),
        },
        skip_sid=True,
    )

def start_simulator():
    cmd = "open " + cfg.SIMULATOR_NAME
    subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)  # subprocess as os.system py doc


if __name__ == '__main__':

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    start_simulator()

    speed_limit = cfg.MAX_SPEED

    # load the self-driving car model
    model_path = os.path.join(cfg.SDC_MODELS_DIR, cfg.SDC_MODEL_NAME)

    if cfg.SDC_MODEL_TYPE == "uwiz":
        model = uwiz.models.load_model(model_path)
    elif cfg.SDC_MODEL_TYPE == "chauffeur":
        model = tensorflow.keras.models.load_model(model_path, custom_objects={"rmse": rmse})
    elif cfg.SDC_MODEL_TYPE == "dave2" or cfg.SDC_MODEL_TYPE == "epoch" or cfg.SDC_MODEL_TYPE == "commaai":
        model = tensorflow.keras.models.load_model(model_path)
    else:
        print("cfg.SDC_MODEL_TYPE option unknown. Exiting...")
        exit()

    # load the self-assessment oracle model
    encoder = tensorflow.keras.models.load_model(
        cfg.SAO_MODELS_DIR + os.path.sep + "encoder-" + cfg.ANOMALY_DETECTOR_NAME
    )
    decoder = tensorflow.keras.models.load_model(
        cfg.SAO_MODELS_DIR + os.path.sep + "decoder-" + cfg.ANOMALY_DETECTOR_NAME
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

    # create the output dir
    if cfg.TESTING_DATA_DIR != "":
        utils.create_output_dir(cfg, utils.csv_fieldnames_improved_simulator)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    server = eventlet.wsgi.server(eventlet.listen(("", 4567)), app)