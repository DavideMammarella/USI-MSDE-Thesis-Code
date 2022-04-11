# Copyright 2021 Testing Automated @ Università della Svizzera italiana (USI)
# Code adapted from https://github.com/naokishibuya/car-behavioral-cloning
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

# Standard library import ----------------------------------------------------------------------------------------------
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

# Local libraries import -----------------------------------------------------------------------------------------------
from config import Config
import utils
from utils import rmse, crop, resize

# Tensorflow library import --------------------------------------------------------------------------------------------
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import load_model
from selforacle.vae import VAE, normalize_and_reshape
import uncertainty_wizard as uwiz

# Flask, eventlet, socketio library import -----------------------------------------------------------------------------
import socketio
from flask import Flask
import eventlet.wsgi

eventlet.patcher.monkey_patch()

# Multiprocessing library import ---------------------------------------------------------------------------------------
import subprocess

# Timer class import --------------------------------------------------------------------------------------------------
from threading import Timer
import time

# Server setup ---------------------------------------------------------------------------------------------------------
sio = socketio.Server(async_mode=None, logger=True)
app = Flask(__name__)

# Model setup ----------------------------------------------------------------------------------------------------------
model = None
prev_image_array = None
anomaly_detection = None
autoencoder_model = None
frame_id = 0
batch_size = 128
uncertainty = -1


def start_simulator():  # DO NOT CHANGE THIS
    cmd = "open " + cfg.SIMULATOR_NAME
    subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT)  # subprocess as os.system py doc


@sio.on('connect')  # DO NOT CHANGE THIS
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0, 1, 0, 1, 1)


@sio.on('disconnect')  # DO NOT CHANGE THIS
def disconnect(sid):
    print("disconnect ", sid)
    sio.disconnect(sid)
    os.kill(os.getpid(), signal.SIGTERM)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:

        # Shutdown input to the simulator after certain lapNumber ------------------------------------------------------
        if int(data["lapNumber"]) > cfg.MAX_LAPS:
            sio.emit('shutdown', data={}, skip_sid=True)

        # Data from the simulator --------------------------------------------------------------------------------------
        speed = float(data["speed"])  # current speed of the car
        wayPoint = int(data["currentWayPoint"])  # current waypoint of the car
        lapNumber = int(data["lapNumber"])  # current lap of the car
        cte = float(data["cte"])  # Cross-Track Error: distance from the center of the lane
        brake = float(data["brake"])
        distance = float(data["distance"])  # distance driven by the car
        sim_time = int(data["sim_time"])  # time driven by the car
        ang_diff = float(data["ang_diff"])  # angular difference
        isCrash = int(data["crash"])  # whether an OBE or crash occurred
        number_obe = int(data["tot_obes"])  # total number of crashes so far
        number_crashes = int(data["tot_crashes"])  # total number of OBEs so far
        image = Image.open(BytesIO(base64.b64decode(data["image"])))  # current image from the center camera of the car

        # Save frame ----------------------------------------------------------------------------------------------------
        image_path = ""
        if cfg.TESTING_DATA_DIR != "":
            timestamp = datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
            image_filename = os.path.join(
                cfg.TESTING_DATA_DIR, cfg.SIMULATION_NAME, "IMG", timestamp
            )
            image_path = "{}.jpg".format(image_filename)
            image.save(image_path)

        try:
            global PREDICT_UNC_FLAG
            global steering_angle
            global uncertainty
            global batch_size
            global speed_limit
            global frame_id

            image = np.asarray(image)  # from PIL image to numpy array

            # Calculate loss -------------------------------------------------------------------------------------------
            image_copy = np.copy(image)
            image_copy = resize(image_copy)
            image_copy = normalize_and_reshape(image_copy)
            loss = anomaly_detection.test_on_batch(image_copy)[2]

            # Process image for prediction -----------------------------------------------------------------------------
            image = utils.preprocess(image)  # apply pre-processing
            image = np.array([image])  # model expects 4D array

            # Predict steering angle and uncertainty -------------------------------------------------------------------
            if cfg.USE_PREDICTIVE_UNCERTAINTY:

                if cfg.SDC_MODEL_TYPE == "uwiz" and PREDICT_UNC_FLAG:
                    PREDICT_UNC_FLAG = False
                    outputs, unc = model.predict_quantified(image, quantifier="std_dev", sample_size=20)
                    print("Unc quantified!")
                    steering_angle = outputs[0][0]
                    uncertainty = unc[0][0]
                else:
                    outputs = model.predict(image)
                    steering_angle = outputs[0][0]
            else:
                steering_angle = float(model.predict(image, batch_size=1))

            # Manage driving -------------------------------------------------------------------------------------------
            # lower the throttle as the speed increases
            # if the speed is above the current speed limit: we are on a downhill
            # make sure we slow down first and then go back to the original max speed

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

            # Send control commands to the simulator -------------------------------------------------------------------
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
                        uncertainty,
                        cte,
                        steering_angle,
                        throttle,
                        speed,
                        brake,
                        isCrash,
                        distance,
                        sim_time,
                        ang_diff,
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


def send_control(steering_angle, throttle, confidence, loss, max_laps, uncertainty):  # DO NOT CHANGE THIS
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


def newTimer():
    global PREDICT_UNC_FLAG
    Timer(1.0, newTimer).start()  # predict uncertainty every second
    PREDICT_UNC_FLAG = True


if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile("config_my.py")
    speed_limit = cfg.MAX_SPEED

    start_simulator()

    # Load the self-driving car model ----------------------------------------------------------------------------------
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

    # Load self-assessment oracle model --------------------------------------------------------------------------------
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

    # Create output dir ------------------------------------------------------------------------------------------------
    if cfg.TESTING_DATA_DIR != "":
        utils.create_output_dir(cfg, utils.csv_fieldnames_improved_simulator)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # Deploy server ----------------------------------------------------------------------------------------------------
    newTimer()
    app = socketio.Middleware(sio, app)  # wrap Flask application with engineio's middleware
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)  # deploy as an eventlet WSGI server
