import base64
import os
import signal
from datetime import datetime
from io import BytesIO
from pathlib import Path

import eventlet.wsgi
import numpy as np
import socketio
from flask import Flask
from PIL import Image

from client import models
from monitors.selforacle.vae import normalize_and_reshape
from utils import navigate
from utils.augmentation import preprocess, resize
from utils.ultracsv import write_row_simulation_csv

prev_image_array = None
frame_id = 0
batch_size = 1
uncertainty = -1

sio = socketio.Server(async_mode=None, logger=False)
app = Flask(__name__)

cfg = navigate.config()
speed_limit = cfg.MAX_SPEED

sim_path, img_path = navigate.training_simulation_dir()

model_path = str(Path(navigate.models_dir(), cfg.SDC_MODEL_NAME))
model = models.load_sdc_model(cfg, model_path)  # load CAR model

sao_path = str(navigate.sao_dir())
anomaly_detection = models.load_sao_models(cfg, sao_path)  # load AUTOENCODER model


def send_control(
    steering_angle, throttle, confidence, loss, max_laps, uncertainty
):  # DO NOT CHANGE THIS
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


@sio.on("connect")  # DO NOT CHANGE THIS
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0, 1, 0, 1, 1)


@sio.on("disconnect")  # DO NOT CHANGE THIS
def disconnect(sid):
    print("disconnect ", sid)
    sio.disconnect(sid)
    os.kill(os.getpid(), signal.SIGTERM)


@sio.on("telemetry")
def telemetry(sid, data):
    if data:

        # Shutdown input to the server after certain lapNumber ------------------------------------------------------
        if int(data["lapNumber"]) > cfg.MAX_LAPS:
            sio.emit("shutdown", data={}, skip_sid=True)

        # Data from the server --------------------------------------------------------------------------------------
        speed = float(data["speed"])  # current speed of the car
        wayPoint = int(data["currentWayPoint"])  # current waypoint of the car
        lapNumber = int(data["lapNumber"])  # current lap of the car
        cte = float(
            data["cte"]
        )  # Cross-Track Error: distance from the center of the lane
        brake = float(data["brake"])
        distance = float(data["distance"])  # distance driven by the car
        sim_time = int(data["sim_time"])  # time driven by the car
        ang_diff = float(data["ang_diff"])  # angular difference
        isCrash = int(data["crash"])  # whether an OBE or crash occurred
        number_obe = int(data["tot_obes"])  # total number of crashes so far
        number_crashes = int(data["tot_crashes"])  # total number of OBEs so far
        image = Image.open(
            BytesIO(base64.b64decode(data["image"]))
        )  # current image from the center camera of the car

        # Save frame ----------------------------------------------------------------------------------------------------
        image_path = ""
        if cfg.TESTING_DATA_DIR:
            timestamp = datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
            image_filename = Path(img_path, timestamp)
            image_path = "{}.jpg".format(image_filename)
            image.save(image_path)

        try:
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
            image = preprocess(image)  # apply pre-processing
            image = np.array([image])  # model expects 4D array

            # Predict steering angle and uncertainty -------------------------------------------------------------------
            if cfg.USE_PREDICTIVE_UNCERTAINTY:

                if cfg.SDC_MODEL_TYPE == "uwiz":
                    outputs, unc = model.predict_quantified(
                        image,
                        quantifier="std_dev",
                        sample_size=20,
                        batch_size=20,
                    )
                    # print("Unc quantified!")
                    steering_angle = outputs[0][0]
                    uncertainty = unc[0][0]
                else:
                    # ORIGINAL PREDICTION
                    x = np.concatenate(
                        [image for idx in range(batch_size)]
                    )  # take batch of data_nominal
                    outputs = model.predict_on_batch(
                        x
                    )  # save predictions from a sample pass
                    steering_angle = outputs.mean(axis=0)[
                        0
                    ]  # average over all passes is the final steering angle
                    uncertainty = outputs.var(axis=0)[
                        0
                    ]  # variance of predictions gives the uncertainty

                    # outputs = model.predict(image, batch_size = 1)
                    # steering_angle = outputs[0][0]
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

            throttle = 1.0 - steering_angle**2 - (speed / speed_limit) ** 2

            # Send control commands to the server -------------------------------------------------------------------
            send_control(
                steering_angle,
                throttle,
                confidence,
                loss,
                cfg.MAX_LAPS,
                uncertainty,
            )

            if cfg.TESTING_DATA_DIR:
                write_row_simulation_csv(
                    Path(sim_path, "driving_log.csv"),
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
                )

                frame_id = frame_id + 1

        except Exception as e:
            print(e)

    else:
        sio.emit("manual", data={}, skip_sid=True)  # DO NOT CHANGE THIS


# Deploy server ----------------------------------------------------------------------------------------------------
app = socketio.WSGIApp(sio, app)  # wrap Flask application with engineio's middleware
eventlet.wsgi.server(
    eventlet.listen(("", 4567)), app
)  # deploy as an eventlet WSGI server
