# !!! Simulations must be in the project folder under: simulations/<name_of_simulation>
# !!! <name_of_simulation> folder must contain: IMG folder (with .jpg) and driving_log.csv

# This script must be used only with UWIZ models
import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy
import pandas as pd
from scipy.stats import gamma

header_original_simulator = [
    "center",
    "left",
    "right",
    "steering",
    "throttle",
    "brake",
    "speed",
]
header_improved_simulator = [
    "frame_id",
    "model",
    "anomaly_detector",
    "threshold",
    "sim_name",
    "lap",
    "waypoint",
    "loss",
    "uncertainty",  # newly added
    "cte",
    "steering_angle",
    "throttle",
    "speed",
    "brake",
    "crashed",
    "distance",
    "time",
    "ang_diff",  # newly added
    "center",
    "tot_OBEs",
    "tot_crashes",
]

########################################################################################################################
# SIMULATIONS VISITOR
########################################################################################################################


def get_crashes(csv_file):
    columns_to_read = ["crashed"]
    df = pd.read_csv(csv_file, usecols=columns_to_read)

    return df.to_dict().get("crashed")


def get_column(csv_file, metric):
    if metric == "unc":
        column_to_read = ["uncertainty"]
    elif metric == "loss":
        column_to_read = ["loss"]
    else:
        try:
            column_to_read = [metric]
        except:
            print("Error: metric not found")
            exit()

    df = pd.read_csv(csv_file, usecols=column_to_read)
    driving_log_2d = df.to_numpy()

    assert driving_log_2d.ndim == 2
    assert len(driving_log_2d) == len(df)

    return driving_log_2d


def visit_nominal_simulation(sim_path):
    csv_file = sim_path / "driving_log_normalized.csv"
    uncertainties = []

    with open(csv_file, mode="r") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            uncertainties.append(float(row.get("uncertainty")))

    return uncertainties


def visit_simulation(sim_path):
    csv_file = sim_path / "driving_log.csv"
    print("\nReading simulation:\t", str(sim_path))
    images_dict = []
    with open(csv_file) as f:
        driving_log = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]
    for d in driving_log:
        images_dict.append(
            {
                "frame_id": d.get("frame_id"),
                "center": Path(sim_path, "IMG", d.get("center")),
            }
        )
    f.close()

    return driving_log, images_dict


########################################################################################################################
# SIMULATIONS EVALUATION
########################################################################################################################


def write_driving_log_evaluated(
    sim_path, driving_log, predictions_dict, metric_to_evaluate
):
    final_output = []

    for d in driving_log:
        for prediction in predictions_dict:
            if d.get("frame_id") == prediction.get("frame_id") and d.get(
                "center"
            ) == prediction.get("center"):
                final_output.append(
                    {
                        "frame_id": d.get("frame_id"),
                        "model": d.get("model"),
                        "anomaly_detector": d.get("anomaly_detector"),
                        "threshold": d.get("threshold"),
                        "sim_name": d.get("sim_name"),
                        "lap": d.get("lap"),
                        "waypoint": d.get("waypoint"),
                        "loss": prediction.get("loss"),
                        "uncertainty": prediction.get("uncertainty"),
                        "cte": d.get("cte"),
                        "steering_angle": prediction.get("steering_angle"),
                        "throttle": d.get("throttle"),
                        "speed": d.get("speed"),
                        "brake": d.get("brake"),
                        "crashed": d.get("crashed"),
                        "distance": d.get("distance"),
                        "time": d.get("time"),
                        "ang_diff": d.get("ang_diff"),
                        "center": prediction.get("center"),
                        "tot_OBEs": d.get("tot_obes"),
                        "tot_crashes": d.get("tot_crashes"),
                    }
                )

    csv_file_name = "driving_log_" + metric_to_evaluate + ".csv"
    write_driving_log(final_output, Path(sim_path, csv_file_name))


def write_driving_log(dict, csv_path):
    with csv_path.open(mode="w") as file:
        writer = csv.DictWriter(file, fieldnames=header_improved_simulator)
        writer.writeheader()
        for data in dict:
            writer.writerow(data)

    file.close()


########################################################################################################################
# SIMULATIONS RECORDING
########################################################################################################################


def write_row_simulation_csv(
    simulation_csv,
    frame_id,
    model,
    anomaly_detector,
    threshold,
    sim_name,
    lap,
    waypoint,
    loss,
    uncertainty,
    cte,
    steering_angle,
    throttle,
    speed,
    brake,
    crashed,
    distance,
    time,
    ang_diff,
    center,
    tot_OBEs,
    tot_crashes,
):

    with simulation_csv.open(mode="a") as f:
        f.write(
            str(frame_id)
            + ","
            + str(model)
            + ","
            + str(anomaly_detector)
            + ","
            + str(threshold)
            + ","
            + str(sim_name)
            + ","
            + str(lap)
            + ","
            + str(waypoint)
            + ","
            + str(loss)
            + ","
            + str(uncertainty)
            + ","
            + str(cte)
            + ","
            + str(steering_angle)
            + ","
            + str(throttle)
            + ","
            + str(speed)
            + ","
            + str(brake)
            + ","
            + str(crashed)
            + ","
            + str(distance)
            + ","
            + str(time)
            + ","
            + str(ang_diff)
            + ","
            + str(center.rsplit("/", 1)[-1])
            + ","
            + str(tot_OBEs)
            + ","
            + str(tot_crashes)
            + ","
            + "\n"
        )

    f.close()


def create_simulation_csv(csv_file):
    with csv_file.open(mode="w") as f:
        writer = csv.DictWriter(f, fieldnames=header_improved_simulator)
        writer.writeheader()

    f.close()

########################################################################################################################
# RESULTS VISITOR AND RECORDING
########################################################################################################################

def write_performance_metrics(
    prec_recall_csv,
    sim_name,
    threshold_type,
    threshold,
    windows_TP,
    windows_FN,
    windows_FP,
    windows_TN,
    crashes,
    precision,
    recall,
    f1,
    fpr,
):

    with prec_recall_csv.open(mode="a") as f:
        f.write(
            sim_name
            + ","
            + str(threshold_type)
            + ","
            + str(threshold)
            + ","
            + str(windows_TP)
            + ","
            + str(windows_FP)
            + ","
            + str(windows_TN)
            + ","
            + str(windows_FN)
            + ","
            + str(precision)
            + ","
            + str(recall)
            + ","
            + str(f1)
            + ","
            + str(crashes)
            + ",,,"
            + str(fpr)
            + ","
            + "\n"
        )

    f.close()


def create_performance_metrics_csv(csv_file_normalized):
    with csv_file_normalized.open(mode="w") as f:
        headers = [
            "simulation",
            "threshold_type",
            "threshold",
            "true_positives",
            "false_positives",
            "true_negatives",
            "false_negatives",
            "prec",
            "recall",
            "f1",
            "num_anomalies",
            "num_normal",
            "auroc",
            "false_positive_rate",
            "pr_auc",
        ]  # based on prec-recall DB-schema

        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

    f.close()