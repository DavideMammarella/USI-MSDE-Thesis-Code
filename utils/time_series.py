import csv

import numpy as np
import pandas as pd

NORMAL_WINDOW_LENGTH, ANOMALY_WINDOW_LENGTH = 39, 39


def get_frame_ids(np_array):
    print("{}, {}".format("frame_id", "uncertainty"))
    for index, val in np.ndenumerate(np_array):
        print("{}, {}".format(index[0], val))


###############################################################################
# CSV MANAGING
###############################################################################


def write_positive_negative(
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
    # TODO: window can be used to add additional information on windows inside csv

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


def create_prec_recall_csv(csv_file_normalized):
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


def get_crashes(csv_file):
    columns_to_read = ["crashed"]
    df = pd.read_csv(csv_file, usecols=columns_to_read)

    return df.to_dict().get("crashed")


def get_uncertainties(csv_file):
    columns_to_read = ["uncertainty"]
    df = pd.read_csv(csv_file, usecols=columns_to_read)
    driving_log_2d = df.to_numpy()

    assert driving_log_2d.ndim == 2
    assert len(driving_log_2d) == len(df)

    return driving_log_2d


###############################################################################
# CHECKS
###############################################################################


def windows_check(len_uncertainties, len_uncertainties_windows):
    actual_len = len_uncertainties_windows
    expected_len = len_uncertainties / NORMAL_WINDOW_LENGTH
    return int(expected_len) == int(actual_len)
