import random
import sys
import warnings
from pprint import pprint

import numpy as np

from evaluations import performance_metrics
from utils import navigate, utils

warnings.simplefilter(action="ignore", category=FutureWarning)

from pathlib import Path

import evaluations.windows_analysis as windows_analysis
import utils.timeseries as utils_ts

THRESHOLDS = {
    "0.68": 0.019586066769424662,
    "0.9": 0.025550906432089442,
    "0.95": 0.028553180589225853,
    "0.99": 0.034772484504920306,
    "0.999": 0.04268394415631666,
    "0.9999": 0.04996533591706328,
    "0.99999": 0.056863799480413445,
}

# REACTION_TIME = 50
NORMAL_WINDOW_LENGTH, ANOMALY_WINDOW_LENGTH = 39, 39
ANALYSIS = 2  # 1: normal (calculate nominal separately), 2: alternative (calculation based on crashes)


def window_stack(
    a: np.array, stepsize=NORMAL_WINDOW_LENGTH, width=NORMAL_WINDOW_LENGTH
):
    """
    stackoverflow.com/questions/15722324/sliding-window-of-m-by-n-shape-numpy-ndarray/15722507#15722507
    """
    return np.hstack(
        a[i : 1 + i - width or None : stepsize] for i in range(0, width)
    )


def perform_analysis_2(uncertainties_windows, crashes_per_frame, threshold):
    # FP in anomale (no nominale) -> prendendo n finestre = n failures -> finestra abbastanza precedente
    # (prendi frame da prima della finestra di crash, da 4 a 6 secondi prima)
    (
        windows,
        windows_TP,
        windows_FN,
        windows_FP,
        windows_TN,
        crashes,
    ) = windows_analysis._on_anomalous_alternative(
        uncertainties_windows, crashes_per_frame, threshold
    )

    return windows_TP, windows_FN, windows_FP, windows_TN, crashes


def perform_analysis_1(
    uncertainties_windows, crashes_per_frame, threshold, windows_nominal
):
    windows_FP, windows_TN, tot_crashes = 0, 0, 0
    (
        windows,
        windows_TP,
        windows_FN,
        tot_crashes
    ) = windows_analysis._on_anomalous(
        uncertainties_windows, crashes_per_frame, threshold
    )
    # vengono tenute FP, TN come variabili fisse, selezionando # window nella nominale pari ai crashes
    for i in range(tot_crashes):
        window = random.choice(windows_nominal)
        print(
            ">> Windows chosen from nominal: " + str(window.get("window_id"))
        )
        windows_FP += window.get("FP")
        windows_TN += window.get("TN")

    assert (windows_FP + windows_TN) == tot_crashes

    return windows_TP, windows_FN, windows_FP, windows_TN, tot_crashes


def main():
    # Load configs and folders -----------------------------------------------------------------------------------------
    cfg = navigate.config()
    data_path = navigate.data_dir()

    sims_path = navigate.simulations_dir()
    sims = navigate.collect_simulations_evaluated(sims_path)
    nominal_sim = navigate.get_nominal_simulation(sims_path)

    print(">> Collected simulations: " + str(len(sims)))

    if ANALYSIS == 1:
        utils_ts.create_prec_recall_csv(data_path / "prec_recall_analysis_1.csv",)
    else:
        utils_ts.create_prec_recall_csv(data_path / "prec_recall_analysis_2.csv",)

    for threshold_type in THRESHOLDS:
        threshold = THRESHOLDS[threshold_type]

        print("\n###########################################################")
        print("THRESHOLD " + str(threshold))
        print("###########################################################")

        if ANALYSIS == 1:
            csv_file = Path(sims_path, nominal_sim, "driving_log_normalized.csv")
            uncertainties = utils_ts.get_uncertainties(
                csv_file
            )  # np array of uncertainties (index is frame_id)

            # WINDOWS SPLITTING
            uncertainties_windows = window_stack(uncertainties)
            crashes_per_frame = utils_ts.get_crashes(
                csv_file
            )  # dict {frame_id : crash}

            # CALCULATE FP, TP, TN, FN
            (
                windows_nominal
            ) = windows_analysis._on_windows_nominal(
                uncertainties_windows, crashes_per_frame, threshold
            )

        for i, sim in enumerate(sims, start=1):
            if sim != nominal_sim:
                csv_file = Path(sims_path, sim, "driving_log_normalized.csv")
                uncertainties = utils_ts.get_uncertainties(
                    csv_file
                )  # np array of uncertainties (index is frame_id)

                print("SIMULATION: " + sim)

                # WINDOWS SPLITTING
                uncertainties_windows = window_stack(uncertainties)
                crashes_per_frame = utils_ts.get_crashes(
                    csv_file
                )  # dict {frame_id : crash}
                print(">> Number of Frames: " + str(len(uncertainties)))
                print(">> Windows created: " + str(len(uncertainties_windows)))

                if ANALYSIS == 1:
                    (
                        windows_TP,
                        windows_FN,
                        windows_FP,
                        windows_TN,
                        crashes,
                    ) = perform_analysis_1(
                        uncertainties_windows,
                        crashes_per_frame,
                        threshold,
                        windows_nominal,
                    )
                else:
                    (
                        windows_TP,
                        windows_FN,
                        windows_FP,
                        windows_TN,
                        crashes,
                    ) = perform_analysis_2(
                        uncertainties_windows, crashes_per_frame, threshold
                    )

                print(
                    "\nWINDOWS SUMMARY:\nTN: "
                    + str(windows_TN)
                    + "\t\tFP: "
                    + str(windows_FP)
                    + "\nFN: "
                    + str(windows_FN)
                    + "\t\tTP: "
                    + str(windows_TP)
                )

                # CALCULATE PRECISION, RECALL, F1
                (
                    precision,
                    recall,
                    f1,
                    fpr,
                ) = performance_metrics._calculate_all(
                    windows_TP, windows_FN, windows_FP, windows_TN
                )
                assert (
                    float(precision) <= 1
                    and float(recall) <= 1
                    and float(f1) <= 1
                )

                if ANALYSIS == 1:
                    utils_ts.write_positive_negative(
                    data_path / "prec_recall_analysis_1.csv",
                    sim,
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
                )
                else:
                    utils_ts.write_positive_negative(
                        data_path / "prec_recall_analysis_2.csv",
                        sim,
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
                    )
                print(
                    "----------------------------------------------------------"
                )
                # print_auroc_timeline(str(db_name))

    print("###########################################################")
    print("\n>> Simulations analyzed: " + str(i))


if __name__ == "__main__":
    main()
