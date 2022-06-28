import random
import warnings

from utils import navigate, performance_metrics

warnings.simplefilter(action="ignore", category=FutureWarning)

from pathlib import Path

import utils.time_series as utils_ts
import utils.windows as windows_analysis

THRESHOLDS = {
    "0.68": 0.019586066769424662,
    "0.9": 0.025550906432089442,
    "0.95": 0.028553180589225853,
    "0.99": 0.034772484504920306,
    "0.999": 0.04268394415631666,
    "0.9999": 0.04996533591706328,
    "0.99999": 0.056863799480413445,
}

NORMAL_WINDOW_LENGTH, ANOMALY_WINDOW_LENGTH = 39, 39
ANALYSIS = 2  # 1: normal (calculate nominal separately), 2: alternative (calculation based on crashes)


def perform_analysis_2(uncertainties_windows, crashes_per_frame, threshold):
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
    (windows, windows_TP, windows_FN, tot_crashes) = windows_analysis._on_anomalous(
        uncertainties_windows, crashes_per_frame, threshold
    )
    # vengono tenute FP, TN come variabili fisse, selezionando # window nella nominale pari ai crashes
    for i in range(tot_crashes):
        window = random.choice(windows_nominal)
        windows_FP += window.get("FP")
        windows_TN += window.get("TN")
    print(">> Analyzed window (nominal): " + str(i + 1))  # because i start from 0

    assert (windows_FP + windows_TN) == tot_crashes

    return windows_TP, windows_FN, windows_FP, windows_TN, tot_crashes


def main():
    # Load configs and folders -----------------------------------------------------------------------------------------
    cfg = navigate.config()
    data_path = navigate.data_dir()
    metrics_path = navigate.performance_metrics_dir()
    metric_eval = "unc"

    sims_path = navigate.simulations_dir()
    sims = navigate.collect_simulations_evaluated(sims_path, metric_eval)
    nominal_sim = navigate.get_nominal_simulation(sims_path)

    print(">> Collected simulations: " + str(len(sims)))

    prec_recall_filename = "analysis_" + str(ANALYSIS) + ".csv"
    utils_ts.create_prec_recall_csv(Path(metrics_path, prec_recall_filename))

    for threshold_type in THRESHOLDS:
        threshold = THRESHOLDS[threshold_type]

        print("\n###########################################################")
        print("THRESHOLD " + str(threshold))
        print("###########################################################")

        if ANALYSIS == 1:
            csv_file = Path(sims_path, nominal_sim, "driving_log_normalized.csv")
            uncertainties = utils_ts.get_uncertainties(
                csv_file
            )  # np array of white_box (index is frame_id)

            # WINDOWS SPLITTING
            uncertainties_windows = windows_analysis.window_stack(uncertainties)
            crashes_per_frame = utils_ts.get_crashes(
                csv_file
            )  # dict {frame_id : crash}

            # CALCULATE FP, TP, TN, FN
            (windows_nominal) = windows_analysis._on_windows_nominal(
                uncertainties_windows, crashes_per_frame, threshold
            )

        for i, sim in enumerate(sims, start=1):
            if sim != nominal_sim:
                csv_file = Path(sim, "driving_log_normalized.csv")
                uncertainties = utils_ts.get_uncertainties(
                    csv_file
                )  # np array of white_box (index is frame_id)

                print("SIMULATION: " + sim)

                # WINDOWS SPLITTING
                uncertainties_windows = windows_analysis.window_stack(uncertainties)
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
                (precision, recall, f1, fpr,) = performance_metrics._calculate_all(
                    windows_TP, windows_FN, windows_FP, windows_TN
                )
                assert float(precision) <= 1 and float(recall) <= 1 and float(f1) <= 1

                utils_ts.write_positive_negative(
                    Path(metrics_path, prec_recall_filename),
                    str(sim).rsplit("/", 1)[-1],
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
                print("----------------------------------------------------------")
                # print_auroc_timeline(str(db_name))

    print("###########################################################")
    print("\n>> Simulations analyzed: " + str(i))


if __name__ == "__main__":
    main()
