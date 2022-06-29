import json
import random
import warnings
from pprint import pprint

from utils import navigate, results
from utils.custom_csv import (
    create_performance_metrics_csv,
    get_column,
    get_crashes,
    write_performance_metrics,
)
from utils.navigate import (
    collect_simulations_evaluated,
    get_nominal_simulation,
    read_json,
    results_dir,
    simulations_dir,
)
from utils.results import calculate_performance_metrics
from utils.windows import (
    anomalous_win_analysis,
    anomalous_win_analysis_alt,
    create_windows_stack,
    nominal_win_analysis,
)

warnings.simplefilter(action="ignore", category=FutureWarning)

from pathlib import Path

ANALYSIS = 2  # 1: normal (calculate nominal separately), 2: alternative (calculation based on crashes)
METRIC_TO_EVAL = "loss"  # unc/loss
THRESHOLDS = read_json(
    str(Path(navigate.thresholds_dir(), "thresholds_" + METRIC_TO_EVAL + ".json"))
)


def perform_analysis_2(windows_to_analyse, crashes_per_frame, threshold):
    (
        windows,
        windows_TP,
        windows_FN,
        windows_FP,
        windows_TN,
        crashes,
    ) = anomalous_win_analysis_alt(windows_to_analyse, crashes_per_frame, threshold)

    return windows_TP, windows_FN, windows_FP, windows_TN, crashes


def perform_analysis_1(
    windows_to_analyse, crashes_per_frame, threshold, windows_nominal
):
    windows_FP, windows_TN, tot_crashes = 0, 0, 0
    (windows, windows_TP, windows_FN, tot_crashes) = anomalous_win_analysis(
        windows_to_analyse, crashes_per_frame, threshold
    )
    for i in range(tot_crashes):
        window = random.choice(windows_nominal)
        windows_FP += window.get("FP")
        windows_TN += window.get("TN")
    print(">> Analyzed window (nominal): " + str(i + 1))

    assert (windows_FP + windows_TN) == tot_crashes

    return windows_TP, windows_FN, windows_FP, windows_TN, tot_crashes


def main():
    results_path = results_dir()
    sims_path = simulations_dir()
    sims = collect_simulations_evaluated(sims_path, METRIC_TO_EVAL)
    nominal_sim = get_nominal_simulation(sims_path)

    analysis_output = Path(results_path, METRIC_TO_EVAL)
    analysis_output.mkdir(parents=True, exist_ok=True)

    output_filename = "analysis_" + str(ANALYSIS) + ".csv"

    create_performance_metrics_csv(Path(analysis_output, output_filename))

    for threshold_type in THRESHOLDS:
        threshold = THRESHOLDS[threshold_type]

        print("\n###########################################################")
        print("THRESHOLD " + str(threshold))
        print("###########################################################")

        if ANALYSIS == 1:
            csv_file = Path(
                sims_path, nominal_sim, "driving_log_" + METRIC_TO_EVAL + ".csv"
            )
            metrics = get_column(
                csv_file, METRIC_TO_EVAL
            )  # np array of uncertainties/losses (index is frame_id)

            # WINDOWS SPLITTING
            windows_to_analyse = create_windows_stack(metrics)
            crashes_per_frame = get_crashes(csv_file)  # dict {frame_id : crash}

            # CALCULATE FP, TP, TN, FN
            (windows_nominal) = nominal_win_analysis(
                windows_to_analyse, crashes_per_frame, threshold
            )

        if Path(sims_path, nominal_sim) in sims:
            sims.remove(Path(sims_path, nominal_sim))

        for i, sim in enumerate(sims, start=1):
            csv_file = Path(sim, "driving_log_" + METRIC_TO_EVAL + ".csv")
            metrics = get_column(
                csv_file, METRIC_TO_EVAL
            )  # np array of uncertainties/losses (index is frame_id)

            print("SIMULATION: " + str(sim))

            # WINDOWS SPLITTING
            metrics_windows = create_windows_stack(metrics)
            crashes_per_frame = get_crashes(csv_file)  # dict {frame_id : crash}
            print(">> Number of Frames: " + str(len(metrics)))
            print(">> Windows created: " + str(len(metrics_windows)))

            if ANALYSIS == 1:
                (
                    windows_TP,
                    windows_FN,
                    windows_FP,
                    windows_TN,
                    crashes,
                ) = perform_analysis_1(
                    metrics_windows,
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
                ) = perform_analysis_2(metrics_windows, crashes_per_frame, threshold)

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
            (precision, recall, f1, fpr,) = calculate_performance_metrics(
                windows_TP, windows_FN, windows_FP, windows_TN
            )
            assert float(precision) <= 1 and float(recall) <= 1 and float(f1) <= 1

            write_performance_metrics(
                Path(analysis_output, output_filename),
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
