import os
import pprint
import sys
import warnings

import numpy as np

warnings.simplefilter(action="ignore", category=FutureWarning)
sys.path.append("")

from pathlib import Path

import evaluations.time_series.calc_positive_negative as calc_positive_negative
import evaluations.time_series.utils_ts as utils_ts

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
NORMAL_WINDOW_LENGTH, ANOMALY_WINDOW_LENGTH = 30, 30


def window_stack(
    a: np.array, stepsize=NORMAL_WINDOW_LENGTH, width=NORMAL_WINDOW_LENGTH
):
    """
    stackoverflow.com/questions/15722324/sliding-window-of-m-by-n-shape-numpy-ndarray/15722507#15722507
    """
    return np.hstack(
        a[i : 1 + i - width or None : stepsize] for i in range(0, width)
    )


def main():
    root_dir, cfg = utils_ts.load_config()
    data_dir = Path(root_dir, "data")
    sims_path = Path(root_dir, cfg.SIMULATIONS_DIR)

    Path(root_dir, cfg.EVALUATIONS_OUT_DIR).mkdir(
        parents=True, exist_ok=True
    )  # create analysis folder
    utils_ts.create_prec_recall_csv(data_dir)

    sims = utils_ts.collect_simulations(sims_path)
    print(">> Collected simulations: " + str(len(sims)))

    for i, sim in enumerate(sims, start=1):
        sim_path = Path(root_dir, cfg.SIMULATIONS_DIR, sim)
        csv_file = sim_path / "driving_log_normalized.csv"

        print("\n###########################################################")
        print("Analyzing " + str(sim))
        print("----------------------------------------------------------")
        uncertainties = utils_ts.get_uncertainties(
            csv_file
        )  # np array of uncertainties (index is frame_id)
        # get_frame_ids(uncertainties)

        # WINDOWS SPLITTING
        uncertainties_windows = window_stack(uncertainties)
        print(">> Number of Frames: " + str(len(uncertainties)))
        print(">> Windows created: " + str(len(uncertainties_windows)))

        crashes_per_frame = utils_ts.get_crashes(
            csv_file
        )  # dict {frame_id : crash}

        for threshold_type in THRESHOLDS:
            threshold = THRESHOLDS[threshold_type]

            # CALCULATE FP, TP, TN, FN
            (
                windows,
                windows_TP,
                windows_FN,
                windows_FP,
                windows_TN,
                crashes,
            ) = calc_positive_negative._on_windows(
                uncertainties_windows, crashes_per_frame, threshold
            )
            assert (
                windows_TP <= len(windows)
                and windows_FN <= len(windows)
                and windows_FP <= len(windows)
                and windows_TN <= len(windows)
            )

            # CALCULATE PRECISION, RECALL, F1
            (
                precision,
                recall,
                f1,
                fpr,
            ) = calc_positive_negative._calc_precision_recall_f1_fpr(
                windows_TP, windows_FN, windows_FP, windows_TN
            )
            assert (
                float(precision) <= 1 and float(recall) <= 1 and float(f1) <= 1
            )

            utils_ts.write_positive_negative(
                data_dir,
                sim,
                windows,
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

            # print_auroc_timeline(str(db_name))
    print("###########################################################")
    print("\n>> Simulations analyzed: " + str(i))


if __name__ == "__main__":
    main()
