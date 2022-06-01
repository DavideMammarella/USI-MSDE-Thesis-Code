import sys
import os
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.append("")

import evaluations.time_series.utils_ts as utils_ts
import evaluations.time_series.calc_positive_negative as calc_positive_negative
from pathlib import Path

THRESHOLDS = {
    "uwiz": {
        "0.68": 0.019586066769424662,
        "0.9": 0.025550906432089442,
        "0.95": 0.028553180589225853,
        "0.99": 0.034772484504920306,
        "0.999": 0.04268394415631666,
        "0.9999": 0.04996533591706328,
        "0.99999": 0.056863799480413445,
    }
}

# REACTION_TIME = 50
NORMAL_WINDOW_LENGTH, ANOMALY_WINDOW_LENGTH = 30, 30



def window_stack(a: np.array, stepsize=NORMAL_WINDOW_LENGTH, width=NORMAL_WINDOW_LENGTH):
    """
    stackoverflow.com/questions/15722324/sliding-window-of-m-by-n-shape-numpy-ndarray/15722507#15722507
    """
    return np.hstack(a[i:1 + i - width or None:stepsize] for i in range(0, width))


def main():
    root_dir, cfg = utils_ts.load_config()
    sims_path = Path(root_dir, cfg.SIMULATIONS_DIR)
    Path(root_dir, cfg.EVALUATIONS_OUT_DIR).mkdir(parents=True, exist_ok=True)  # create analysis folder

    sims = utils_ts.collect_simulations(sims_path)
    print(">> Collected simulations: " + str(len(sims)))

    for i, sim in enumerate(sims, start=1):
        sim_path = Path(root_dir, cfg.SIMULATIONS_DIR, sim)
        csv_file = sim_path / "driving_log_normalized.csv"

        print("\n###########################################################")
        print("Analyzing " + str(sim))
        print("----------------------------------------------------------")
        uncertainties = utils_ts.driving_log_to_np(csv_file)  # np array of uncertainties (index is frame_id)
        # get_frame_ids(uncertainties)

        # WINDOWS SPLITTING
        uncertainties_windows = window_stack(uncertainties)
        assert utils_ts.windows_check(len(uncertainties), len(uncertainties_windows)), True
        print(">> Windows created: " + str(len(uncertainties_windows)))

        # OTTIENI IL FRAME_ID DELLA WINDOW FACENDO WINDOW_ID/WINDOW_LENGTH SSE WINDOW_ID > WINDOW_LENGTH
        # TODO: in caso di step?
        calc_positive_negative._on_nominal(uncertainties_windows, 0.019586066769424662)


        # Calculate True Labels, Precision, Recall, Auroc ------------------------------------------------------
        # set_true_labels(str(db_name))
        # calc_precision_recall(str(db_name))
        # print_auroc_timeline(str(db_name))
    print("###########################################################")
    print("\n>> Simulations analyzed: " + str(i))


if __name__ == "__main__":
    main()
