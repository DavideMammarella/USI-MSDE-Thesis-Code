import sys

import numpy as np

sys.path.append("")

import evaluations.time_series.utils_ts as utils_ts
from pathlib import Path

from config import Config

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
ANOMALY_WINDOW_LENGTH = 30
NORMAL_WINDOW_LENGTH = 30


def window_stack(a, stepsize=NORMAL_WINDOW_LENGTH, width=NORMAL_WINDOW_LENGTH):
    """
    stackoverflow.com/questions/15722324/sliding-window-of-m-by-n-shape-numpy-ndarray/15722507#15722507
    """
    return np.hstack(a[i:1 + i - width or None:stepsize] for i in range(0, width))


def main():
    root_dir, cfg = utils_ts.load_config()
    sims_path = Path(root_dir, cfg.SIMULATIONS_DIR)
    Path(root_dir, cfg.EVALUATIONS_OUT_DIR).mkdir(parents=True, exist_ok=True)  # create analysis folder

    sims = utils_ts.collect_simulations(sims_path)
    print(">> Collected simulations: " + str(len(sims)) + "\n")

    for i, sim in enumerate(sims, start=1):
        sim_path = Path(root_dir, cfg.SIMULATIONS_DIR, sim)
        csv_file = sim_path / "driving_log_normalized.csv"

        print("\nAnalyzing " + str(sim))
        uncertainties = utils_ts.driving_log_to_np(csv_file)  # np array of uncertainties (index is frame_id)
        # get_frame_ids(uncertainties)

        # WINDOWS SPLITTING
        uncertainties_windows = window_stack(uncertainties)
        assert utils_ts.windows_check(len(uncertainties), len(uncertainties_windows)), True
        print(">> Created windows: " + str(len(uncertainties_windows)))

        # Calculate True Labels, Precision, Recall, Auroc ------------------------------------------------------
        # set_true_labels(str(db_name))
        # calc_precision_recall(str(db_name))
        # print_auroc_timeline(str(db_name))

    print("\n>> Simulations analyzed: " + str(i))


if __name__ == "__main__":
    main()
