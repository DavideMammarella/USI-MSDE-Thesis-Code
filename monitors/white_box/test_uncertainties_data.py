# !!! Simulations must be in the project folder under: simulations/<name_of_simulation>
# !!! <name_of_simulation> folder must contain: IMG folder (with .jpg) and driving_log.csv
# ALL SIMULATIONS MUST BE NORMALIZED using simulations_normalizer.py

# This script must be used only with UWIZ models

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings

import csv
import statistics
from pathlib import Path

from utils import navigate


def get_avg_unc(sim_path):
    csv_file = sim_path / "driving_log_normalized.csv"

    uncertainties = []
    with open(csv_file) as f:
        driving_log_normalized = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]

    for d in driving_log_normalized:
        uncertainties.append(float(d.get("uncertainty")))
    f.close()

    avg_unc = statistics.mean(uncertainties)

    return avg_unc


def main():
    cfg = navigate.config()
    sims_path = navigate.simulations_dir()
    simulations = navigate.collect_simulations_evaluated(sims_path)

    print(
        "{:<30} {:<15}\n----------------------------------------------------".format(
            "Model", "Average Uncertainty"
        )
    )
    avgs = []
    for sim in simulations:
        sim_path = Path(sims_path, sim)
        avg_unc = get_avg_unc(sim_path)
        avgs.append(avg_unc)
        print(
            "{:<30} {:<15}".format(
                str(sim).replace("-uncertainty-evaluated", ""), avg_unc
            )
        )
    print("----------------------------------------------------\n")

    avgs_sorted = sorted(avgs)
    nominal_unc = avgs_sorted[0]
    min_unc = avgs_sorted[1]  # second min, because first is nominal
    max_unc = avgs_sorted[-1]
    avg_unc = statistics.mean(avgs_sorted)
    print(
        ">> Nominal Unc: {} \n>> Min Unc: {} \n>> Max Unc: {} \n>> Avg Unc: {}".format(
            nominal_unc, min_unc, max_unc, avg_unc
        )
    )


if __name__ == "__main__":
    main()
