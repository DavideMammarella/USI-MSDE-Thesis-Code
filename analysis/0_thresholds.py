import json
import os
import sys
from pathlib import Path

import numpy
from scipy.stats import gamma

from utils import navigate
from utils.custom_csv import get_column
from utils.navigate import get_nominal_simulation

METRIC_TO_EVAL = "unc"


def calc_and_store_thresholds(metrics: numpy.array, thresholds_location) -> dict:

    print("Fitting reconstruction error distribution using Gamma distribution params")

    shape, loc, scale = gamma.fit(metrics, floc=0)
    thresholds = {}

    conf_intervals = [0.68, 0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999]

    print("Creating thresholds using the confidence intervals: %s" % conf_intervals)

    for c in conf_intervals:
        thresholds[str(c)] = gamma.ppf(c, shape, loc=loc, scale=scale)

    as_json = json.dumps(thresholds)

    json_filename = str(thresholds_location) + "/thresholds_" + METRIC_TO_EVAL + ".json"

    if os.path.exists(json_filename):
        os.remove(json_filename)

    with open(json_filename, "a") as fp:
        fp.write(as_json)

    print(">> Thresholds saved to %s" % json_filename)

    return thresholds


def main():
    sims_path = navigate.simulations_dir()
    output_path = navigate.thresholds_dir()

    nominal_sim = get_nominal_simulation(sims_path)

    try:
        csv_to_analyse = Path(
            sims_path, nominal_sim, ("driving_log_" + METRIC_TO_EVAL + ".csv")
        )
    except:
        print("Error: driving_log_" + METRIC_TO_EVAL + ".csv not found")
        sys.exit(1)

    metrics = get_column(csv_to_analyse, METRIC_TO_EVAL)

    calc_and_store_thresholds(metrics, output_path)


if __name__ == "__main__":
    main()
