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
from scipy.stats import gamma

import utils.ultracsv
from utils import navigate, utils


def calc_and_store_thresholds(uncertainties: numpy.array, thresholds_location) -> dict:
    """
    Calculates all thresholds stores them on a file system
    :param losses: array of shape (n,),
                    where n is the number of training data points, containing the uncertainties calculated for these points
    :return: a dictionary of where key = threshold_identifier and value = threshold_value
    """
    Path(thresholds_location).mkdir(parents=True, exist_ok=True)

    print(
        "Fitting reconstruction error distribution of UWIZ using Gamma distribution params"
    )

    shape, loc, scale = gamma.fit(uncertainties, floc=0)
    thresholds = {}

    conf_intervals = [0.68, 0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999]

    print("Creating thresholds using the confidence intervals: %s" % conf_intervals)

    for c in conf_intervals:
        thresholds[str(c)] = gamma.ppf(c, shape, loc=loc, scale=scale)

    as_json = json.dumps(thresholds)

    json_filename = str(thresholds_location) + "/uwiz-thresholds.json"

    print("Saving thresholds to %s" % json_filename)

    if os.path.exists(json_filename):
        os.remove(json_filename)

    with open(json_filename, "a") as fp:
        fp.write(as_json)

    return thresholds


def visit_nominal_simulation(sim_path):
    csv_file = sim_path / "driving_log_normalized.csv"
    uncertainties = []

    with open(csv_file, mode="r") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            uncertainties.append(float(row.get("uncertainty")))

    return uncertainties


def get_nominal_simulation(sims_path):
    for sim_path in sims_path.iterdir():
        if (
            sim_path.is_dir()
            and "normal" in str(sim_path).casefold()
            and sim_path.name.endswith("-uncertainty-evaluated")
        ):
            return sim_path.name


def main():
    sims_path = navigate.simulations_dir()
    data_path = navigate.data_dir()
    nominal_sim = get_nominal_simulation(sims_path)
    uncertainties = visit_nominal_simulation(sims_path / nominal_sim)
    calc_and_store_thresholds(uncertainties, data_path)


if __name__ == "__main__":
    main()
