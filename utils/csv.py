# !!! Simulations must be in the project folder under: simulations/<name_of_simulation>
# !!! <name_of_simulation> folder must contain: IMG folder (with .jpg) and driving_log.csv

# This script must be used only with UWIZ models

import sys
import csv
import json
import logging
import os
from pathlib import Path

import numpy
from scipy.stats import gamma

import os
from pathlib import Path


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
