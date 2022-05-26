# !!! Simulations must be in the project folder under: simulations/<name_of_simulation>
# !!! <name_of_simulation> folder must contain: IMG folder (with .jpg) and driving_log.csv

import sys

sys.path.append("..")

from tqdm import tqdm
import os
import pathlib
from pathlib import Path
import csv
from config import Config


class MyDictReader(csv.DictReader):

    @property
    def fieldnames(self):
        fields = ["_".join(field.lower().split()) for field in super(MyDictReader, self).fieldnames]
        for i in range(len(fields)):
            if fields[i].startswith("self_driving"):
                fields[i] = "model"
            if fields[i].startswith("track_name"):
                fields[i] = "sim_name"
            if fields[i].startswith("lap_number"):
                fields[i] = "lap"
            if fields[i].startswith("check_point"):
                fields[i] = "waypoint"
        return fields


def check_driving_log(sim_path):
    csv_file = sim_path / "driving_log.csv"
    csv_file_normalized = sim_path / "driving_log_normalized.csv"

    with open(csv_file) as f:
        driving_log = [{k: v for k, v in row.items()}
                       for row in MyDictReader(f, skipinitialspace=True)]
    with open(csv_file_normalized) as f_normalized:
        driving_log_to_check = [{k: v for k, v in row.items()}
                                for row in MyDictReader(f_normalized, skipinitialspace=True)]
    print("Check CSV robustness...")
    for d in tqdm(driving_log):
        for d_to_check in driving_log_to_check:
            if d.get("frameid") == d_to_check.get("frame_id"):
                assert normalize_img_path(str(d.get("center")), sim_path) == d_to_check.get("center")
    f.close()
    f_normalized.close()

def write_driving_log(dict, sim_path):
    csv_file_normalized = sim_path / "driving_log_normalized.csv"

    print("Writing CSV for simulation: " + str(sim_path))
    with csv_file_normalized.open(mode="w") as f_normalized:
        headers = ["frame_id", "model", "anomaly_detector", "threshold", "sim_name", "lap", "waypoint", "loss",
                   "uncertainty", "cte", "steering_angle", "throttle", "speed", "brake", "crashed", "distance", "time",
                   "ang_diff", "center", "tot_OBEs", "tot_crashes"]
        writer = csv.DictWriter(f_normalized, fieldnames=headers)
        writer.writeheader()
        for data in tqdm(dict):
            writer.writerow(data)

    f_normalized.close()


def normalize_img_path(img_path, sim_path):
    normalize_path = img_path.replace("\\", "/")
    img_name = normalize_path.rsplit("/", 1)[-1]
    sim_relative_path = "/".join(str(sim_path).rsplit("/", 3)[-2:])
    return sim_relative_path + "/" + img_name


def normalize_simulation(sim_path):
    csv_file = sim_path / "driving_log.csv"

    with open(csv_file) as f:
        driving_log = [{k: v for k, v in row.items()}
                       for row in MyDictReader(f, skipinitialspace=True)]

    final_output = []
    print("Normalizing CSV for simulation: " + str(sim_path))
    for d in driving_log:
        final_output.append(
            {'frame_id': d.get("frameid"),
             'model': str(d.get("model", "")).rsplit("/", 1)[-1],
             'anomaly_detector': str(d.get("anomaly_detector", "")).rsplit("/", 1)[-1],
             'threshold': d.get("threshold", ""),
             'sim_name': d.get("sim_name", ""),
             'lap': d.get("lap", ""),
             'waypoint': d.get("waypoint", ""),
             'loss': d.get("loss", ""),
             'uncertainty': d.get("uncertainty", ""),
             'cte': d.get("cte", ""),
             'steering_angle': d.get("steering_angle", ""),
             'throttle': d.get("throttle", ""),
             'speed': d.get("speed", ""),
             'brake': d.get("brake", ""),
             'crashed': d.get("crashed", ""),
             'distance': d.get("distance", ""),
             'time': d.get("time", ""),
             'ang_diff': d.get("ang_diff", ""),
             'center': normalize_img_path(str(d.get("center")), sim_path),
             'tot_OBEs': d.get("tot_obes", ""),
             'tot_crashes': d.get("tot_crashes", "")
             })

    f.close()

    return final_output


def collect_simulations(sims_path):
    _, dirs, _ = next(os.walk(sims_path))  # list all folders in simulations_path (only top level)
    sims = [d for d in dirs]  # collect all simulations name
    print("Summary..\n>> Simulations to evaluate:\t", len(sims))
    return sims


def main():
    curr_project_path = Path(os.path.normpath(os.getcwd() + os.sep + os.pardir))  # overcome OS issues

    cfg = Config()
    cfg_pyfile_path = curr_project_path / "config_my.py"
    cfg.from_pyfile(cfg_pyfile_path)

    # Analyse all simulations ------------------------------------------------------------------------------------------
    sims_path = Path(curr_project_path, cfg.SIMULATIONS_DIR)
    simulations = collect_simulations(sims_path)

    # Normalize all simulations ----------------------------------------------------------------------------------------
    for i, sim in enumerate(simulations):
        sim_path = Path(curr_project_path, cfg.SIMULATIONS_DIR, sim)
        print("\nAnalyzing simulation: " + str(sim_path))
        csv_file_normalized = sim_path / "driving_log_normalized.csv"
        if os.path.exists(csv_file_normalized) and os.path.isfile(csv_file_normalized):
            os.remove(csv_file_normalized)
            print("CSV normalized file already exists, deleting it..")
        dict_to_print = normalize_simulation(sim_path)
        write_driving_log(dict_to_print, sim_path)
        check_driving_log(sim_path)

    print(">> Simulations evaluated: ", i)

if __name__ == "__main__":
    main()
