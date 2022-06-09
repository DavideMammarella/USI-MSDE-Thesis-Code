# !!! Simulations must be in the project folder under: simulations/<name_of_simulation>
# !!! <name_of_simulation> folder must contain: IMG folder (with .jpg) and driving_log.csv

import csv
import os
from pathlib import Path

from tqdm import tqdm

from utils import navigate
from utils import ultracsv as csv_utils


class MyDictReader(csv.DictReader):
    @property
    def fieldnames(self):
        fields = [
            "_".join(field.lower().split())
            for field in super(MyDictReader, self).fieldnames
        ]
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


def check_driving_log(csv_file, csv_file_normalized):
    with open(csv_file) as f:
        driving_log = [
            {k: v for k, v in row.items()}
            for row in MyDictReader(f, skipinitialspace=True)
        ]
    with open(csv_file_normalized) as f_normalized:
        driving_log_to_check = [
            {k: v for k, v in row.items()}
            for row in MyDictReader(f_normalized, skipinitialspace=True)
        ]
    for d in tqdm(driving_log, position=0, leave=False):
        for d_to_check in driving_log_to_check:
            if d.get("frameid") == d_to_check.get("frame_id"):
                assert normalize_img_path(
                    str(d.get("center"))
                ) == d_to_check.get("center")
    f.close()
    f_normalized.close()


def normalize_img_path(img_path):
    normalize_path = img_path.replace("\\", "/")
    img_name = normalize_path.rsplit("/", 1)[-1]
    return "IMG/" + img_name


def normalize_simulation(sim_path):
    csv_file = sim_path / "driving_log.csv"

    with open(csv_file) as f:
        driving_log = [
            {k: v for k, v in row.items()}
            for row in MyDictReader(f, skipinitialspace=True)
        ]

    final_output = []
    for d in driving_log:
        final_output.append(
            {
                "frame_id": d.get("frameid"),
                "model": str(d.get("model", "")).rsplit("/", 1)[-1],
                "anomaly_detector": str(d.get("anomaly_detector", "")).rsplit(
                    "/", 1
                )[-1],
                "threshold": d.get("threshold", ""),
                "sim_name": d.get("sim_name", ""),
                "lap": d.get("lap", ""),
                "waypoint": d.get("waypoint", ""),
                "loss": d.get("loss", ""),
                "uncertainty": d.get("uncertainty", ""),
                "cte": d.get("cte", ""),
                "steering_angle": d.get("steering_angle", ""),
                "throttle": d.get("throttle", ""),
                "speed": d.get("speed", ""),
                "brake": d.get("brake", ""),
                "crashed": d.get("crashed", ""),
                "distance": d.get("distance", ""),
                "time": d.get("time", ""),
                "ang_diff": d.get("ang_diff", ""),
                "center": normalize_img_path(str(d.get("center"))),
                "tot_OBEs": d.get("tot_obes", ""),
                "tot_crashes": d.get("tot_crashes", ""),
            }
        )
    f.close()

    return final_output


def main():
    sims_path = navigate.simulations_dir()
    simulations = navigate.collect_simulations_to_normalize(sims_path)

    # Normalize all simulations ----------------------------------------------------------------------------------------
    for i, sim in enumerate(simulations, start=0):
        sim_path = Path(sims_path, sim)

        print("\nAnalyzing simulation: " + str(sim_path))
        csv_file_normalized = Path(sim_path, "driving_log_normalized.csv")
        if os.path.exists(csv_file_normalized) and os.path.isfile(
            csv_file_normalized
        ):
            os.remove(csv_file_normalized)
            print(">> Normalized CSV file already exists, deleting it..")

        print("Normalizing CSV for simulation: " + str(sim_path))
        dict_to_print = normalize_simulation(sim_path)
        print(">> CSV Normalized!")

        print("Writing CSV for simulation: " + str(sim_path))
        csv_utils.write_driving_log(dict_to_print, sim_path)
        print(">> CSV written!")

        # print("Check CSV integrity (Original vs Normalized)...")
        # check_driving_log(Path(sim_path / "driving_log.csv"), Path(sim_path / "driving_log_normalized.csv"))
        # print(">> Normalized CSV is OK!")

    print(">> Simulations normalized: ", i)


if __name__ == "__main__":
    main()
