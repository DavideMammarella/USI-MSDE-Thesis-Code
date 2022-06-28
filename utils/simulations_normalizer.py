# !!! Simulations must be in the project folder under: simulations/<name_of_simulation>
# !!! <name_of_simulation> folder must contain: IMG folder (with .jpg) and driving_log.csv

import csv
import os
from pathlib import Path

from tqdm import tqdm

from utils import custom_csv as csv_utils
from utils import navigate


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


def normalize_img_path(img_path):
    normalize_path = img_path.replace("\\", "/")
    img_name = normalize_path.rsplit("/", 1)[-1]
    return img_name


def normalize_simulation(sim_path):
    csv_file = sim_path / "driving_log.csv"

    with open(csv_file) as f:
        driving_log = [
            {k: v for k, v in row.items()}
            for row in MyDictReader(f, skipinitialspace=True)
        ]

    final_output = []
    for i, d in enumerate(driving_log):
        final_output.append(
            {
                "frame_id": str(i),
                "model": str(d.get("model", "")).rsplit("/", 1)[-1],
                "anomaly_detector": str(d.get("anomaly_detector", "")).rsplit("/", 1)[
                    -1
                ],
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
    simulations = navigate.collect_simulations(sims_path)

    # Normalize all simulations ----------------------------------------------------------------------------------------
    for i, sim in enumerate(simulations, start=1):
        sim_path = Path(sims_path, sim)

        print("\nAnalyzing simulation: " + str(sim_path))
        csv_file_normalized = Path(sim_path, "driving_log.csv")

        print("Normalizing CSV for simulation: " + str(sim_path))
        dict_to_print = normalize_simulation(sim_path)
        print(">> CSV Normalized!")

        if os.path.exists(csv_file_normalized) and os.path.isfile(csv_file_normalized):
            os.remove(csv_file_normalized)

        print("Writing CSV for simulation: " + str(sim_path))
        csv_path = Path(sim_path, "driving_log.csv")
        csv_utils.write_driving_log(dict_to_print, csv_path)
        print(">> CSV written!")

    print("\n>> Simulations normalized: ", i)


if __name__ == "__main__":
    main()
