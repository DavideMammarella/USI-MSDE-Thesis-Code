# !!! Works only on CSV files

import sys
sys.path.append("..")

import os
import pathlib
from pathlib import Path

import logging
import csv
import numpy

import utils_logging
from eval_scripts.a_set_true_labels import set_true_labels
from eval_scripts.b_precision_recall_auroc import calc_precision_recall
from eval_scripts.c_auroc_timeline import print_auroc_timeline
from eval_db.database import Database
from eval_db.eval_setting import Setting
from eval_db.eval_single_img_uncertainty import SingleImgUncertainty

SINGLE_IMAGE_ADS = ["UWIZ"]

logger = logging.Logger("main")
utils_logging.log_info(logger)

EVAL_AGENTS = ["xai", "DAVE2"]
EVAL_TRACKS = ["Track1"]

# TODO Change this every time you want to merge generated tables to compatible start ids
SETTING_START_ID = 3000
EVAL_TIME = ["DayNight", "DayOnly"]
EVAL_WEATHER = ["Fog", "Rain", "Snow", "Sunny"]


def main():
    curr_project_path = Path(os.path.normpath(os.getcwd() + os.sep + os.pardir))  # overcome OS issues
    sims = collect_simulations(curr_project_path)

    for i, sim in enumerate(sims, start=1):
        # Database creation (1 DB for each simulation) -----------------------------------------------------------------
        Path(curr_project_path, "databases").mkdir(parents=True, exist_ok=True)  # create DB folder if not exists
        db_name = "../databases/" + sim + "-based-eval.sqlite"
        db = Database(db_name, True) # create DB
        # TODO: fix on settings (to be asked, settings from folders?, we analyze Track1 under different conditions)
        settings = _create_all_settings(db)

        # Extract uncertainties (from CSV to DB) -----------------------------------------------------------------------
        store_uncertainties(db, curr_project_path, sim)

        # Calculate True Labels, Precision, Recall, Auroc --------------------------------------------------------------
        set_true_labels(db_name)
        calc_precision_recall(db_name)
        print_auroc_timeline(db_name)

    logger.info("Simulations analyzed: " + str(i))


def store_uncertainties(db, curr_project_path, sim):
    sim_path = Path(curr_project_path, "simulations", sim) #forse
    csv_file = sim_path / "driving_log.csv"

    logger.info("Analyzing " + sim)

    with open(csv_file, mode='r') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for i, row in enumerate(reader, start=1):
            # setting_id = setting.id # TODO: fix on settings (to be asked)
            frame_id, uncertainty, is_crash, x = row["frameId"], row["uncertainty"], bool(int(row["crashed"])), row["center"]
            to_store = SingleImgUncertainty(setting_id=3000, row_id=frame_id, is_crash=is_crash, uncertainty=uncertainty)
            to_store.insert_into_db(db)
            if i % 1000:
                db.commit()
            db.commit()

    logger.info("Rows extracted: " + str(i) + "\n")

def _create_all_settings(db: Database):
    settings = []
    id = SETTING_START_ID
    for agent in EVAL_AGENTS:
        for track in EVAL_TRACKS:
            for time in EVAL_TIME:
                for weather in EVAL_WEATHER:
                    setting = Setting(id=id, agent=agent, track=track, time=time, weather=weather)
                    setting.insert_into_db(db=db)
                    id = id + 1
                    settings.append(setting)
    db.commit()
    return settings

def collect_simulations(curr_project_path):
    """
    Visit all simulation folders and collect only names that contain ("-uncertainty-evaluated").
    :return: list of simulations
    """
    sims_path = Path(curr_project_path, "simulations")
    sims = []

    for sim_path in sims_path.iterdir():
        if sim_path.is_dir() and sim_path.name.endswith("-uncertainty-evaluated"):
            sims.append(sim_path.name)
    logger.info("Collected simulations: " + str(len(sims)) + "\n")

    return sims

if __name__ == '__main__':
    main()