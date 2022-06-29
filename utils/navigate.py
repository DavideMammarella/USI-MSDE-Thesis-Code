import json
import os
import shutil
from pathlib import Path
from pprint import pprint

from configurations.config import Config

########################################################################################################################
# SIMULATIONS
########################################################################################################################
from utils.custom_csv import create_simulation_csv


def collect_simulations(simulations_path):
    sims = []
    for sim_path in simulations_path.iterdir():
        if sim_path.is_dir() and sim_path.name != "__pycache__":
            sims.append(sim_path.name)
    print(">> Total simulations:\t", len(sims))
    return sims


def collect_simulations_to_evaluate(simulations_path, metric_to_evaluate):
    sims = collect_simulations(simulations_path)
    sims_to_evaluate = []

    for sim in sims:
        sim_path = Path(simulations_path, sim)
        files = [
            f
            for f in os.listdir(sim_path)
            if os.path.isfile(os.path.join(sim_path, f)) and f.endswith(".csv")
        ]
        if not any(metric_to_evaluate in f for f in files):
            sims_to_evaluate.append(sim_path)

    print(">> Simulations already evaluated (" + metric_to_evaluate + "):\t", len(sims) - len(sims_to_evaluate))
    return sims_to_evaluate


def collect_simulations_evaluated(simulations_path, metric_evaluated):
    sims = collect_simulations(simulations_path)
    sims_evaluated = []

    for sim in sims:
        sim_path = Path(simulations_path, sim)
        files = [
            f
            for f in os.listdir(sim_path)
            if os.path.isfile(os.path.join(sim_path, f)) and f.endswith(".csv")
        ]
        if any(metric_evaluated in f for f in files):
            sims_evaluated.append(sim_path)

    print(">> Simulations monitored (" + metric_evaluated + "):\t", len(sims_evaluated))
    return sims_evaluated


def get_nominal_simulation(simulations_path):
    for sim_path in simulations_path.iterdir():
        if sim_path.is_dir() and "normal" in str(sim_path).casefold():
            return sim_path.name


########################################################################################################################
# PATHS
########################################################################################################################


def delete_dir(dir_path):
    if dir_path.exists() and dir_path.is_dir():
        print("Deleting folder at {}".format(dir_path))
        shutil.rmtree(dir_path)


def root_dir() -> Path:
    return Path(__file__).parent.parent


# Edit the @configurations/config_my.py file to change following folder names


def data_dir() -> Path:
    p = Path(root_dir(), config().DATA_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def training_data_dir() -> Path:
    p = Path(data_dir(), config().TRAINING_DATA_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def simulations_dir() -> Path:
    p = Path(data_dir(), config().SIMULATIONS_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir() -> Path:
    p = Path(data_dir(), config().RESULTS_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = Path(root_dir(), config().SDC_MODELS_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sao_dir() -> Path:
    p = Path(models_dir(), config().SAO_MODELS_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def training_set_dir() -> Path:
    p = Path(data_dir(), config().TRAINING_DATA_DIR, config().TRAINING_SET_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def simulator_dir() -> Path:
    p = Path(root_dir(), config().SIMULATOR_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def thresholds_dir() -> Path:
    p = Path(data_dir(), "thresholds")
    p.mkdir(parents=True, exist_ok=True)
    return p


def training_simulation_dir() -> Path:
    simulation_path, img_path = None, None
    if config().TESTING_DATA_DIR:
        simulation_name = str(config().SDC_MODEL_NAME + "-" + config().SIMULATION_NAME)
        simulation_path = Path(simulations_dir(), simulation_name)
        img_path = Path(simulation_path, "IMG")
        delete_dir(img_path)
        img_path.mkdir(parents=True, exist_ok=True)
        create_simulation_csv(Path(simulation_path, "driving_log.csv"))
        print("RECORDING THIS RUN ...")

    else:
        print("NOT RECORDING THIS RUN ...")
    return simulation_path, img_path


########################################################################################################################
# FILES
########################################################################################################################


def read_json(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)


def config():
    cfg = Config()
    cfg.from_pyfile(Path(root_dir(), "configurations", "config_my.py"))
    return cfg
