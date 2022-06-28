import os
import shutil
from pathlib import Path

from configurations.config import Config

########################################################################################################################
# SIMULATIONS
########################################################################################################################
from utils.custom_csv import create_simulation_csv


def collect_simulations_evaluated(simulations_path):
    sims = []
    for sim_path in simulations_path.iterdir():
        if sim_path.is_dir() and sim_path.name.endswith("-uncertainty-evaluated"):
            sims.append(sim_path.name)

    return sims


def collect_simulations(simulations_path):
    sims = []
    for sim_path in simulations_path.iterdir():
        if sim_path.is_dir() and sim_path.name != "__pycache__":
            sims.append(sim_path.name)
    print(">> Total simulations:\t", len(sims))

    return sims

def collect_simulations_to_evaluate(simulations_path, metric_to_evaluate):
    sims = []
    for sim_path in simulations_path.iterdir():
        if sim_path.is_dir() and sim_path.name.endswith("-uncertainty-evaluated"):
            sims.append(sim_path.name)

def collect_simulations_to_evaluate_loss(simulations_path):
    # First Iteration: collect all simulations -------------------------------------------------------------------------
    _, dirs, _ = next(
        os.walk(simulations_path)
    )  # list all folders in simulations_path (only top level)

    # Second iteration: collect all simulations to exclude -------------------------------------------------------------
    exclude = ["__pycache__"]
    for d in dirs:
        if "-loss-evaluated" in d:
            exclude.append(d)
            exclude.append(d[: -len("-loss-evaluated")])

    sims_evaluated = int(len(exclude) / 2)
    print(">> Total simulations:\t", len(dirs) - sims_evaluated)
    print(">> Simulations already evaluated:\t", sims_evaluated)

    # Third iteration: collect all simulations to evaluate (excluding those already evaluated) -------------------------
    sims = [d for d in dirs if d not in exclude]
    print(">> Simulations to evaluate:\t", len(sims))

    return sims[1]


def collect_simulations_to_evaluate_unc(simulations_path):
    # First Iteration: collect all simulations -------------------------------------------------------------------------
    _, dirs, _ = next(
        os.walk(simulations_path)
    )  # list all folders in simulations_path (only top level)

    # Second iteration: collect all simulations to exclude -------------------------------------------------------------
    exclude = ["__pycache__"]
    for d in dirs:
        if "-uncertainty-evaluated" in d:
            exclude.append(d)
            exclude.append(d[: -len("-uncertainty-evaluated")])

    sims_evaluated = int(len(exclude) / 2)
    print(">> Total simulations:\t", len(dirs) - sims_evaluated)
    print(">> Simulations already evaluated:\t", sims_evaluated)

    # Third iteration: collect all simulations to evaluate (excluding those already evaluated) -------------------------
    sims = [d for d in dirs if d not in exclude]
    print(">> Simulations to evaluate:\t", len(sims))

    return sims


def get_nominal_simulation(simulations_path):
    for sim_path in simulations_path.iterdir():
        if (
            sim_path.is_dir()
            and "normal" in str(sim_path).casefold()
            and sim_path.name.endswith("-uncertainty-evaluated")
        ):
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


def performance_metrics_dir() -> Path:
    p = Path(data_dir(), config().PERFORMANCE_METRICS_DIR)
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


def config():
    cfg = Config()
    cfg.from_pyfile(Path(root_dir(), "configurations", "config_my.py"))
    return cfg
