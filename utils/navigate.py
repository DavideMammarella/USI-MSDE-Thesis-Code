import os
from pathlib import Path

from configurations.config import Config

########################################################################################################################
# SIMULATIONS
########################################################################################################################


def collect_simulations_evaluated(simulations_path):
    sims = []
    for sim_path in simulations_path.iterdir():
        if sim_path.is_dir() and sim_path.name.endswith("-uncertainty-evaluated"):
            sims.append(sim_path.name)

    return sims


def collect_simulations_to_normalize(simulations_path):
    sims = []
    for sim_path in simulations_path.iterdir():
        if sim_path.is_dir() and sim_path.name.endswith("-uncertainty-evaluated"):
            sims.append(str(sim_path.name).replace("-uncertainty-evaluated", ""))

    return sims


def collect_simulations_to_evaluate(simulations_path):
    # First Iteration: collect all simulations -------------------------------------------------------------------------
    _, dirs, _ = next(
        os.walk(simulations_path)
    )  # list all folders in simulations_path (only top level)

    # Second iteration: collect all simulations to exclude -------------------------------------------------------------
    exclude = []
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
    p = Path(data_dir(), config().TRAINING_SET_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def simulator_dir() -> Path:
    p = Path(root_dir(), config().SIMULATOR_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


########################################################################################################################
# FILES
########################################################################################################################


def config():
    cfg = Config()
    cfg.from_pyfile(Path(root_dir(), "configurations", "config_my.py"))
    return cfg
