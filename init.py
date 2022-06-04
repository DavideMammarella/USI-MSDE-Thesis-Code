from gevent import monkey
monkey.patch_all()

import os
import subprocess
from pathlib import Path

from evaluations.time_series import utils_ts

from config import Config

def start_simulator():  # DO NOT CHANGE THIS
    root_dir, cfg = utils_ts.load_config()

    cfg = Config()
    cfg.from_pyfile("config_my.py")
    simulator_path = Path(root_dir, cfg.SIMULATOR_DIR)
    print(str(simulator_path))

    for file in os.listdir(simulator_path):
        if file.endswith(".app") or file.endswith(".exe"):
            simulator_name = file

    cmd = "open " + str(Path(simulator_path, simulator_name))

    subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )  # subprocess as os.system py doc


if __name__ == '__main__':
    start_simulator()
    from app import app
    app()
