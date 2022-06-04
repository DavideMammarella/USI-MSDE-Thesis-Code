from gevent import monkey
monkey.patch_all()

import os
import subprocess
from pathlib import Path

from utils import utils

def start_simulator():  # DO NOT CHANGE THIS
    root_dir, cfg = utils.load_config()
    simulator_path = Path(root_dir, cfg.SIMULATOR_DIR)

    for file in os.listdir(simulator_path):
        if file.endswith(".app") or file.endswith(".exe"):
            simulator_name = file

    cmd = "open " + str(Path(simulator_path, simulator_name))

    subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )  # subprocess as os.system py doc


if __name__ == '__main__':
    start_simulator()
    from selfdrivingcar.drive import app
    app()
