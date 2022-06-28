from gevent import monkey

monkey.patch_all()

import os
import subprocess
from pathlib import Path

from utils import navigate


def start_simulator():  # DO NOT CHANGE THIS
    cfg = navigate.config()
    simulator_path = navigate.simulator_dir()

    for file in os.listdir(simulator_path):
        if file.endswith(".app") or file.endswith(".exe"):
            simulator_name = file

    cmd = "open " + str(Path(simulator_path, simulator_name))

    subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )  # subprocess as os.system py doc


if __name__ == "__main__":
    start_simulator()
    from client.sdc_drive import app

    app()
