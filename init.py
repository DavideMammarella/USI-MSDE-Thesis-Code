import os
import subprocess

from config import Config
from gevent import monkey
monkey.patch_all() # we need to patch very early

from app import app  # re-export


def start_simulator():  # DO NOT CHANGE THIS
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    for file in os.listdir(cfg.SIMULATOR_DIR):
        if file.endswith(".app") or file.endswith(".exe"):
            simulator_name = file

    cmd = "open " + str(cfg.SIMULATOR_DIR + "/" + simulator_name)
    subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )  # subprocess as os.system py doc


if __name__ == '__main__':
    start_simulator()
    app()
