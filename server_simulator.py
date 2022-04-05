import logging

from config import Config
from flask import Flask
import socketio
import eventlet.wsgi
eventlet.monkey_patch()
import subprocess
import os

# None chose the best option (Threading, Eventlet, Gevent) based on installed packages
async_mode = None
sio = socketio.Server(async_mode=async_mode, logger=True)
app = Flask(__name__)

def log_subprocess_output(pipe):
    for line in iter(pipe.readline, b''): # b'\n'-separated lines
        logging.info('got line from subprocess: %r', line)

def run_simulator():
    cmd = "open udacity-sim-mac.app"

    try:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)  # subprocess as os.system py doc

        with p.stdout:
            log_subprocess_output(p.stdout)

        exitcode = p.wait()
    except (OSError, subprocess.CalledProcessError) as exception:
        logging.info('Exception occured: ' + str(exception))
        logging.info('Subprocess failed')
        return False
    else:
        # no exception was raised
        logging.info('Subprocess finished')

    return True

if __name__ == '__main__':

    # cfg = Config()
    # cfg.from_pyfile("config_my.py")

    run_simulator()

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    server = eventlet.wsgi.server(eventlet.listen(("", 5000)), app)


