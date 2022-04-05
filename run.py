import multiprocessing
import signal
import subprocess
import os
from datetime import datetime
from multiprocessing import Process, Event

from multiprocessing.pool import ThreadPool

########################################################################################################################
# Multiprocessing good for CPU related jobs (https://www.oreilly.com/library/view/hands-on-microservices-with/9781789342758/82531d54-039c-4c96-8fcb-58a53cee28e6.xhtml)
# Multithreading could be better for I/O, but I/O is fast on this project (while calculation of unc require a lot of computation)
########################################################################################################################
pool = []
pids = []

def kill(proc_pid):
    import psutil
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

def close_threads():
    print('{} | terminating processes'.format(datetime.now()))
    for p in pool:
        print(p)
        p.terminate()
    for p in pool:
        p.join()
    print('{} | all processes joined'.format(datetime.now()))
    for p in pids:
        os.kill(p, signal.SIGTERM)
    return

if __name__ == '__main__':
    scripts = ["server_simulator.py", "client_drive.py"]
    terminate_processes_event = Event()

    for script in scripts:
        process = multiprocessing.Process(target=subprocess.run, args=(["python", script],))
        print('{} | pid: {} created'.format(datetime.now(), os.getpid()))
        pids.append(os.getpid())
        process.start()
        pool.append(process)


########################################################################################################################
##### MULTITHREAD SOLUTION, SLOW
########################################################################################################################
# _FINISH = False
#
# def close_threads():
#     print("Closing threads...")
#     _FINISH = True
#     pool.terminate()
#     pool.join()
#     print("Threads successfully closed!")
#
# def start_thread(script):
#     if _FINISH:
#         return
#     process = multiprocessing.Process(target=subprocess.run, args=(["python", script],))
#     process.start()
#
# if __name__ == '__main__':
#     pool = ThreadPool(processes=2)
#     scripts = ["server_simulator.py", "client_drive.py"]
#
#     for script in scripts:
#         pool.apply_async(start_thread(script))
#
#     if _FINISH:
#         sys.exit()