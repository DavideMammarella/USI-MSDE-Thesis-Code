# Predicting Safety-Critical Misbehaviours in Autonomous Driving Systems using Uncertainty Quantification

A **client-server** architecture to control an **autonomous driving vehicle** within the **Udacity simulator** including two monitors (**black-box** and **white-box**) to predict **misbehaviours**.

# Contents

<!--ts-->
   * [Repository Structure](#repository-structure)
   * [Development Environment](#development-environment)
     * [System Requirements](#system-requirements)
     * [Simulator](#simulator)
   * [Usage](#usage)
     * [Train](#train)
     * [Drive](#drive)
   * [Replicate Experiments](#replicate-experiments)
     * [Collect Simulations](#collect-simulations)
     * [Calculate Uncertainties](#calculate-uncertainties)
     * [Calculate Thresholds](#calculate-thresholds)
     * [Time Series Analysis](#time-series-analysis)
<!--te-->

# Repository Structure

    .
    ├── analysis                # Analysis scripts (Thresholds, Time Series Analysis).
    ├── client                  # Self-Driving Car scripts (train and drive).
    ├── configurations          # Configurations files (conda, client).
    ├── data                    
    │   ├── datasets            # Training datasets.
    │   ├── metrics             # Generated metrics from analysis (as CSV).
    │   └── simulations         # Evaluation datasets.
    ├── docs                    # Documentation files.
    ├── models                  # Trained and serialized models (Autoencoders, Self-Driving Car)
    ├── monitors
    │   ├── black_box           # SelfOracle (Black-Box monitor) scripts.
    │   └── white_box           # White-Box monitor scripts.
    ├── server                  # Simulator binaries.
    ├── utils                   # Utils files.
    └── main.py

Folders not in the repository are automatically created as needed by the code, except for the ``server`` folder, which must be created and within it the simulator binary must be placed.

[Back to top ↑](#contents)

# Development Environment

## System Requirements

The client requires [Python 3.6](https://www.python.org/downloads/release/python-360/). Development with [PyCharm](https://www.jetbrains.com/pycharm/) IDE is recommended. <br>
The use of [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html) is also recommended for installing dependencies in a dedicated virtual environment. Alternatively, the libraries can be installed using ``pip``.

According to the operating system, the environment can be created with the command:
```python
conda env create -f configurations/lin-minenv.yml   # Linux
conda env create -f configurations/mac-minenv.yml   # Mac
conda env create -f configurations/win-minenv.yml   # Windows
```

**Note on the Hardware**: Deep Neural Network training is intensive, so the use of a machine with a GPU is recommended.

[Back to top ↑](#contents)

## Simulator

The simulator is made available as binary file here. <br>
The binary must be downloaded and placed in the ``server`` folder.

[Back to top ↑](#contents)

# Usage
The client allows different configurations (both for training and driving). <br>
Firstly, the ``configurations/config_my.py.sample`` file must be duplicated and renamed to ``config_my.py``.<br>
Next, one's own configuration can be created by modifying the variables within the file. <br>

[Back to top ↑](#contents)

## Train
Datasets obtained by driving manually in the simulator are required. <br>
They can be collected using the simulator in Training Mode. <br>
Alternatively, datasets used in experiments are available on request.

The workflow for training is as follows:
* Insert the datasets into the ``data/datasets`` folder
* Run the script ``client/sdc_train.py``

This will generate a model ``<track>-<model>-final`` inside ``models`` folder. <br>
Together with the final model, ``<track>-<model>-<epoch>`` models will be generated whenever the performance in the epoch is better than the previous best. <br>
For example, the first epoch within the first track will generate a file called ``track1-<model>-000``.

[Back to top ↑](#contents)

## Drive

Autoencoder are required and must be placed in the ``models/sao`` folder. <br>
They can be trained, without using the simulator but the datasets collected in the [Train](#train) section, by running ``monitors/black_box/vae_train.py``. <br>
Alternatively, autoencoder used in experiments are available on request.

The workflow for autonomous driving is as follows:
* Make sure the models (Autoencoders, Self-Driving Car) are in the ``models`` folder
* * Edit the variable ``SDC_MODEL_NAME`` within ``configurations/config_my.py`` according to the self-driving car model
* Run the script ``main.py``
* Select Track / Time / Weather Effect / Emission Rate from the simulator main menu
* Click on **Autonomous Mode**

[Back to top ↑](#contents)

# Replicate Experiments

## Collect Simulations
Experiments are performed offline. <br>
The simulations are necessary, they are data collected by the simulator during autonomous driving. <br>
The simulations used in the experiments are available on request. <br>
Alternatively, they can be collected as follows:
* Edit the variable ``TESTING_DATA_DIR`` within ``configurations/config_my.py`` by entering the value ``"data/simulations"``
* Start autonomous driving (See [Drive](#drive))

This will generate a folder within ``data/simulations`` containing the simulation data. <br>
The process can be repeated by inserting various Track / Time / Weather Effect / Emission Rate, thus obtaining different types of simulations.

[Back to top ↑](#contents)

## Calculate Uncertainties

The simulations to be analysed are necessary (See [Collect Simulations](#collect-simulations)). <br>
The workflow for uncertainties calculation is as follows:
* Make sure the stochastic model (i.e. MC-Dropout) is in the ``models`` folder
* Edit the variable ``SDC_MODEL_NAME`` within ``configurations/config_my.py`` according to the self-driving car model
* Run the script ``monitors/white_box/uncertainties.py``

This will generate for each simulation a folder with the name of the simulation ending with ``-uncertainty-evaluated``. <br>
Each folder contains a CSV file with all the original telemetry data and the uncertainties calculated by the model for each frame.

[Back to top ↑](#contents)

## Calculate Thresholds

The simulations to be analysed are necessary (See [Collect Simulations](#collect-simulations)). <br>
The workflow for thresholds calculation is as follows:
* Make sure the stochastic model (i.e. MC-Dropout) is in the ``models`` folder
* Make sure there is a **nominal simulation** within the ``data/simulations`` and that it contains the word "**normal**" in the folder name
* Run the script ``analysis/thresholds.py``

This will generate a JSON file under ``data/`` containing thresholds divided by confidence intervals. <br>
The thresholds are calculated by fitting the gamma distribution to the uncertainties.

[Back to top ↑](#contents)

## Time Series Analysis

Two different types of analysis are performed and performance metrics are calculated for each. <br>
The uncertainties to be analysed are necessary (See [Collect Uncertainties](#collect-uncertainties)). <br>
The thresholds are also necessary (See [Collect Thresholds](#collect-thresholds)).

The workflow for the analysis is as follows:
* Edit the script ``analysis/time_series.py`` by adding the thresholds under ``THRESHOLDS`` variable
* Run the script 

This will generate time series analysis results in the form of performance metrics to the ``data/metrics`` folder.

[Back to top ↑](#contents)