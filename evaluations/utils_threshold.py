import json
import logging
import os
from pathlib import Path

import numpy
import utils_logging
from scipy.stats import gamma

logger = logging.Logger("utils_thresholds")
utils_logging.log_info(logger)


def calc_and_store_thresholds(
    uncertainties: numpy.array, thresholds_location
) -> dict:
    """
    Calculates all thresholds stores them on a file system
    :param losses: array of shape (n,),
                    where n is the number of training data points, containing the losses calculated for these points
    :return: a dictionary of where key = threshold_identifier and value = threshold_value
    """
    Path(thresholds_location).mkdir(parents=True, exist_ok=True)

    logger.info(
        "Fitting reconstruction error distribution of UWIZ using Gamma distribution params"
    )

    shape, loc, scale = gamma.fit(uncertainties, floc=0)
    thresholds = {}

    conf_intervals = [0.68, 0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999]

    logger.info(
        "Creating thresholds using the confidence intervals: %s"
        % conf_intervals
    )

    for c in conf_intervals:
        thresholds[str(c)] = gamma.ppf(c, shape, loc=loc, scale=scale)

    as_json = json.dumps(thresholds)

    json_filename = str(thresholds_location) + "/uwiz.json"

    print("Saving thresholds to %s" % json_filename)

    if os.path.exists(json_filename):
        os.remove(json_filename)

    with open(json_filename, "a") as fp:
        fp.write(as_json)

    return thresholds
