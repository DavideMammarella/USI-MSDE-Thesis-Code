import sys

sys.path.append("..")

import logging

import evaluations.utils_logging as utils_logging
from evaluations.eval_db.database import Database
from evaluations.eval_scripts import b_precision_recall_auroc
from evaluations.eval_scripts.utils import threshold_independent_plotters

logger = logging.Logger("c_timeline")
utils_logging.log_info(logger)


def print_auroc_timeline(db_name):
    db = Database(name=db_name, delete_existing=False)

    if b_precision_recall_auroc.AUROC_CALC_SAMPLING_FACTOR != 1:
        logger.warning(
            "Sampling is >1, this is good for testing, but must not be the case for final version graphs"
        )

    # threshold_dependent_reaction_plotter = ThresholdDependentReactionTimePlotter(db=db)
    # threshold_dependent_reaction_plotter.compute_and_plot()

    reaction_plotter = threshold_independent_plotters.ReactionTimePlotter(
        db=db
    )
    reaction_plotter.compute_and_plot(db_name)

    # k_plotter = threshold_independent_plotters.KSizePlotter(db=db)
    # k_plotter.compute_and_plot()


if __name__ == "__main__":
    db_name = None
    print_auroc_timeline(db_name)
