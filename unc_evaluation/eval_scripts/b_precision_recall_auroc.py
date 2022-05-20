import sys
sys.path.append("..")

import logging
from typing import Tuple

import numpy
from sklearn.metrics import auc

import utils_logging
from eval_db import eval_prec_recall, eval_window
from eval_db.database import Database
from eval_db.eval_prec_recall import PrecisionRecallAnalysis

CALC_AUROC = True
CALC_PREC_RECALL = False

AUROC_CALC_SAMPLING_FACTOR = 20  # 1 = No Sampling, n = 1/n of the losses are considered

# TODO: edit this based on my experiments
THRESHOLDS = {
    "uwiz": {"0.68": 29.361018856328876, "0.9": 40.45529729327319, "0.95": 46.16822971023903, "0.99": 58.20754461663782,
            "0.999": 73.82588456863898, "0.9999": 88.42246606777259, "0.99999": 102.40142769284085}
}

logger = logging.Logger("Calc_Precision_Recall")
utils_logging.log_info(logger)


def calc_precision_recall(db_name):
    logger.warning("ATTENTION: Thresholds are hardcoded. Copy-paste after recalculating thresholds " +
                   "(hence, after each training of the models)!")


    db = Database(name=db_name, delete_existing=False)
    eval_prec_recall.remove_all_from_prec_recall(db=db)

    for ad_name, ad_thresholds in THRESHOLDS.items():
        _eval(ad_name=ad_name, ad_thresholds=ad_thresholds, db=db)
    db.commit()


def _eval(ad_name, ad_thresholds, db):
    # TODO Store auc_prec_recall in db as well
    auroc, auc_prec_recall = calc_auroc_and_auc_prec_recall(db=db, ad_name=ad_name)
    for threshold_type, threshold in ad_thresholds.items():
        precision_recall_analysis = create_precision_recall_analysis(ad_name=ad_name,
                                                                     auroc=auroc, auc_prec_recall=auc_prec_recall,
                                                                     db=db, threshold=threshold,
                                                                     threshold_type=threshold_type)
        eval_prec_recall.insert_into_db(db=db, precision_recall=precision_recall_analysis)
    db.commit()


def create_precision_recall_analysis(ad_name, auroc, auc_prec_recall, db, threshold, threshold_type):
    true_positives = eval_window.get_true_positives_count(db=db, ad_name=ad_name, threshold=threshold)
    false_positives = eval_window.get_false_positives_count_ignore_subsequent(db=db, ad_name=ad_name,
                                                                              threshold=threshold)
    true_negatives = eval_window.get_true_negatives_count(db=db, ad_name=ad_name, threshold=threshold)
    false_negatives = eval_window.get_false_negatives_count(db=db, ad_name=ad_name, threshold=threshold)
    precision_recall_analysis = PrecisionRecallAnalysis(anomaly_detector=ad_name,
                                                        threshold_type=threshold_type,
                                                        threshold=threshold,
                                                        true_positives=true_positives,
                                                        false_positives=false_positives,
                                                        true_negatives=true_negatives,
                                                        false_negatives=false_negatives,
                                                        auroc=auroc,
                                                        auc_prec_recall=auc_prec_recall
                                                        )
    return precision_recall_analysis


# Method also used by auroc plotter
def _calc_auc_roc(false_positive_rates, true_positive_rates):
    pass


def calc_auroc_and_auc_prec_recall(db: Database, ad_name: str) -> Tuple[float, float]:
    labels_ignore_this, losses_list = eval_window.get_all_uncertainties_and_true_labels_for_ad(db=db, ad_name=ad_name)
    false_positive_rates = []
    true_positive_rates = []
    precisions = []
    f1s = []
    logger.info("Calc auc-roc for " + ad_name + " based on " + str(
        len(losses_list) / AUROC_CALC_SAMPLING_FACTOR) + " thresholds. Sampling factor: " + str(
        AUROC_CALC_SAMPLING_FACTOR))
    i = 0
    losses_list.sort()
    losses_list = losses_list[::AUROC_CALC_SAMPLING_FACTOR]
    for loss in losses_list:
        i = i + 1
        if i % 100 == 0:
            logger.info("---> " + str(i) + " out of " + str(len(losses_list)))
        # Temporary, non persisted precision_recall_analysis to calculate TPR and FPR
        precision_recall_analysis = create_precision_recall_analysis(ad_name=ad_name,
                                                                     auroc=None,
                                                                     auc_prec_recall=None,
                                                                     db=db,
                                                                     threshold=loss,
                                                                     threshold_type=None)

        fpr = precision_recall_analysis.false_positive_rate
        tpr = precision_recall_analysis.recall
        prec = precision_recall_analysis.prec
        false_positive_rates.append(fpr)
        true_positive_rates.append(tpr)
        precisions.append(prec)
    false_positive_rates = numpy.asarray(false_positive_rates)
    true_positive_rates = numpy.asarray(true_positive_rates)
    precisions = numpy.asarray(precisions)
    auc_roc = _calc_auc(x=false_positive_rates, y=true_positive_rates)
    auc_prec_recall = _calc_auc(x=true_positive_rates, y=precisions)
    return auc_roc, auc_prec_recall


def get_threshold(ad_name: str, threshold_type: str) -> float:
    if ad_name in THRESHOLDS:
        if threshold_type in THRESHOLDS[ad_name]:
            return THRESHOLDS[ad_name][threshold_type]
    assert False


def _calc_auc(x, y):
    sorted_ids = x.argsort()
    sorted_x = x[sorted_ids]
    co_sorted_y = y[sorted_ids]
    return auc(x=sorted_x, y=co_sorted_y)


if __name__ == '__main__':
    db_name = None
    calc_precision_recall(db_name)
