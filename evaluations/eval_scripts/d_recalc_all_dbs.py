import sys

sys.path.append("..")

from evaluations.eval_scripts import b_precision_recall_auroc, a_set_true_labels
from pathlib import Path
import utils
import os

def collect_databases(sims_path):
    dbs = []
    for db_path in sims_path.iterdir():
        dbs.append(db_path.name)
    print(">> Collected db: " + str(len(dbs)) + "\n")
    return dbs

def recalc_all(dbs, dbs_path):
    for db in dbs:
        db_path = Path(dbs_path, db)
        print(db_path)
        if db == '../../models/trained-anomaly-detectors/20190821-ALL-MODELS-MODIFIED_TRACKS.sqlite':
            b_precision_recall_auroc.AUROC_CALC_SAMPLING_FACTOR = 20
        else:
            b_precision_recall_auroc.AUROC_CALC_SAMPLING_FACTOR = 1
        a_set_true_labels.set_true_labels(str(db_path))
        b_precision_recall_auroc.calc_precision_recall(db)

def main():
    root_dir = utils.get_project_root()
    dbs_path = Path(root_dir, "databases")
    dbs = collect_databases(dbs_path)
    recalc_all(dbs, dbs_path)


if __name__ == '__main__':
    main()