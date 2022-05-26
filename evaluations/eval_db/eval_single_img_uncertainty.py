import sys

sys.path.append("..")
import logging
from typing import List

import evaluations.utils_logging as utils_logging
from evaluations.eval_db.database import Database

INSERT_STATEMENT = "INSERT INTO single_image_based_uncertainties ('setting_id', 'row_id', 'is_crash', 'uncertainty') values (?,?,?,?);"
logger = logging.Logger("SingleImgUncertainty")
utils_logging.log_info(logger)


class SingleImgUncertainty:
    def __init__(
        self, setting_id: int, row_id: int, is_crash: bool, uncertainty: float
    ):
        self.setting_id = setting_id
        self.row_id = row_id
        self.is_crash = is_crash
        self.uncertainty = uncertainty
        self.true_label = None
        self.count_to_crash = None

    def insert_into_db(self, db: Database) -> None:
        int_is_crash = 0
        if self.is_crash:
            int_is_crash = 1
        db.cursor.execute(
            INSERT_STATEMENT,
            (self.setting_id, self.row_id, int_is_crash, self.uncertainty),
        )

    def uncertainty_of(self, ad_name: str):
        if ad_name == "uwiz":
            return self.uncertainty
        logger.error("Unknown ad_name")
        assert False


def load_all_for_setting(
    db: Database, setting_id: int
) -> List[SingleImgUncertainty]:
    cursor = db.cursor.execute(
        "select * from single_image_based_uncertainties where setting_id=? "
        + "order by row_id",
        (setting_id,),
    )
    var = cursor.fetchall()
    result = []
    for db_record in var:
        int_is_crash = db_record[2]
        is_crash = False
        if int_is_crash == 1:
            is_crash = True
        elif int_is_crash != 0:
            logger.error("Unknown is_crash bool encoding")
            exit(1)
        uncertainty_object = SingleImgUncertainty(
            setting_id=db_record[0],
            row_id=db_record[1],
            is_crash=is_crash,
            uncertainty=db_record[3],
        )
        result.append(uncertainty_object)
    return result


def update_true_label_on_db(db: Database, records: List[SingleImgUncertainty]):
    for record in records:
        insertable = (
            record.true_label,
            record.count_to_crash,
            record.setting_id,
            record.row_id,
        )
        db.cursor.execute(
            "update single_image_based_uncertainties set true_label=?, count_to_crash=? "
            + "where main.single_image_based_uncertainties.setting_id = ?"
            + " and main.single_image_based_uncertainties.row_id = ? ",
            insertable,
        )
