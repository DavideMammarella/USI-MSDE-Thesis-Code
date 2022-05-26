import sys
sys.path.append("..")
import logging

import evaluations.utils_logging as utils_logging
from evaluations.eval_db.database import Database

SINGLE_IMAGE_BASED_ADS = ["uwiz"]

INSERT_STRING = "INSERT INTO 'windows' ('setting', 'window_id', 'ad_name', 'uncertainty_score', 'type', 'start_frame', 'end_frame') VALUES (?,?,?,?,?,?,?)"

CALC_SCOPE = 4

logger = logging.Logger("eval_windows")
utils_logging.log_info(logger)


def calc_uncertainty_score(all_uncertainty_objects, start_frame, end_frame_excl, ad_name) -> float:
    sums = []
    for i in range(start_frame, end_frame_excl - CALC_SCOPE + 1):
        sum = 0
        for j in range(i, i + CALC_SCOPE):
            sum = sum + all_uncertainty_objects[i].uncertainty_of(ad_name=ad_name)
        sums.append(sum)
    max_sum = max(sums)
    return max_sum / CALC_SCOPE

class Window:
    def __init__(self,
                 setting: int,
                 window_id: int,
                 ad_name: str,
                 window_type: str,
                 start_frame: int,
                 end_frame: int,
                 all_uncertainty_objects
                 ):
        self.setting = setting
        self.window_id = window_id
        self.ad_name = ad_name
        self.uncertainty_score = calc_uncertainty_score(all_uncertainty_objects=all_uncertainty_objects, start_frame=start_frame,
                                          end_frame_excl=end_frame, ad_name=ad_name)
        self.window_type = window_type
        self.start_frame = start_frame
        self.end_frame = end_frame


def insert_into_db(db: Database, window: Window):
    insertable = (
        window.setting,
        window.window_id,
        window.ad_name,
        window.uncertainty_score,
        window.window_type,
        window.start_frame,
        window.end_frame,
    )
    db.cursor.execute(INSERT_STRING, insertable)


def get_true_positives_count(db: Database, ad_name: str, threshold: float):
    cursor = db.cursor.execute("select count(*) from windows " +
                               "where ad_name = ? and uncertainty_score >= ? and type = 'anomaly'", (ad_name, threshold))
    return _unpack_single_result_cursor(cursor)


def get_false_positives_count_ignore_subsequent(db: Database, ad_name: str, threshold: float):
    cursor = db.cursor.execute('''select count(*) from windows 
                               where ad_name = ? and uncertainty_score >= ? and type = 'normal'
                               and not exists(
                                    select 1 
                                    from windows as w2
                                    where w2.window_id = windows.window_id -1
                                    and w2.setting = windows.setting
                                    and ad_name = ? and uncertainty_score >= ? and type = 'normal'                                    
                               )
                               '''
                               , (ad_name, threshold, ad_name, threshold))
    return _unpack_single_result_cursor(cursor)


def get_true_negatives_count(db: Database, ad_name: str, threshold: float):
    cursor = db.cursor.execute("select count(*) from windows " +
                               "where ad_name = ? and uncertainty_score < ? and type = 'normal'", (ad_name, threshold))
    return _unpack_single_result_cursor(cursor)


def get_false_negatives_count(db: Database, ad_name: str, threshold: float):
    cursor = db.cursor.execute("select count(*) from windows " +
                               "where ad_name = ? and uncertainty_score < ? and type = 'anomaly'", (ad_name, threshold))
    return _unpack_single_result_cursor(cursor)


def _unpack_single_result_cursor(cursor):
    records = cursor.fetchall()
    assert len(records) == 1
    assert len(records[0]) == 1
    result = records[0][0]
    return result


def get_all_uncertainties_and_true_labels_for_ad(db, ad_name):
    sql = "select type, uncertainty_score from windows where ad_name=?"
    cursor = db.cursor.execute(sql, (ad_name,))
    uncertainties, true_labels = _unpack_labelled_uncertainties_cursor(cursor)
    return true_labels, uncertainties


def _unpack_labelled_uncertainties_cursor(cursor):
    records = cursor.fetchall()
    true_labels = []
    uncertainties = []
    for db_record in records:
        label = db_record[0]
        if label == "normal":
            int_label = 0
        elif label == "anomaly":
            int_label = 1
        else:
            logger.error("Must be either normal or anomaly")
            assert False

        true_labels.append(int_label)
        uncertainties.append(db_record[1])
    return uncertainties, true_labels

def store_single_image_windows(setting: int, window_id: int, window_type: str, entries, start_inc: int, end_excl: int,
                               db: Database,
                               ad_type=str):
    ads = SINGLE_IMAGE_BASED_ADS

    for ad_name in ads:
        window = Window(setting=setting, window_id=window_id, ad_name=ad_name, start_frame=start_inc,
                        end_frame=end_excl, all_uncertainty_objects=entries, window_type=window_type)
        insert_into_db(window=window, db=db)


def remove_all_stored_records(db: Database):
    db.cursor.execute("delete from windows")
