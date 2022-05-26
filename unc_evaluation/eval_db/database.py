import sys
sys.path.append("..")

import logging
import os
import sqlite3

import unc_evaluation.utils_logging as utils_logging

logger = logging.Logger("Database")
utils_logging.log_info(logger)


class Database:
    """
        A class to create and access a sqlite database. Tables are automatically initialized.
        The classes in the database/models package allow to read / insert / update data to such a database
    """
    connection: sqlite3.Connection = None
    cursor: sqlite3.Cursor = None

    def __init__(self, name: str, delete_existing: bool):
        if delete_existing and os.path.exists(name):
            logger.warning("Deleting sqlite file of previously ran evaluation " + name)
            os.remove(name)
        self.connection = sqlite3.connect(name)
        self.cursor = self.connection.cursor()

        if delete_existing:
            self.create_table_settings()
            self.create_table_single_img_uncertainties()
            self.create_table_prec_recall()
            self.create_table_windows()
        self.commit()

    def commit(self):
        self.connection.commit()
        self.cursor = self.connection.cursor()

    def close(self):
        self.connection.close()

    def create_table_settings(self):
        self.cursor.execute('''CREATE TABLE settings (
                    id INTEGER PRIMARY KEY ,
                    agent text NOT NULL ,
                    track INTEGER NOT NULL ,
                    time text NOT NULL,
                    weather text NOT NULL)''')

    def create_table_single_img_uncertainties(self):
        self.cursor.execute('''create table single_image_based_uncertainties
                                (
                                    setting_id INTEGER not null,
                                    row_id INTEGER not null,
                                    is_crash INTEGER not null,
                                    uncertainty NUMERIC,
                                    true_label TEXT,
                                    count_to_crash int,
                                    constraint single_image_based_uncertainties_pk
                                        primary key (setting_id, row_id)
                                );
		''')
        self.cursor.execute('''
                                create index single_image_based_uncertainties__index_row_id
                                    on single_image_based_uncertainties (row_id desc);
                                    ''')

    def create_table_prec_recall(self):
        self.cursor.execute('''
            create table prec_recall
            (
            	anomaly_detector text not null,
                threshold_type text not null,
                threshold numeric not null,
                true_positives int not null,
                false_positives int not null,
                true_negatives int not null,
                false_negatives int not null,
                prec numeric not null,
                recall numeric not null,
                f1 numeric not null,
                num_anomalies int not null,
                num_normal int not null,
                auroc float not null,
                false_positive_rate NUMERIC,
                pr_auc NUMERIC,
                constraint prec_recall_pk
                     unique (anomaly_detector, threshold_type)
            );
		    ''')

    def create_table_windows(self):
        self.cursor.execute('''
        create table windows
            (
                setting int
		            references settings,
                window_id int,
                ad_name TEXT,
                uncertainty_score NUMERIC,
                type TEXT,
                start_frame int,
                end_frame int,
                constraint windows_pk
                    primary key (setting, window_id, ad_name)
            );
            

        ''')
        self.cursor.execute('''
        create index windows_type_ad_index
            on windows (type, ad_name);
            ''')
