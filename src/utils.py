"""
utils.py

This module provides utility functions for supporting the
feature engineering, training and inference pipelines. 

DESCRIPTION: This module contains utility classes and functions.
AUTHOR: Federico Glancszpigel
DATE: 26/7/2023
"""

import os
import pandas as pd
import logging
import pickle


def setup_logger(label):
    """
    Set up a logger with the specified label.

    :param label: A label to be included in the log messages.
    :type label: str
    :rtype: logging.logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    format = '%(asctime)s - %(levelname)s -' + label + '- %(message)s'
    formatter = logging.Formatter(format, datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


class FTPDatabase:
    """
    Class representing an FTP database.

    :param folder_path: The path to the folder containing
        the database files.
    :type folder_path: str
    """

    def __init__(self, folder_path):
        self.folder_path = folder_path

    def import_file(self, file_name):
        """
        Import a file from the database.

        :param file_name: The name of the file to import.
        :type file_name: str
        :return: The imported data as a DataFrame.
        :rtype: pd.DataFrame
        """
        extension = file_name.split(".")[-1]
        file_path = os.path.join(self.folder_path, file_name)
        if extension == "csv":
            return pd.read_csv(file_path)
        elif extension == "json":
            return pd.read_json(file_path,
                                orient="index").T
        elif extension == "pickle":
            with open(file_path, "rb") as file:
                data = pickle.load(file)
                file.close()
            return data
        else:
            raise NotImplementedError(f"{extension} not implemented. "
                                      "Please provide a file_name with "
                                      "csv, json or pickle extension.")

    def save_file(self, file, file_name):
        """
        Save a file to the database.

        :param file: The data to be saved.
        :type file: pd.DataFrame
        :param file_name: The name of the file to save.
        :type file_name: str
        """
        extension = file_name.split(".")[-1]
        file_path = os.path.join(self.folder_path, file_name)
        if extension == "csv":
            file.to_csv(file_path)
        elif extension == "xlsx":
            file.to_excle(file_path)
        elif extension == "pickle":
            with open(file_path, "wb") as f:
                pickle.dump(file, f)
                f.close()
        else:
            raise NotImplementedError(f"{extension} not implemented. "
                                      "Please provide a file_name with "
                                      "csv, json or pickle extension.")
