"""
train.py

Module for training a machine learning model.

DESCRIPTION: This module contains a ModelTrainingPipeline class 
that reads input data, trains a linear regression model, and saves
 the trained model to a file.
AUTHOR: Federico Glancszpigel
DATE: 15/7/2023
"""

# Package Imports
from utils import setup_logger, FTPDatabase
import pandas as pd
from sklearn.linear_model import LinearRegression
import sys
import os

# Local imports
sys.path.append(os.path.dirname(__file__))


class ModelTrainingPipeline(object):
    """
    ModelTrainingPipeline class for training a linear regression model.

    This class provides methods to read input data, train a
    linear regression model, and save the trained model to a file.
    """

    def __init__(self, folder_path: str):
        """
        Initialize the ModelTrainingPipeline object.

        :param folder_path: Path to the folder containing the data files.
        :type folder_path: str
        """
        # Setup database
        self.database = FTPDatabase(folder_path)

        # Set up logger
        self.logger = setup_logger("Training")

    def read_data(self, file_name: str) -> pd.DataFrame:
        """
        Read input data from a file and return as a DataFrame.

        :param file_name: The name of the input data file.
        :type file_name: str
        :return: The input data as a DataFrame.
        :rtype: pd.DataFrame
        """
        try:
            pandas_df = self.database.import_file(file_name)
            try:
                pandas_df = pandas_df.drop(columns=["Unnamed: 0"])
            except:
                pass
            self.logger.debug("data was read succesfully.")
            return pandas_df
        except Exception as err:
            self.logger.error("data could not be read. "
                              f"The following error was raised: {err}")
            return None

    def model_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Train a linear regression model using the input DataFrame.

        :param df: The input DataFrame containing training data.
        :type df: pd.DataFrame
        :return: The trained linear regression model.
        :rtype: LinearRegression
        """
        # Separate X_train & Y_train
        try:
            y_train = df["Item_Outlet_Sales"]
            x_train = df.drop(columns=["Item_Outlet_Sales"])
            self.logger.debug("data was splitted into Xs and y.")
        except Exception as err:
            self.logger.error("data could not be split into Xs and y. "
                              f"Error: {err}.")
            return None

        # Train model
        try:
            model = LinearRegression()
            model.fit(x_train, y_train)
            self.logger.debug("model was trained sucesfully.")
            return model
        except Exception as err:
            self.logger.error("model couldn't be trained. "
                              f"Error: {err}.")
            return None

    def model_dump(self, model_trained, file_name: str) -> None:
        """
        Save the trained model to a file using pickle.

        :param model_trained: The trained model.
        :type model_trained: LinearRegression
        :param file_name: The name of the output file to save the model.
        :type file_name: str
        """
        try:
            self.database.save_file(model_trained, file_name)
            self.logger.debug("model was saved sucesfully.")
        except Exception as err:
            self.logger.error("model couldn't be saved. "
                              f"Error: {err}.")
        return None

    def run(self, data_file_name, model_file_name):
        """
        Execute model training.

        :param data_file_name: The name of the input data file.
        :type data_file_name: str
        :param model_file_name: The name of the output file to save the trained model.
        :type model_file_name: str
        """
        df = self.read_data(data_file_name)
        if df is not None:
            model_trained = self.model_training(df)
            if model_trained is not None:
                self.model_dump(model_trained, model_file_name)
