"""
train.py

Module for training a machine learning model.

DESCRIPTION: This module contains a ModelTrainingPipeline class 
that reads input data, trains a linear regression model, and saves
 the trained model to a file.
AUTHOR: Federico Glancszpigel
DATE: 15/7/2023
"""

# Imports
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import argparse
import logging


class ModelTrainingPipeline(object):
    def __init__(self, input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path

        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        format = '%(asctime)s - %(levelname)s - Training - %(message)s'
        formatter = logging.Formatter(format, datefmt='%Y-%m-%d %H:%M:%S')
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def read_data(self) -> pd.DataFrame:
        """
        Read input data from a file and return as a DataFrame.

        :return: The input data as a DataFrame
        :rtype: pd.DataFrame
        """
        try:
            pandas_df = pd.read_csv(self.input_path, index_col=0)
            self.logger.debug("data was read succesfully.")
            return pandas_df
        except Exception as err:
            self.logger.error("data could not be read. "
                              f"The following error was raised: {err}")
            return None

    def model_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Train a linear regression model using the input DataFrame.

        :param df: The input DataFrame containing training data
        :type df: pd.DataFrame
        :return: The trained linear regression model
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
            seed = 28
            model = LinearRegression()
            model.fit(x_train, y_train)
            self.logger.debug("model was trained sucesfully.")
            return model
        except Exception as err:
            self.logger.error("model couldn't be trained. "
                              f"Error: {err}.")
            return None

    def model_dump(self, model_trained) -> None:
        """
        Save the trained model to a file using pickle.

        :param model_trained: The trained model
        :type model_trained: LinearRegression
        """
        try:
            with open(self.model_path, "wb") as file:
                pickle.dump(model_trained, file)
            self.logger.debug("model was saved sucesfully.")
        except Exception as err:
            self.logger.error("model couldn't be saved. "
                              f"Error: {err}.")
        return None

    def run(self):
        """
        Executes model training.
        """
        df = self.read_data()
        if df is not None:
            model_trained = self.model_training(df)
            if model_trained is not None:
                self.model_dump(model_trained)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path", help="Path of the input data (transformed)")
    parser.add_argument(
        "model_path", help="Path where to save the model's pickle")
    args = parser.parse_args()

    ModelTrainingPipeline(
        input_path=args.data_path,
        model_path=args.model_path,
    ).run()
