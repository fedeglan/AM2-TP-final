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
import os


class ModelTrainingPipeline(object):
    def __init__(self, input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path

    def read_data(self) -> pd.DataFrame:
        """
        Read input data from a file and return as a DataFrame.

        :return: The input data as a DataFrame
        :rtype: pd.DataFrame
        """
        pandas_df = pd.read_csv(self.input_path, index_col=0)
        pandas_df = pandas_df.loc[pandas_df["isTrain"] == 1]
        return pandas_df

    def model_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Train a linear regression model using the input DataFrame.

        :param df: The input DataFrame containing training data
        :type df: pd.DataFrame
        :return: The trained linear regression model
        :rtype: LinearRegression
        """
        # Separate X_train & Y_train
        y_train = df["Item_Outlet_Sales"]
        x_train = df.drop(columns=["Item_Outlet_Sales", "isTrain"])

        # Train model
        seed = 28
        model = LinearRegression()
        model.fit(x_train, y_train)
        return model

    def model_dump(self, model_trained) -> None:
        """
        Save the trained model to a file using pickle.

        :param model_trained: The trained model
        :type model_trained: LinearRegression
        """
        with open(self.model_path, "wb") as file:
            pickle.dump(model_trained, file)
        return None

    def run(self):
        """
        Executes model training.
        """
        df = self.read_data()
        model_trained = self.model_training(df)
        self.model_dump(model_trained)


if __name__ == "__main__":
    path_to_data_folder = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data")
    ModelTrainingPipeline(
        input_path=os.path.join(path_to_data_folder, 
                                 "transformed_dataset.csv"),
        model_path=os.path.join(path_to_data_folder, 
                                 "model.pickle"),
    ).run()
