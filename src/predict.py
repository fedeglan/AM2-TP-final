"""
predict.py

Script for making predictions using a trained model.

DESCRIPTION: This script loads a trained model and makes
predictions on new data. The input data should be in CSV 
format with the same features as used during training. 
The script outputs the predicted values as a CSV file.

AUTHOR: Federico Glancszpigel

DATE: 15/7/2023
"""

# Imports
import pandas as pd
import os
import sys

# Local imports
sys.path.append(os.path.dirname(__file__))
from utils import setup_logger, FTPDatabase

class MakePredictionPipeline(object):
    """
    MakePredictionPipeline class for making predictions using
    a trained model.

    This class provides methods to load input data, load a trained model,
    make predictions, and save the predicted data to a file.
    """
    
    def __init__(self, folder_path: str):
        """
        Initialize the MakePredictionPipeline object.

        :param folder_path: Path to the folder containing the data files.
        :type folder_path: str
        """
        # Setup database
        self.database = FTPDatabase(folder_path)

        # Set up logger
        self.logger = setup_logger("Predict")

    def load_data(self, file_name: str) -> pd.DataFrame:
        """
        Load the input data for making predictions.

        :param file_name: The name of the input data file.
        :type file_name: str
        :return: The input data as a pandas DataFrame.
        :rtype: pd.DataFrame
        """
        try:
            data = self.database.import_file(file_name)
            try:
                data = data.drop(columns=["Unnamed: 0"])
            except:
                pass
            if "Item_Outlet_Sales" in data.columns:
                data = data.drop(columns=["Item_Outlet_Sales"])
            self.logger.debug("data was loaded sucesfully.")
            return data
        except Exception as err:
            self.logger.error("data could not be loaded. "
                              f"Error: {err}.")

    def load_model(self, file_name: str) -> None:
        """
        Load the trained model for making predictions.

        :param file_name: The name of the trained model file.
        :type file_name: str
        """
        try:
            self.model = self.database.import_file(file_name)
            self.logger.debug("model was loaded sucesfully.")
        except Exception as err:
            self.logger.error("model could not be loaded. "
                              f"Error: {err}.")
        return None

    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on the input data using the trained model.

        :param data: The input data as a pandas DataFrame.
        :type data: pd.DataFrame
        :return: The predicted values as a pandas DataFrame.
        :rtype: pd.DataFrame
        """
        if hasattr(self, "model"):
            try:
                new_data = self.model.predict(data)
                new_data = pd.DataFrame(
                    new_data, columns=["Item_Outlet_Sales"])
                self.logger.debug("a new prediction was made.")
                return new_data
            except Exception as err:
                self.logger.error("couldn't make a prediction. "
                                  f"Error: {err}")
                return None
        else:
            self.logger.error("model has not been loaded.")
            return None

    def write_predictions(self, predicted_data: pd.DataFrame,
                          file_name: str) -> None:
        """
        Write the predicted data to a CSV file.

        :param predicted_data: The predicted data as a pandas DataFrame.
        :type predicted_data: pd.DataFrame
        """
        try:
            self.database.save_file(predicted_data, file_name)
            self.logger.debug("predictions were saved sucesfully.")
        except Exception as err:
            self.logger.error("couldn't make a prediction. "
                              f"Error: {err}")
        return None

    def run(self, data_file_name, model_file_name, output_file_name):
        """
        Execute the prediction pipeline.

        :param data_file_name: The name of the input data file.
        :type data_file_name: str
        :param model_file_name: The name of the trained model file.
        :type model_file_name: str
        :param output_file_name: The name of the output file to save the predictions.
        :type output_file_name: str
        """
        data = self.load_data(data_file_name)
        if data is not None:
            self.load_model(model_file_name)
            if hasattr(self, "model"):
                df_preds = self.make_predictions(data)
                if df_preds is not None:
                    self.write_predictions(df_preds, output_file_name)