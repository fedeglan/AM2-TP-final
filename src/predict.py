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
import pickle
import os
import argparse
import logging


class MakePredictionPipeline(object):
    def __init__(self, input_path, output_path, model_path: str = None):
        """
        Initialize the MakePredictionPipeline object.

        :param input_path: The file path of the input data.
        :param output_path: The file path to save the predicted data.
        :param model_path: The file path of the trained model (optional).
        """
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path

        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        format = '%(asctime)s - %(levelname)s - Prediction - %(message)s'
        formatter = logging.Formatter(format, datefmt='%Y-%m-%d %H:%M:%S')
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def load_data(self) -> pd.DataFrame:
        """
        Load the input data for making predictions.

        :return: The input data as a pandas DataFrame.
        """
        try:
            data = pd.read_csv(self.input_path, index_col=0)
            if "Item_Outlet_Sales" in data.columns:
                data = data.drop(columns=["Item_Outlet_Sales"])
            self.logger.debug("data was loaded sucesfully.")
            return data
        except Exception as err:
            self.logger.error("data could not be loaded. "
                              f"Error: {err}.")

    def load_model(self) -> None:
        """
        Load the trained model for making predictions.
        """
        try:
            with open(self.model_path, "rb") as file:
                self.model = pickle.load(file)
                file.close()
            self.logger.debug("model was loaded sucesfully.")
        except Exception as err:
            self.logger.error("model could not be loaded. "
                              f"Error: {err}.")
        return None

    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on the input data using the trained model.

        :param data: The input data as a pandas DataFrame.
        :return: The predicted values as a pandas DataFrame.
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

    def write_predictions(self, predicted_data: pd.DataFrame) -> None:
        """
        Write the predicted data to a CSV file.

        :param predicted_data: The predicted data as a pandas DataFrame.
        """
        try:
            predicted_data.to_csv(self.output_path)
            self.logger.debug("predictions were saved sucesfully.")
        except Exception as err:
            self.logger.error("couldn't make a prediction. "
                              f"Error: {err}")
        return None

    def run(self):
        """
        Runs the prediction pipeline.
        """
        data = self.load_data()
        if data is not None:
            self.load_model()
            if hasattr(self, "model"):
                df_preds = self.make_predictions(data)
                if df_preds is not None:
                    self.write_predictions(df_preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path", help="Path of the input data (transformed)")
    parser.add_argument(
        "model_path", help="Path where to save the model's pickle")
    parser.add_argument(
        "output_path", help="Path where to save the predictions")
    args = parser.parse_args()

    MakePredictionPipeline(
        input_path=args.data_path,
        model_path=args.model_path,
        output_path=args.output_path
    ).run()
