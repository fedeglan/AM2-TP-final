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

    def load_data(self) -> pd.DataFrame:
        """
        Load the input data for making predictions.

        :return: The input data as a pandas DataFrame.
        """
        data = pd.read_csv(self.input_path, index_col=0)
        data = data.loc[data["isTrain"] == 0]
        data = data.drop(columns=["Item_Outlet_Sales", "isTrain"])
        return data

    def load_model(self) -> None:
        """
        Load the trained model for making predictions.
        """
        with open(self.model_path, "rb") as file:
            self.model = pickle.load(file)
        return None

    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on the input data using the trained model.

        :param data: The input data as a pandas DataFrame.
        :return: The predicted values as a pandas DataFrame.
        """
        new_data = self.model.predict(data)
        new_data = pd.DataFrame(new_data, columns=["Item_Outlet_Sales"])
        return new_data

    def write_predictions(self, predicted_data: pd.DataFrame) -> None:
        """
        Write the predicted data to a CSV file.

        :param predicted_data: The predicted data as a pandas DataFrame.
        """
        predicted_data.to_csv(self.output_path)
        return None

    def run(self):
        """
        Runs the prediction pipeline.
        """
        data = self.load_data()
        self.load_model()
        df_preds = self.make_predictions(data)
        self.write_predictions(df_preds)


if __name__ == "__main__":
    path_to_data_folder = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data"
    )
    pipeline = MakePredictionPipeline(
        input_path=os.path.join(
            path_to_data_folder, "transformed_dataset.csv"
        ),
        output_path=os.path.join(path_to_data_folder, "predictions.csv"),
        model_path=os.path.join(path_to_data_folder, "model.pickle"),
    )
    pipeline.run()
