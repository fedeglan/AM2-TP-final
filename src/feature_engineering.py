"""
feature_engineering.py

Module for performing feature engineering on input data.

DESCRIPTION: This module contains a FeatureEngineeringPipeline
    class that implements various data transformation steps to 
    preprocess the input data for machine learning tasks.
AUTHOR: Federico Glancszpigel
DATE: 15/7/2023
"""

# Package Imports
import pandas as pd
import copy
import sys
import os

# Local imports
sys.path.append(os.path.dirname(__file__))
from utils import setup_logger, FTPDatabase

class FeatureEngineeringPipeline(object):
    """
    Feature Engineering Pipeline for data preprocessing.

    This class performs data preprocessing and transformation
    steps for feature engineering.
    """

    def __init__(self, folder_path: str):
        """
        Initialize the FeatureEngineeringPipeline object.

        :param folder_path: Path to the folder containing the data files.
        :type folder_path: str
        """

        # Setup database
        self.database = FTPDatabase(folder_path)

        # Set up logger
        self.logger = setup_logger("FeatureEngineering")

    def read_data(self, file_name: str) -> pd.DataFrame:
        """
        Read the input data from a file.

        :param file_name: The name of the input data file.
        :type file_name: str
        :return: The input data as a DataFrame.
        :rtype: pd.DataFrame
        """
        try:
            pandas_df = self.database.import_file(file_name)
            self.logger.debug("data was read succesfully.")
            return pandas_df
        except Exception as err:
            self.logger.error("data could not be read. "
                              f"The following error was raised: {err}")
            return None

    def data_transformation(self, df: pd.DataFrame,
                            is_train: bool = False,
                            train_file_name: str = 'nan') -> pd.DataFrame:
        """
        Apply data transformations to the input DataFrame.

        :param df: The input DataFrame.
        :type df: pd.DataFrame
        :param is_train: Indicates if the DataFrame is
          for training or inference.
        :type is_train: bool
        :param train_file_name: The name of the training data file.
        :type train_file_name: str
        :return: The transformed DataFrame.
        :rtype: pd.DataFrame
        """
        df_transformed = copy.deepcopy(df)

        # Import training data
        if not is_train:
            if train_file_name != 'nan':
                train_data = self.database.import_file(train_file_name)
            else:
                self.logger.error("if is_train=False, a data file must "
                                  "be provided.")
                return None

        # 1) Correct establishment years
        try:
            df_transformed["Outlet_Establishment_Year"] = (
                2020 - df_transformed["Outlet_Establishment_Year"]
            )
            self.logger.debug("establishment years were corrected.")
        except Exception as err:
            self.logger.error("establishment years couldn't "
                              f"be corrected. Error: {err}")
            return None

        # 3) Clean missing products weight
        try:
            products = list(
                df_transformed[df_transformed["Item_Weight"].isnull()][
                    "Item_Identifier"
                ].unique()
            )
            for prod in products:
                if is_train:
                    mode = (
                        (
                            df_transformed[
                                df_transformed["Item_Identifier"] == prod
                            ][["Item_Weight"]]
                        )
                        .mode()
                    )
                else:
                    mode = (
                        (
                            train_data[
                                train_data["Item_Identifier"] == prod
                            ][["Item_Weight"]]
                        )
                        .mode()
                    )
                if not mode.empty:
                    mode = mode.iloc[0, 0]
                else:
                    mode = 0.0
                df_transformed.loc[
                    df_transformed["Item_Identifier"] == prod, "Item_Weight"
                ] = mode
            self.logger.debug("missing Item_Weight was cleaned.")
        except Exception as err:
            self.logger.error("missing Item_Weight couldn't be cleaned. "
                              f"Error: {err}")
            return None

        # 4) Missing stores size
        try:
            outlets = list(
                df_transformed[df_transformed["Outlet_Size"].isnull()][
                    "Outlet_Identifier"
                ].unique()
            )
            for outlet in outlets:
                df_transformed.loc[
                    df_transformed["Outlet_Identifier"] == outlet,
                    "Outlet_Size",
                ] = "Small"
            self.logger.debug("missing Outlet_Size was cleaned.")
        except Exception as err:
            self.logger.error("missing Outlet_Size couldn't be cleaned. "
                              f"Error: {err}")
            return None

        # 5) Embeding products prices
        try:
            if is_train:
                df_transformed["Item_MRP"] = pd.qcut(
                    df_transformed["Item_MRP"], 4, labels=[1, 2, 3, 4]
                )
            else:
                item_mrp = pd.concat([train_data["Item_MRP"],
                                      df_transformed["Item_MRP"]])
                qs = item_mrp.quantile([0, 0.25, 0.5, 0.75, 1])
                df_transformed["Item_MRP"]
                df_transformed["Item_MRP"] = pd.cut(
                    df_transformed["Item_MRP"], qs.values,
                    labels=[1, 2, 3, 4]
                )
            self.logger.debug("product prices were embedded.")
        except Exception as err:
            self.logger.error("product prices couldn't be embedded "
                              f"Error: {err}")
            return None

        # 6) Embeding of ordinal variables
        try:
            df_transformed["Outlet_Size"] = df_transformed[
                "Outlet_Size"
            ].replace({"High": 2, "Medium": 1, "Small": 0})
            df_transformed["Outlet_Location_Type"] = df_transformed[
                "Outlet_Location_Type"
            ].replace({"Tier 1": 2, "Tier 2": 1, "Tier 3": 0})
            self.logger.debug("ordinal variables were embedded.")
        except Exception as err:
            self.logger.error("ordinal variables couldn't be embedded "
                              f"Error: {err}")
            return None

        # 7) Embeding of nominal variables
        try:
            df_transformed = pd.get_dummies(
                df_transformed, columns=["Outlet_Type"]
            )
            if not is_train:
                # Get all the dummy variables
                df = pd.get_dummies(
                    train_data, columns=["Outlet_Type"]
                )
                dummies = [ele for ele in df.columns
                           if ele.startswith("Outlet_Type")]
                for col in dummies:
                    if col not in df_transformed.columns:
                        df_transformed[col] = False
            self.logger.debug("nominal variables were embedded.")
        except Exception as err:
            self.logger.error("nominal variables couldn't be embedded "
                              f"Error: {err}")
            return None

        # 8) Remove columns
        try:
            df_transformed = df_transformed.drop(
                columns=["Item_Identifier",
                         "Outlet_Identifier",
                         "Item_Type",
                         "Item_Fat_Content"]
            )
            self.logger.debug("columns removed.")
            return df_transformed.sort_index(axis=1)
        except Exception as err:
            self.logger.error("columns couldn't be removed. "
                              f"Error: {err}")
            return None

    def write_prepared_data(
        self, transformed_dataframe: pd.DataFrame,
        file_name: str
    ) -> None:
        """
        Write the transformed data to a CSV file.

        :param transformed_dataframe: The transformed DataFrame.
        :type transformed_dataframe: pd.DataFrame
        :param file_name: The name of the output CSV file.
        :type file_name: str
        """
        try:
            self.database.save_file(transformed_dataframe, file_name)
            self.logger.debug("data was saved sucesfully.")
        except Exception as err:
            self.logger.error("data could not be save. "
                              f"The following error was raised: {err}")
        return None

    def run(self, input_file_name: str,
            output_file_name: str, is_train: bool = False,
            train_data_file_name: str = 'nan'):
        """
        Executes the feature engineering pipeline.

        :param input_file_name: The name of the input data file.
        :type input_file_name: str
        :param output_file_name: The name of the output data file.
        :type output_file_name: str
        :param is_train: Indicates if the DataFrame is for training or inference.
        :type is_train: bool
        :param train_data_file_name: The name of the training data file.
        :type train_data_file_name: str
        """
        df = self.read_data(input_file_name)
        if df is not None:
            df_transformed = self.data_transformation(
                                    df, 
                                    is_train=is_train,
                                    train_file_name=train_data_file_name
                            )
            if df_transformed is not None:
                self.write_prepared_data(df_transformed, output_file_name)
