"""
feature_engineering.py

Module for performing feature engineering on input data.

DESCRIPTION: This module contains a FeatureEngineeringPipeline
    class that implements various data transformation steps to 
    preprocess the input data for machine learning tasks.
AUTHOR: Federico Glancszpigel
DATE: 15/7/2023
"""

# Imports
import pandas as pd
import copy
import logging
import argparse

class FeatureEngineeringPipeline(object):
    def __init__(self, input_path, output_path):
        """
        Initialize the FeatureEngineeringPipeline object.

        :param input_path: Path to the input data file.
        :type input_path: str
        :param output_path: Path to save the transformed data.
        :type output_path: str
        """
        self.input_path = input_path
        self.output_path = output_path

        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        format = '%(asctime)s - %(levelname)s - FeatureEngineering - %(message)s'
        formatter = logging.Formatter(format, datefmt='%Y-%m-%d %H:%M:%S')
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def read_data(self) -> pd.DataFrame:
        """
        Read the input data from a CSV file.

        :return: The input data as a DataFrame.
        :rtype: pd.DataFrame
        """
        try:
            pandas_df = pd.read_csv(self.input_path)
            self.logger.debug("data was read succesfully.")
            return pandas_df
        except Exception as err:
            self.logger.error("data could not be read. "
                              f"The following error was raised: {err}")
            return None

    def data_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data transformations to the input DataFrame.

        :param df: The input DataFrame.
        :type df: pd.DataFrame
        :return: The transformed DataFrame.
        :rtype: pd.DataFrame
        """
        df_transformed = copy.deepcopy(df)

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

        # 2) Unique labels for fat content
        try:
            df_transformed["Item_Fat_Content"] = df_transformed[
                "Item_Fat_Content"
            ].replace(
                {"low fat": "Low Fat", "LF": "Low Fat", "reg": "Regular"}
            )
            self.logger.debug("unique labels were created " 
                              "for Item_Fat_Content")
        except Exception as err:
            self.logger.error("unique labels for Item_Fat_Content "
                              f"couldn't be created. Error: {err}")
            return None

        # 3) Clean missing products weight
        try:
            products = list(
                df_transformed[df_transformed["Item_Weight"].isnull()][
                    "Item_Identifier"
                ].unique()
            )
            for prod in products:
                mode = (
                    (
                        df_transformed[
                            df_transformed["Item_Identifier"] == prod
                        ][["Item_Weight"]]
                    )
                    .mode()
                )
                if not mode.empty:
                    mode = mode.iloc[0,0]
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

        # 5) Assign None category to item_fat_content
        try:
            df_transformed.loc[
                df_transformed["Item_Type"] == "Household", "Item_Fat_Content"
            ] = "NA"
            df_transformed.loc[
                df_transformed["Item_Type"] == "Health and Hygiene",
                "Item_Fat_Content",
            ] = "NA"
            df_transformed.loc[
                df_transformed["Item_Type"] == "Hard Drinks",
                "Item_Fat_Content",
            ] = "NA"
            df_transformed.loc[
                df_transformed["Item_Type"] == "Soft Drinks",
                "Item_Fat_Content",
            ] = "NA"
            df_transformed.loc[
                df_transformed["Item_Type"] == "Fruits and Vegetables",
                "Item_Fat_Content",
            ] = "NA"
            self.logger.debug("none category was added " 
                              "to Item_Fat_Content.")
        except Exception as err:
            self.logger.error("none category couldn't be added "
                              "to Item_Fat_Content. "
                              f"Error: {err}")
            return None

        # 6) Create new categories for 'Item_Type'
        try:
            df_transformed["Item_Type"] = df_transformed["Item_Type"].replace(
                {
                    "Others": "Non perishable",
                    "Health and Hygiene": "Non perishable",
                    "Household": "Non perishable",
                    "Seafood": "Meats",
                    "Meat": "Meats",
                    "Baking Goods": "Processed Foods",
                    "Frozen Foods": "Processed Foods",
                    "Canned": "Processed Foods",
                    "Snack Foods": "Processed Foods",
                    "Breads": "Starchy Foods",
                    "Breakfast": "Starchy Foods",
                    "Soft Drinks": "Drinks",
                    "Hard Drinks": "Drinks",
                    "Dairy": "Drinks",
                }
            )
            self.logger.debug("new categories were created for Item_Type.")
        except Exception as err:
            self.logger.error("new categories couldn't be added "
                              "to Item_Type. "
                              f"Error: {err}")
            return None

        # 7) Assign new categories for 'Item_Fat_Content'
        try:
            df_transformed.loc[
                df_transformed["Item_Type"] == "Non perishable",
                "Item_Fat_Content",
            ] = "NA"
            self.logger.debug("new categories were assigned " 
                              "for Item_Fat_Content.")
        except Exception as err:
            self.logger.error("new categories couldn't be assigned "
                              "to Item_Fat_Content. "
                              f"Error: {err}")
            return None

        # 8) Embeding products prices
        try:
            df_transformed["Item_MRP"] = pd.qcut(
                df_transformed["Item_MRP"], 4, labels=[1, 2, 3, 4]
            )
            self.logger.debug("product prices were embedded.")
        except Exception as err:
            self.logger.error("product prices couldn't be embedded "
                              f"Error: {err}")
            return None

        # 9) Embeding of ordinal variables
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

        # 10) Embeding of nominal variables
        try:
            df_transformed = pd.get_dummies(
                df_transformed, columns=["Outlet_Type"]
            )
            self.logger.debug("nominal variables were embedded.")
        except Exception as err:
            self.logger.error("nominal variables couldn't be embedded "
                              f"Error: {err}")
            return None

        # 11) Remove columns that don't contribute to prediction
        try:
            df_transformed = df_transformed.drop(
                columns=["Item_Identifier", "Outlet_Identifier"]
            )
            self.logger.debug("Item_Identifier and Outlet_Identifier "
                              "removed.")
        except Exception as err:
            self.logger.error("Item_Identifier and Outlet_Identifier "
                              f"couldn't be removed. Error: {err}")
            return None

        # 12) Remove Item_Type & Item_Fat_Content
        try:
            df_transformed = df_transformed.drop(
                columns=["Item_Type", "Item_Fat_Content"]
            )
            self.logger.debug("Item_Type and Item_Fat_Content "
                              "removed.")
            return df_transformed
        except Exception as err:
            self.logger.error("Item_Type and Item_Fat_Content "
                              f"couldn't be removed. Error: {err}")
            return None

    def write_prepared_data(
        self, transformed_dataframe: pd.DataFrame
    ) -> None:
        """
        Write the transformed data to a CSV file.

        :param transformed_dataframe: The transformed DataFrame.
        :type transformed_dataframe: pd.DataFrame
        """
        try:
            transformed_dataframe.to_csv(self.output_path)
            self.logger.debug("data was saved sucesfully.")
        except Exception as err:
            self.logger.error("data could not be save. "
                              f"The following error was raised: {err}")
        return None

    def run(self):
        """
        Executes the feature engineering pipeline.
        """
        df = self.read_data()
        if df is not None:
            df_transformed = self.data_transformation(df)
            if df_transformed is not None:
                self.write_prepared_data(df_transformed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path of the input data")
    parser.add_argument("output_path", help="Path where to save the transformed data")
    args = parser.parse_args()

    FeatureEngineeringPipeline(
        input_path=args.input_path,
        output_path=args.output_path,
    ).run()