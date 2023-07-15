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
import os

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

    def read_data(self) -> pd.DataFrame:
        """
        Read the input data from a CSV file.

        :return: The input data as a DataFrame.
        :rtype: pd.DataFrame
        """
        pandas_df = pd.read_csv(self.input_path)
        return pandas_df

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
        df_transformed["Outlet_Establishment_Year"] = (
            2020 - df_transformed["Outlet_Establishment_Year"]
        )

        # 2) Unique labels for fat content
        df_transformed["Item_Fat_Content"] = df_transformed[
            "Item_Fat_Content"
        ].replace(
            {"low fat": "Low Fat", "LF": "Low Fat", "reg": "Regular"}
        )

        # 3) Clean missing products weight
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
                .iloc[0, 0]
            )
            df_transformed.loc[
                df_transformed["Item_Identifier"] == prod, "Item_Weight"
            ] = mode

        # 4) Missing stores size
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

        # 5) Assign None category to item_fat_content
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

        # 6) Create new categories for 'Item_Type'
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

        # 7) Assign new categories for 'Item_Fat_Content'
        df_transformed.loc[
            df_transformed["Item_Type"] == "Non perishable",
            "Item_Fat_Content",
        ] = "NA"

        # 8) Embeding products prices
        df_transformed["Item_MRP"] = pd.qcut(
            df_transformed["Item_MRP"], 4, labels=[1, 2, 3, 4]
        )

        # 9) Embeding of ordinal variables
        df_transformed["Outlet_Size"] = df_transformed[
            "Outlet_Size"
        ].replace({"High": 2, "Medium": 1, "Small": 0})
        df_transformed["Outlet_Location_Type"] = df_transformed[
            "Outlet_Location_Type"
        ].replace({"Tier 1": 2, "Tier 2": 1, "Tier 3": 0})

        # 10) Embeding of nominal variables
        df_transformed = pd.get_dummies(
            df_transformed, columns=["Outlet_Type"]
        )

        # 11) Remove columns that don't contribute to prediction
        df_transformed = df_transformed.drop(
            columns=["Item_Identifier", "Outlet_Identifier"]
        )

        return df_transformed

    def write_prepared_data(
        self, transformed_dataframe: pd.DataFrame
    ) -> None:
        """
        Write the transformed data to a CSV file.

        :param transformed_dataframe: The transformed DataFrame.
        :type transformed_dataframe: pd.DataFrame
        """
        transformed_dataframe.to_csv(self.output_path)
        return None

    def run(self):
        """
        Executes the feature engineering pipeline.
        """
        df = self.read_data()
        df_transformed = self.data_transformation(df)
        self.write_prepared_data(df_transformed)


if __name__ == "__main__":
    path_to_data_folder = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data")
    FeatureEngineeringPipeline(
        input_path=os.path.join(path_to_data_folder, 
                                "raw_dataset.csv"),
        output_path=os.path.join(path_to_data_folder, 
                                 "transformed_dataset.csv"),
    ).run()