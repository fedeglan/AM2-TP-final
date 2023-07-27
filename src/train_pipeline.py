"""
train_pipeline.py

Pipeline for training the model over a training dataset.

DESCRIPTION: This script runs the feature_engineering.py and
train.py scripts subsequently in order to train a ML model, and
then save it as a pickle file.
AUTHOR: Federico Glancszpigel
DATE: 26/7/2023
"""

# Package imports
from train import ModelTrainingPipeline
from feature_engineering import FeatureEngineeringPipeline
import sys
import os
import argparse

# Local imports
sys.path.append(os.path.dirname(__file__))

# Default parameters for running the pipeline
folder_path_placeholder = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data")
input_file_name_placeholder = "Train_BigMart.csv"
output_file_name_placeholder = "transformed_train_data.csv"
model_file_name_placeholder = "model.pickle"

if __name__ == "__main__":
    # Add params to parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder_path",
            help="Path to data folder where the input data is stored.")
    parser.add_argument("--input_file_name",
            help="File name of the input data")
    parser.add_argument("--output_file_name",
            help="File name to save the transformed data")
    parser.add_argument("--model_file_name",
            help="File name to save the trained model's pickle")
    args = parser.parse_args()

    # Get params from console
    args = parser.parse_args()
    if args.data_folder_path is None:
        folder_path = folder_path_placeholder
    else:
        folder_path = args.data_folder_path

    if args.input_file_name is None:
        input_file_name = input_file_name_placeholder
    else:
        input_file_name = args.input_file_name

    if args.output_file_name is None:
        output_file_name = output_file_name_placeholder
    else:
        output_file_name = args.output_file_name

    if args.model_file_name is None:
        model_file_name = model_file_name_placeholder
    else:
        model_file_name = args.model_file_name

    # Generate and run FeatureEngineeringPipeline
    fe_pipeline_obj = FeatureEngineeringPipeline(folder_path)
    fe_pipeline_obj.run(input_file_name, output_file_name, is_train=True)

    # Generate and run ModelTrainingPipeline
    train_pipeline_obj = ModelTrainingPipeline(folder_path)
    train_pipeline_obj.run(output_file_name, model_file_name)
