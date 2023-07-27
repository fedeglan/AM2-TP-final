"""
inference_pipeline.py

Pipeline for performing inference on input data.

DESCRIPTION: This script runs the feature_engineering.py and
predict.py scripts subsequently in order to generate a series
of predictions based on an input dataset.
AUTHOR: Federico Glancszpigel
DATE: 26/7/2023
"""

# Package imports
import sys
import os
import argparse

# Local imports
sys.path.append(os.path.dirname(__file__))
from predict import MakePredictionPipeline
from feature_engineering import FeatureEngineeringPipeline

# Default parameters for running the pipeline
folder_path_placeholder = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data")
input_file_name_placeholder = "Test_BigMart.csv"
transformed_data_file_name_placeholder = "test_data_transformed.csv"
train_data_file_name_placeholder = "Train_BigMart.csv"
model_file_name_placeholder = "model.pickle"
output_file_name_placeholder = "predictions.csv"

if __name__ == "__main__":
    # Add params to parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder_path",
                help="Path to data folder where the input data is stored")
    parser.add_argument("--input_file_name",
                help="File name of the input data")
    parser.add_argument("--transformed_data_file_name",
                help="File name to save the transformed data")
    parser.add_argument("--train_data_file_name",
                help="File name where the training data is stored")
    parser.add_argument("--model_file_name",
                help="File name to save the trained model's pickle")
    parser.add_argument("--output_file_name",
                help="File name to save the model's predictions")
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

    if args.transformed_data_file_name is None:
        transformed_data_file_name =\
              transformed_data_file_name_placeholder
    else:
        transformed_data_file_name = args.transformed_data_file_name

    if args.train_data_file_name is None:
        train_data_file_name = train_data_file_name_placeholder
    else:
        train_data_file_name = args.train_data_file_name

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
    fe_pipeline_obj.run(input_file_name, 
                        transformed_data_file_name,
                        is_train=False, 
                        train_data_file_name=train_data_file_name)

    # Generate and run MakePredictionPipeline
    pred_pipeline_obj = MakePredictionPipeline(folder_path)
    pred_pipeline_obj.run(transformed_data_file_name,
                          model_file_name, output_file_name)
