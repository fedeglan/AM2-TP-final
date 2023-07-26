import subprocess
import os
import argparse

# Default parameters for running the pipeline
path_to_data_folder = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data")
input_path_placeholder = os.path.join(path_to_data_folder, 
                                      "Test_BigMart.csv")
transformed_path_placeholder = os.path.join(path_to_data_folder, 
                                       "test_data_transformed.csv")
output_path_placeholder = os.path.join(path_to_data_folder, 
                                       "predictions.csv")
model_path_placeholder = os.path.join(path_to_data_folder, 
                                       "model.pickle")

if __name__ == "__main__":
    # Add params to parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path of the input data")
    parser.add_argument("--transformed_path", help="Path where to save the transformed data")
    parser.add_argument("--model_path", help="Path where to save the trained model's pickle")
    parser.add_argument("--output_path", help="Path where to save the predictions")
    
    # Get params from console
    args = parser.parse_args()
    if args.input_path is None:
        input_path = input_path_placeholder
    else:
        input_path = args.input_path
    
    if args.transformed_path is None:
        transformed_path = transformed_path_placeholder
    else:
        transformed_path = args.transformed_path

    if args.model_path is None:
        model_path = model_path_placeholder
    else:
        model_path = args.model_path
    
    if args.output_path is None:
        output_path = output_path_placeholder
    else:
        output_path = args.output_path

subprocess.run(['Python', 'feature_engineering.py', input_path, transformed_path])

subprocess.run(['Python', 'predict.py', transformed_path, model_path, output_path])