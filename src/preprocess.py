import pandas as pd
import sys
import yaml
import os

# Load the parameters from the yaml file

params = yaml.safe_load(open("params.yaml"))['preprocess']

def preprocess(input_path, output_path):
    
    data = pd.read_csv(input_path)
    
    os.makedirs(os.path.dirname(output_path), exist_ok = True)  # Create the output directory for the preprocessed data
    data.to_csv(output_path, header=None, index=False)  # Save the preprocessed data to the output directory
    print(f"Preprocessed data saved at {output_path}")


if __name__ == "__main__":
    preprocess(params["input"], params["output"])    # Call the preprocess function, and the params parameter is passed to the function which will call the params.yaml file with preprocess key
    
    