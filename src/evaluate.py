import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import yaml
import pickle
import os
import mlflow

os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/ganu0811/machinelearningpipeline.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'ganu0811'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '408b0d3ca30244534b813dfd774bda4b64cc1013' 


## Load the parameters

params= yaml.safe_load(open('params.yaml'))['train']

def evaluate(data_path, model_path):
    
    data= pd.read_csv(data_path)
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']
    
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    
    
    # load the model from the directory
    
    model = pickle.load(open(model_path, 'rb'))
    
    # Predicting the model
    
    predictions = model.predict(X)  
    accuracy = accuracy_score(y, predictions)
    
    # Log the metrics
    
    mlflow.log_metric('accuracy', accuracy)
    print(f"Accuracy: {accuracy}")    
    


if __name__ == "__main__":
    evaluate(params['data'],params['model'])