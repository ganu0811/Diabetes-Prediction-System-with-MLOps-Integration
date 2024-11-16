import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
import mlflow
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/ganu0811/machinelearningpipeline.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'ganu0811'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '408b0d3ca30244534b813dfd774bda4b64cc1013' 

def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
    grid_search.fit(X_train, y_train)
    return grid_search


## Load the parameters from the yaml file

params = yaml.safe_load(open("params.yaml"))['train'] # Load the parameters from train key in params.yaml file


def train(data_path, model_path, random_state, n_estimators, max_depth):
    
    data=pd.read_csv(data_path)
    X =  data.drop(columns=['Outcome'])
    y = data['Outcome']
    
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    
    # Start the mlflow run
    
    with mlflow.start_run():
        
        # Split the data into training and testing sets
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,)
        
        signature = infer_signature(X_train, y_train) # Infer signature understands how the input and output data is structured

        
        ## Define Hyperparemter Grid
        
        param_grid = {
            'n_estimators': [50,100,200],
            'max_depth': [5, 10,20, None],
            'min_samples_split': [2,5,10],
            'min_samples_leaf': [1,2,4]
        }
        
        
        # Perform hyperparameter tuning
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
        
        # Get the best model
        
        best_model = grid_search.best_estimator_
        
        
        # Predict and evaluate the model
        
        y_pred = best_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        
        
        # Log the metrics
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators']) # best n_estimator value is coming from n_estimators key which is saved in best_estimator_ post hyperparmeter tuning
        mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
        mlflow.log_param("best_min_samples_split", grid_search.best_params_['min_samples_split'])
        mlflow.log_param("best_min_samples_leaf", grid_search.best_params_['min_samples_leaf'])
        
        # Log the confusion matrix and classification report
        
        confusion = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        
        mlflow.log_text(str(confusion), "confusion_matrix.txt")  # Storing the confusion matrix as text
        mlflow.log_text(classification_rep, "classification_report.txt")  # Storing the classification report as text
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(best_model, "model",  registered_model_name = "Best Model") 
        else:
            mlflow.sklearn.log_model(best_model, "model",signature = signature)
        
        
        # Create a directory to save the model
        
        os.makedirs(os.path.dirname(model_path), exist_ok = True) # The model path  is coming from the params.yaml file which will be parsed in the train function
        
        filename = model_path
        pickle.dump(best_model, open(filename, 'wb'))  # Save the model to the model path and the model name will model.pkl as per the params.yaml file
        
        print(f"Model saved at {model_path}")
        

if __name__ == "__main__":
    train(params['data'],params['model'], params['random_state'], params['n_estimators'], params['max_depth'])  # Call the train function, and the params parameter is passed to the function which will call the params.yaml file with train key