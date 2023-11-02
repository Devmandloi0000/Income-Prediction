import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

import mlflow
import mlflow.sklearn

from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        logging.info("there may be some error in save object")
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,model):
    try:
        report = {}
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        
        scores=accuracy_score(y_test,y_pred)
        report[model]=scores
        return(report)
        
    except Exception as e:
        logging.info("there may be error in evaluate model")
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
    
scores=[]    
def dif_estimators(X_train,y_train,X_test,y_test):
    try:
        with mlflow.start_run():
        
            best_param_LR = {
                    "penalty": ['l1', 'l2'],
                    'C': [100, 10, 1, 0.1, 0.001],
                    "solver": ['lbfgs', 'liblinear']
                }

            best_param_DC = {
                    "criterion": ['gini', "entropy"],
                    "splitter": ['best', 'random'],
                    "max_depth": [1, 10, 20, 50],
                    "min_samples_split": [2, 8, 16, 20],
                    "min_samples_leaf": [1, 5, 10, 12]
                }

            best_param_SVC = {
                    "C": [1, 0.01, 0.0001],
                    "kernel": ['rbf', 'poly'],
                    "decision_function_shape": ['ovr', 'ovo']
                }

            best_param_RF = {
                    "n_estimators": [20, 200, 250, 350],
                    "criterion": ['gini', 'entropy'],
                    "max_depth": [1, 5, 10, 15, 45, 75, 150, 250],
                    "min_samples_split": [1, 5, 10, 15, 20, 25],
                    "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                }

            # Create models
            models = {
                    "LogisticRegression": LogisticRegression(),
                    "DecisionTreeClassifier": DecisionTreeClassifier(),
                    #"SVC": SVC(),
                    "RandomForestClassifier": RandomForestClassifier(),
                }
            
            for model_name, model in models.items():
                if model_name == "LogisticRegression":
                    hyperparameters = best_param_LR
                elif model_name == "DecisionTreeClassifier":
                    hyperparameters = best_param_DC
                elif model_name == "SVC":
                    hyperparameters = best_param_SVC
                elif model_name == "RandomForestClassifier":
                    hyperparameters = best_param_RF

            # Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=hyperparameters,
                    scoring="accuracy",
                    cv=5,
                )
                search.fit(X_train, y_train)

                # Evaluate the model
                y_pred = search.predict(X_test)
                accu = accuracy_score(y_test, y_pred)
                tran = search.score(X_train, y_train)
                para = search.best_params_,
                best_sc = search.best_score_

                scores.append({
                    "model": model_name,
                    "acc_score": accu,
                    "train_score": tran,
                    "best_param":para,
                    "best_sc":best_sc
                })
                
                
                mlflow.sklearn.log_model(search,"RandomizedSearchCV")
                
                results = pd.DataFrame(scores, columns=['model', 'acc_score', 'train_score','best_para','best_sc'])
                print(results)
            
        return results
        
    except  Exception as e:
        logging.info("there may be an error in dif estimator")
        raise CustomException(e,sys)
    
    