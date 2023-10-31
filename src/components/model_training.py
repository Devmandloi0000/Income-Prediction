import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from sklearn.svm import SVC

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model,dif_estimators
from src.components.data_filestation import Data_File_Station

class Model_Trainer:
    
    def __init__(self):
        self.Model_Trainer_config = Data_File_Station()
        
    def model_trainer_primary(self,train_array,test_array):
        logging.info("model trainer primary initiated")
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1] 
            )
            
            model = SVC(kernel='rbf',
                        decision_function_shape='ovr',
                        C=1)
            
            model_report = evaluate_model(X_train,y_train,X_test,y_test,model)
            logging.info(f"==="*50)
            logging.info(f"MODEL ACCURACY \n {model_report}")
            logging.info(f"==="*50)
            
            save_object(
                self.Model_Trainer_config.trained_model_file_path,
                model
            )
            
            logging.info("finally model trainined succesfully ")
            print(f"===="*25)
            print(f"Model Accuracy :- {model_report}")
            print(f"===="*25)
                   
        except Exception as e:
            logging.info("there may be error in model trainer primary")
            raise CustomException(e,sys) 

    def model_trainer_secondary(self,train_array,test_array):
        logging.info("model trainer secondary initiated")
        try:
            best_model = {}
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1] 
            )
            
            
            model_report = dif_estimators(X_train,y_train,X_test,y_test)
            logging.info(f"==="*50)
            logging.info(f"MODEL ACCURACY \n {model_report}\n")
            logging.info(f"==="*50)
        
            
            logging.info("finally model trainined succesfully ")
            print(f"===="*25)
            print(f"Model Accuracy :- {model_report}")
            print(f"===="*25)
        except Exception as e:
            logging.info("there may be error in model trainer secondary")
            raise CustomException(e,sys) 