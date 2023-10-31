import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.components.data_filestation import Data_File_Station
import pandas as pd

class PredictionPipeline:
    def __init__(self):
        self.prediction_pipeline_config = Data_File_Station()
    
    def Predict(self,feature):
        logging.info("prediction for new data has started")
        try:
            preprocess_pkl = self.prediction_pipeline_config.preprocessor_file
            model_pkl = self.prediction_pipeline_config.trained_model_file_path
            
            preprocessor=load_object(preprocess_pkl)
            model=load_object(model_pkl)
            
            data_scaled = preprocessor.transform(feature)
            
            pred = model.predict(data_scaled)
            return pred
            
        except Exception as e:
            logging.info("there may be some problem in Predict")
            raise CustomException(e,sys)
        
        
        
class CustomData:
    def __init__ (self,
                 age:float,
                 workclass:str,
                 education_num:float,
                 occupation:str,
                 race:str,
                 sex:str,
                 capital_gain:float,	
                 capital_loss:float,	
                 hours_per_week:float,
                 country:str):
        self.age = age
        self.workclass = workclass
        self.education_num = education_num
        self.occupation = occupation
        self.race = race
        self.sex = sex
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        self.country = country
        
    def get_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "age":[self.age],
                "workclass":[self.workclass], 
                "education_num":[self.education_num],
                "occupation":[self.occupation],
                "race":[self.race], 
                "sex":[self.sex], 
                "capital_gain":[self.capital_gain],
                "capital_loss":[self.capital_loss], 
                "hours_per_week":[self.hours_per_week], 
                "country":[self.country] 
                }
            
            df = pd.DataFrame(custom_data_input_dict)
            return df
            
        except Exception as e:
            logging.info("there may be some problem in get as data frame")
            raise CustomData(e,sys)
        
        