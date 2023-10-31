import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from src.exception import CustomException
from src.logger import logging
from src.components.data_filestation import Data_File_Station

class DataIngestion:
    """
        Data File Station    :- Files path
        Data Ingestion class :- Collecting data 
        Return               :- File path of Training set and Test set
    
    """
    def __init__(self):
        logging.info("File stoage house")
        self.file_station_config = Data_File_Station()
        
    def DataIngestion_primary(self):
        logging.info("Data Ingestion process has started")
        try:
            # Importing data from file location 
            df = pd.read_csv(os.path.join("notebook/data",'adult.csv'))
            logging.info("data initiated")
            
            # Creating directory of artifacts name where file store 
            os.makedirs(os.path.dirname(self.file_station_config.raw_file),
                                                exist_ok=True)
            
            df.to_csv(self.file_station_config.raw_file,
                                                index=False)
            
            # Split data into train and test
            train_set,test_set = train_test_split(df,
                                                test_size=0.3,
                                                random_state=42)
            
            train_set.to_csv(self.file_station_config.training_file,
                                                index=False,
                                                header=True)
            
            test_set.to_csv(self.file_station_config.test_file,
                                                index=False,
                                                header=True)
            
            
            logging.info("Data Ingestion completed")
            
            # File path return training and test set 
            return (
                self.file_station_config.training_file,
                self.file_station_config.test_file
            )
                        
        except Exception as e:
            logging.info("there may be some problem in data ingestion primary")
            raise CustomException(e,sys)


