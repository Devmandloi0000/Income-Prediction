import pandas as pd
import numpy as np
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_filestation import Data_File_Station

from pymongo import MongoClient

#url="mongodb+srv://devmandloi37:Devendra123456789@cluster0.rkgp8jf.mongodb.net/?retryWrites=true&w=majority"
#client = MongoClient(url)



class MGDB:
    def __init__(self):
        pass
        
    
    def Mongo_connection(self,id='devmandloi37',password='devmandloi'):
        try:
            url = f"mongodb+srv://{id}:{password}@cluster0.rkgp8jf.mongodb.net/?retryWrites=true&w=majority"
            client=MongoClient(url)
            logging.info('connection completed')
            
            DB= client['Income_prediction']
            col=DB['data_record']
            
            return col
        except Exception as e:
            logging.info("there may be some error in mongo connection")
            raise CustomException(e,sys)
        
        
    def Insertion(self,data):
        try:
            collection=self.Mongo_connection()
            logging.info("connection setup sucess")
            collection.insert_one(data)
            logging.info("inserting of data is completed")
            
            #extraction as csv
            data_list = list(collection.find())
            self.new_file_location = Data_File_Station()
            
            # store new test data as csv
            if data_list:
                data_dict_list = [x for x in data_list]
                new_csv=pd.DataFrame(data_dict_list,columns=['age','workclass','education_number','occupation','race','sex','capital_gain','capital_loss','hours_per_week','country'])
                new_csv.to_csv(self.new_file_location.new_data_csv,index=False)
                logging.info("new data is added to csv file")
            #print(new_csv)
        except Exception as e:
            logging.info("there may be some error in mongo connection")
            raise CustomException(e,sys)
        
    def Extraction(self,data):
        try:
            collection=self.Mongo_connection()
            logging.info("connection setup sucess")
            #collection.find
            
        except Exception as e:
            logging.info("there may be some error in the extraction method")
            raise CustomException(e,sys)
        
        
        
        
        
