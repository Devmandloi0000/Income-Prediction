import os
import pickle
from dataclasses import dataclass


@dataclass
class Data_File_Station:
    
    """_summary_:- This class is only for location of the file and file type 
        dataclass :- is used for without initializing method
    """
    # Trainig file :- Only stored Training data 
    training_file:str = os.path.join("artifacts","training.csv")
    
    # Test file :- Stored unseen data for testing 
    test_file:str = os.path.join("artifacts","test.csv")
    
    # Raw file :- Combination of training and testing data
    raw_file:str = os.path.join("artifacts","raw.csv")
    
    # Preprocessed file :- stored in pikle file ,stored all preprocessing pipeline and all
    preprocessor_file :str = os.path.join("artifacts","preprocessor.pkl")
    
    # Model pickle :- store Model which is trained
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
    # new data csv :- Store all the new predication data as a csv
    new_data_csv =os.path.join("artifacts",'new_data_file.csv')