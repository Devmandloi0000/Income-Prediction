import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.exception import CustomException
from src.logger import logging
from src.components.data_filestation import Data_File_Station
from src.utils import save_object

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler


class Data_Transformation:
    """
    
    Data_Transformation :- Class handle data and return preprocess data as a pickle file
    Data_Transformation contain methods :-
                                            Data_Transformation_primary         :- Segregatting depedent and indepedent data and futher process are done by the other methods 
                                            Data_Transformation_secondary       :- Replacing and handlling missing values ,outliers and return clean data or we can say cleaning process is done inside this method
                                            Data_Transformation_preprocessed    :- Setting up Pipeline of the data                                       
    """
    
    def __init__(self):
        self.file_station_config=Data_File_Station()
        
    def Data_Transformation_preprocessed(self):
        """ Data_Transformation_preprocessed are used for preprocessing of the data and return preprocessed pipeline"""
        
        logging.info("data preprocessed has started")
        try:
            cate_col= ['workclass', 'occupation', 'race', 'sex', 'country']
            num_col = ['age', 'education_num', 'capital_gain', 'capital_loss','hours_per_week']
            
            report={
                    'workclass': ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov','Local-gov', 'Self-emp-inc', 'Without-pay'],
                    'occupation': ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners','Prof-specialty', 'Other-service', 'Sales', 'Craft-repair','Transport-moving', 'Farming-fishing', 'Machine-op-inspct','Tech-support', 'Protective-serv', 'Armed-Forces','Priv-house-serv'],
                    'race': ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo','Other'],
                    'sex': ['Male', 'Female'],
                    'country': ['United-States', 'Cuba', 'Jamaica', 'India', 'Mexico','Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran','Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand','Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal','Dominican-Republic', 'El-Salvador', 'France', 'Guatemala', 'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru','Scotland', 'Trinadad&Tobago', 'Greece', 'Nicaragua', 'Vietnam','Hong', 'Ireland', 'Hungary', 'Holand-Netherlands']
                    }
            
            num_pipeline = Pipeline(
                                    steps=[
                                        ("imputer",SimpleImputer(strategy='median')),
                                        ("scaler",StandardScaler())
                                        ]
                                    )


            cate_pipeline = Pipeline(
                                    steps=[
                                        ("imputer",SimpleImputer(strategy='most_frequent')),
                                        ("ordinal_encoder",OrdinalEncoder(categories=[report['workclass'],report['occupation'],report['race'],report['sex'],report['country']])),
                                        ("scaler",StandardScaler())
                                        ]
                                    )
            
            preprocessor = ColumnTransformer([
                                        ("num_pipeline",num_pipeline,num_col),
                                        ("cate_pipeline",cate_pipeline,cate_col)
                                        ]
                                    )
                                
            logging.info("data preprocessed has completed")
            return (preprocessor)
            
        except Exception as e:
            logging.info(("there may be error in data transformation preprocessed"))
            raise CustomException(e,sys)    
    
        
    def Data_Transformation_secondary(self,data):
        """ Data_Transformation_secondary are used for cleaning of the data handling missing value , nan vaue and others and return cleaned data """
        
        logging.info("Data Transforamtion process has started")
        try:
            logging.info('Data Transformation secondary initiated')
            # Used for replacing marks with NaN value
            for req in data:
                data[req].replace(" ?",np.NaN,inplace=True)
                
            # Droping all the null value in workclass and occupation
            data.dropna(subset=['workclass','occupation'],inplace=True)
            
            data.rename(columns={'education-num':"education_num",
                   "marital-status":"marital_status",
                   "capital-gain":"capital_gain",
                   "capital-loss":"capital_loss",
                   "hours-per-week":"hours_per_week"
                   },inplace=True)
            
            # Replacing null value by mode
            val = str(data['country'].mode())
            data['country'].fillna(val,inplace=True)
            
            # Mapping or replaceing with other words
            data['country']=data['country'].replace({"United-States":" United-States",
                                     "0     United-States\nName: country, dtype: object":" United-States",
                                     " Outlying-US(Guam-USVI-etc)":" United-States"})
            
            # Age column contain outliers so replacing with them upper limit and lower limit
            q1= data['age'].quantile(0.10)
            q3= data["age"].quantile(0.75)
            IQR=q3-q1
            upper_limit = q3 + 1.5 * IQR
            lower_limit = q1 - 1.5 * IQR
            data.loc[(data["age"] > upper_limit), "age"]=upper_limit
            data.loc[(data["age"] < lower_limit), "age"]=lower_limit
            
            # There are some additional spaces in every words 
            for i in data:
                if data[i].dtypes == "O":
                    data[i]=data[i].str.replace(" ","")
                    
            report = {}
            for i in data:
                a=data[i].unique()
                report[i]=a
                
            cate_col=data.select_dtypes(include='O').columns
            num_col=data.select_dtypes(exclude='O').columns
            
            return data
                        
        except Exception as e:
            logging.info("there may be some error in data transforamtion secondary")
            raise CustomException(e,sys)
        
        
    def Data_Transformation_primary(self,
                                    train_path,
                                    test_path):
        logging.info("Data Transforamtion primary process has started")
        try:
            # Method 
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Data cleaning and preprocessing
            train_df = self.Data_Transformation_secondary(train_df)
            test_df = self.Data_Transformation_secondary(test_df)
            
            # Indepedent and dependent variarible and here negative corr column are also removed
            target_col = 'salary'
            drop_col = [target_col,
                        'education',
                        'fnlwgt',
                        'marital_status',
                        'relationship']
            
            input_x_train = train_df.drop(columns=drop_col,
                                          axis=1)
            input_y_train = train_df[target_col]
            input_x_test = test_df.drop(columns=drop_col,
                                          axis=1)
            input_y_test = test_df[target_col]
            
            #logging.info(f"{input_x_train}")            
            #input_x_train = self.Data_Transformation_secondary(input_x_train)
            #input_x_test = self.Data_Transformation_secondary(input_x_test)
            
            preprocessor_obj = self.Data_Transformation_preprocessed()
            
            input_x_train_arr = preprocessor_obj.fit_transform(input_x_train)
            input_x_test_arr = preprocessor_obj.transform(input_x_test)
            
            
            
            train_array = np.c_[input_x_train_arr,np.array(input_y_train)]
            test_array = np.c_[input_x_test_arr,np.array(input_y_test)]
            
            #logging.info(f'{train_array,test_array}')
            
            save_object(
                file_path=self.file_station_config.preprocessor_file,
                obj=preprocessor_obj
            )
            
            logging.info(f'data saved inside the preprocessed object')
            logging.info(f'preproceeing finished')
            
            return (
                train_array,
                test_array,
                self.file_station_config.preprocessor_file
            )
            
        except Exception as e:
            logging.info("there may be some error in data transforamtion primary")
            raise CustomException(e,sys)