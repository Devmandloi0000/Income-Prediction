from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import Data_Transformation
from src.components.model_training import Model_Trainer
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    obj = DataIngestion()
    X_train_path,X_test_path=obj.DataIngestion_primary()
    obj1 = Data_Transformation()
    train_array,test_array,_=obj1.Data_Transformation_primary(X_train_path,X_test_path)
    obj3 = Model_Trainer()
    obj3.model_trainer_primary(train_array,test_array)
    obj3.model_trainer_secondary(train_array,test_array)