import os #path location file
import sys #Error handling
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass #New one

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformerConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import Modeltrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

'''
The above class is used to store the path of the train, test and raw data. 
IT IS NOT CREATED - THE FOLDER ARE CREATED IN BELOW CODE

'''


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            logging.info(" Loading the data") 
            df = pd.read_csv(r'src\notebook\uber.csv') # takes data , we can 

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train - Test Split Initiated")

            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            '''
            The below is returning the path of the file train and test 
            '''
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("Exception occured at Data Ingestion stage")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_df, test_df = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_df, test_df)

    modeltrainer = Modeltrainer()
    modeltrainer.initiate_model_trainer(train_arr, test_arr)

