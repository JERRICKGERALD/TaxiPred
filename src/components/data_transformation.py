import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformerConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformerConfig()
        
    def get_data_transformer_object(self, df):
        try:
            logging.info("Entered the DataTransformation.get_data_transformer_object method or component")

            # Convert 'pickup_datetime' to datetime and extract components
            df['datetime'] = pd.to_datetime(df['pickup_datetime'], utc=True)
            df['weekday'] = df['datetime'].dt.weekday
            df['month'] = df['datetime'].dt.month
            df['year'] = df['datetime'].dt.year
            df['hour'] = df['datetime'].dt.hour
            df['minute'] = df['datetime'].dt.minute

            # Drop unnecessary columns
            df.drop(columns=['Unnamed: 0', 'datetime', 'pickup_datetime'], inplace=True, errors='ignore')

            # Define numerical columns for imputation pipeline (excluding dropped columns)
            numerical_columns = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
            #cat_columns = []
            # Pipeline for numerical data imputation
            num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),('scaler',StandardScaler())])
            #cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('one_hot_encoder', OneHotEncoder()),(StandardScaler())])
    
            preprocessor = ColumnTransformer([("num_pipeline",num_pipeline,numerical_columns)])#("cat_pipeline",cat_pipeline,cat_columns)])
            
            logging.info("Finished columntransformer")

            return preprocessor

        except Exception as e:

            logging.error("Failed to get the data transformer object: " + str(e))
            raise CustomException(e, sys)
        
        
    def initiate_data_transformation(self, train_path, test_path):
            
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

         
            logging.info("Read train and test data completed")
            
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object(train_df)
            
            logging.info("Obtained preprocessing object")
            
            logging.info("Applying preprocessing object on training dataframe and testing dataframe")
            

            target_column = ['fare_amount']
            
            logging.info("Applied preprocessing object on training and testing dataframe")
            
            input_train_df = train_df.drop(columns=target_column, axis=1)
            input_test_df = test_df.drop(columns=target_column, axis=1)

            target_train_df = train_df[target_column]
            target_test_df = test_df[target_column]

            logging.info("INDEPENDENT AND DEPENDENT VARIABLES SPLITTED")

            #print(input_train_df.columns)
            train_arr = preprocessing_obj.fit_transform(input_train_df)
            test_arr = preprocessing_obj.transform(input_test_df)

            logging.info("Applied preprocessing object on training and testing arrays")

            train_arr = np.c_[train_arr,np.array(target_train_df)]
            test_arr = np.c_[test_arr,np.array(target_test_df)]

            logging.info("Applied preprocessing object on training and testing arrays")

            #utils - Saved below- Purpose to save as pkl file
            save_object(

                file_path = self.data_transformation_config.preprocessor_obj_file_path,obj = preprocessing_obj
            )



            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            raise CustomException(e,sys)
            