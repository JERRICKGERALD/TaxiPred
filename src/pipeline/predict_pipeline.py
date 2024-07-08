import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):

        try:
    
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path) # Import the pkl file
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,key,pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude, dropoff_latitude,passenger_count):

        self.key = key
        self.pickup_datetime = pickup_datetime
        self.pickup_longitude = pickup_longitude
        self.pickup_latitude = pickup_latitude
        self.dropoff_longitude = dropoff_longitude
        self.dropoff_latitude = dropoff_latitude
        self.passenger_count = passenger_count

    def get_data_as_df(self):
        #Returns dataframe
        try:
            custom_data_input_dict = {

                "key":[self.key],
                "pickup_datetime":[self.pickup_datetime],
                "pickup_longitude":[self.pickup_longitude],
                "pickup_latitude":[self.pickup_latitude],
                "dropoff_longitude":[self.dropoff_longitude],
                "dropoff_latitude":[self.dropoff_latitude],
                "passenger_count":[self.passenger_count]

            }

        except Exception as e:
            raise CustomException(e,sys)

        return pd.DataFrame(custom_data_input_dict)


