from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application  = Flask(__name__)
app = application

#Route Creation

#home
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata',methods=['POST','GET'])
def predict_datapoint():
    
    if request.method == 'GET':
        return render_template('home.html')

    else:
        data = CustomData(

            key = request.form.get('key'),
            pickup_datetime = request.form.get('pickup_datetime'),
            pickup_longitude = request.form.get('pickup_longitude'),
            pickup_latitude = request.form.get('pickup_latitude'),
            dropoff_longitude = request.form.get('dropoff_longitude'),
            dropoff_latitude = request.form.get('dropoff_latitude'),
            passenger_count =request.form.get('passenger_count')

        )
    
        pred_df = data.get_data_as_df()
        print(pred_df)

        predict_p = PredictPipeline()
        results = predict_p.predict(pred_df)
        return render_template('home.html',results = results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")