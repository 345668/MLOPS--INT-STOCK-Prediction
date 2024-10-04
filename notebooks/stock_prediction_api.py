from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
import yfinance as yf
import joblib
import json

# Load the model and scaler
model = tf.keras.models.load_model("/Users/aleksandra.rancic/Desktop/MLOps_Project/MLOps/ML_Model_API/venv/lstm_model.h5")

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse', 'accuracy'])

scaler = joblib.load('/Users/aleksandra.rancic/Desktop/MLOps_Project/MLOps/ML_Model_API/venv/minmax_scaler.pkl')

with open('/Users/aleksandra.rancic/Desktop/MLOps_Project/MLOps/ML_Model_API/venv/lstm_model_metrics.json', 'r') as f:
    metrics = json.load(f)

api = FastAPI()

class StockData(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float

def log_model_performance(evaluation, filename="model_performance_log.json"):
    try:
        with open(filename, 'r') as f:
            logs = json.load(f)
    except FileNotFoundError:
        logs = []

    logs.append({
        "test_loss": evaluation[0],
        "test_mae": evaluation[1],
        "test_mse": evaluation[2],
        "timestamp": str(datetime.now())
    })

    with open(filename, 'w') as f:
        json.dump(logs, f, indent=4)

@api.get("/evaluate")
def evaluate_model():
    # Load the test data (you need to replace this with actual test data)
    X_test = np.load('X_test.npy')  # Placeholder, replace with real data
    y_test = np.load('y_test.npy')  # Placeholder, replace with real data

    # Evaluate the model
    evaluation = model.evaluate(X_test, y_test, verbose=0)
    
    # Return the evaluation results as JSON
    return {
        "Test Loss": evaluation[0],
        "Test MAE": evaluation[1],
        "Test MSE": evaluation[2]
    }

@api.get("/")
def root():
    return {"message": "Stock prediction API is running"}

@api.get("/download_stock_data")
def download_stock_data(ticker: str):
    try: 
        data = yf.download(ticker, period="10y")
        data.to_csv(f'{ticker}_data.csv')
        return {"message": f"Data for {ticker} downloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api.post("/predict")
def predict(stock_data: StockData):
    try: 
        data = np.array([[stock_data.open, stock_data.high, stock_data.low, stock_data.close, stock_data.volume]])
        scaled_data = scaler.transform(data)
        lstm_input = scaled_data.reshape((1, 1, scaled_data.shape[1]))
        prediction = model.predict(lstm_input)
        
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api.get("/metrics")
def get_metrics():
    try:
        return metrics  # Returning the metrics loaded from the JSON file
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api.post("/preprocess")
def preprocess(stock_data: StockData):
    try:
        data = np.array([[stock_data.open, stock_data.high, stock_data.low, stock_data.close, stock_data.volume]])
        scaled_data = scaler.transform(data)
    
        return {"scaled_data": scaled_data.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))