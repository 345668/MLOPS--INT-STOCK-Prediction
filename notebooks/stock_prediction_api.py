from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import REGISTRY
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
import yfinance as yf
import joblib
import json
from datetime import datetime  

# Load the model and scaler
model = tf.keras.models.load_model("lstm_model.h5")
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse', 'accuracy'])
scaler = joblib.load("minmax_scaler.pkl")

# Load model metrics
with open("lstm_model_metrics.json", 'r') as f:
    metrics = json.load(f)

# Initialize FastAPI
api = FastAPI()

# Prometheus metrics
model_evaluation_counter = Counter('model_evaluations', 'Number of model evaluations')
evaluation_duration = Histogram('evaluation_duration_seconds', 'Time taken to evaluate the model')

# Pydantic model for stock data
class StockData(BaseModel):
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float  
    volume: float

# Function to log model performance
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

# API endpoints
@api.get("/")
def root():
    return {"message": "Welcome to API for Stock predictions. The API is running!"}

@api.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    return generate_latest()

@api.get("/download_stock_data")
def download_stock_data(ticker: str):
    try: 
        data = yf.download(ticker, period="10y")
        data.to_csv(f'{ticker}_data.csv')
        return {"message": f"Data for {ticker} downloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api.post("/preprocess")
def preprocess(stock_data: StockData):
    try:
        data = np.array([[stock_data.adjusted_close]])
        scaled_data = scaler.transform(data)
    
        return {"scaled_data": scaled_data.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

SUCCESS_CRITERIA = {
    "mae_threshold": 10.0
}

@api.get("/evaluate")
@evaluation_duration.time()
def evaluate_model():
    model_evaluation_counter.inc()  # Increment the counter for each evaluation
    # Load test data (replace this with actual test data)
    X_test = np.load('X_test.npy')  
    y_test = np.load('y_test.npy')  

    # Evaluate the model
    evaluation = model.evaluate(X_test, y_test, verbose=0)
    
    if evaluation[1] > SUCCESS_CRITERIA["mae_threshold"]:
        return {"message": "Model underperforms. Retraining needed!"}

    # Log model performance
    log_model_performance(evaluation)
    
    # Return evaluation results as JSON
    return {
        "Test Loss": evaluation[0],
        "Test MAE": evaluation[1],
        "Test MSE": evaluation[2]
    }

@api.post("/predict")
def predict(stock_data: StockData):
    try: 
        data = np.array([[stock_data.adjusted_close]])  
        scaled_data = scaler.transform(data)
        lstm_input = scaled_data.reshape((1, 1, scaled_data.shape[1]))
        prediction = model.predict(lstm_input)
        predicted_value = scaler.inverse_transform(prediction.reshape(1, -1))
        
        return {"prediction": predicted_value.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api.get("/metrics")
def get_metrics():
    try:
        return metrics  # Returning the metrics loaded from the JSON file
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api.post("/retrain")
def retrain_model():
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')

    model.fit(X_train, y_train, epochs=20, batch_size=32)
    model.save('lstm_model.h5')

    return {"message": "Model retrained successfully"}
