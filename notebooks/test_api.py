import pytest
from fastapi.testclient import TestClient
from stock_prediction_api import api  
import numpy as np
import yfinance as yf
import pandas as pd

client = TestClient(api)

# Function to download stock data from yfinance and format it for API
def download_stock_data(ticker: str, start_date: str, end_date: str):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Convert Timestamps to ISO format
    stock_data['Date'] = stock_data.index
    stock_data['Date'] = stock_data['Date'].apply(lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x)
    
    return stock_data.reset_index(drop=True).to_dict(orient='records')

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Stock prediction API is running"}

def test_download_stock_data():
    response = client.get("/download_stock_data?ticker=AAPL")
    assert response.status_code == 200
    assert "Data for AAPL downloaded successfully" in response.json().values()

def test_preprocess():
    # Download stock data for the last 10 years
    stock_data = download_stock_data(ticker="AAPL", start_date="2014-01-01", end_date="2024-01-01")
    
    # Send the stock data to the API
    response = client.post("/preprocess", json=stock_data)
    print(response.json())  # Debugging output
    assert response.status_code == 200
    assert "scaled_data" in response.json()

def test_evaluate_model(mocker):
    # Mock the loading of test data
    mocker.patch('numpy.load', side_effect=[np.array([[1]]), np.array([[1]])])  # Dummy test data
    response = client.get("/evaluate")
    print(response.json())  # Debugging output
    assert response.status_code == 200
    assert "Test Loss" in response.json()

def test_predict():
    # Download stock data for the last 10 years
    stock_data = download_stock_data(ticker="AAPL", start_date="2014-01-01", end_date="2024-01-01")
    
    response = client.post("/predict", json=stock_data)
    print(response.json())  # Debugging output
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_retrain_model(mocker):
    # Mock the loading of train data
    mocker.patch('numpy.load', side_effect=[np.array([[1]]), np.array([[1]])])  # Dummy train data
    response = client.post("/retrain")
    print(response.json())  # Debugging output
    assert response.status_code == 200
    assert response.json() == {"message": "Model retrained successfully"}

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200

    # The response should be in Prometheus text format, so we'll check if it contains typical Prometheus elements
    metrics_text = response.text
    assert "# HELP" in metrics_text  # Check for help comments in Prometheus format
    assert "# TYPE" in metrics_text  # Check for type declaration in Prometheus format
    assert "python_gc_objects_collected_total" in metrics_text  # Example of Prometheus metric

