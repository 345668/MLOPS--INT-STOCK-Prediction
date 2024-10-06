import pytest
from fastapi.testclient import TestClient
from stock_prediction_api import api  
import numpy as np
import yfinance as yf

client = TestClient(api)

# Function to download stock data from yfinance
def download_stock_data(ticker: str, start_date: str, end_date: str):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data.reset_index().to_dict(orient='records')  # Convert to a suitable format for the API

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
    
    response = client.post("/preprocess", json=stock_data)
    print(response.json())  # Print the error response for debugging
    assert response.status_code == 200
    assert "scaled_data" in response.json()

def test_evaluate_model(mocker):
    # Mock the loading of test data
    mocker.patch('numpy.load', side_effect=[np.array([[1]]), np.array([[1]])])  # Dummy test data
    response = client.get("/evaluate")
    print(response.json())  # Print the error response for debugging
    assert response.status_code == 200
    assert "Test Loss" in response.json()

def test_predict():
    # Download stock data for the last 10 years
    stock_data = download_stock_data(ticker="AAPL", start_date="2014-01-01", end_date="2024-01-01")
    
    response = client.post("/predict", json=stock_data)
    print(response.json())  # Print the error response for debugging
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_retrain_model(mocker):
    # Mock the loading of train data
    mocker.patch('numpy.load', side_effect=[np.array([[1]]), np.array([[1]])])  # Dummy train data
    response = client.post("/retrain")
    print(response.json())  # Print the error response for debugging
    assert response.status_code == 200
    assert response.json() == {"message": "Model retrained successfully"}

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)  # Check if the response is a dictionary
    # Add more specific assertions based on the expected metrics content
