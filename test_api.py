import pytest
from fastapi.testclient import TestClient
from stock_prediction_api import api  # Ensure you import your FastAPI app instance
import numpy as np

client = TestClient(api)

# Sample stock data for testing
sample_stock_data = {
    "open": 150.0,
    "high": 155.0,
    "low": 148.0,
    "close": 152.0,
    "adjusted_close": 152.0,
    "volume": 1000000
}

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to API for Stock predictions. The API is running!"}

def test_download_stock_data():
    response = client.get("/download_stock_data?ticker=AAPL")
    assert response.status_code == 200
    assert "Data for AAPL downloaded successfully" in response.json().values()

def test_preprocess():
    response = client.post("/preprocess", json=sample_stock_data)
    assert response.status_code == 200
    assert "scaled_data" in response.json()
    assert isinstance(response.json()["scaled_data"], list)  # Check if scaled_data is a list

def test_evaluate_model(mocker):
    # Mock the loading of test data
    mocker.patch('numpy.load', side_effect=[np.array([[1]]), np.array([[1]])])  # Dummy test data
    response = client.get("/evaluate")
    assert response.status_code == 200
    print(response.json())  # Print evaluation metrics for debugging
    assert "Test Loss" in response.json()

def test_predict():
    response = client.post("/predict", json=sample_stock_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], list)  # Check if prediction is a list
    # You can add more specific checks based on the expected prediction output range

def test_retrain_model(mocker):
    # Mock the loading of train data
    mocker.patch('numpy.load', side_effect=[np.array([[1]]), np.array([[1]])])  # Dummy train data
    response = client.post("/retrain")
    assert response.status_code == 200
    assert response.json() == {"message": "Model retrained successfully"}

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    # Validate that the response is in Prometheus format
    assert isinstance(response.text, str)  # Ensure it's a string, as Prometheus returns plain text
    assert len(response.text) > 0  # Check that there is content
