import pytest
from fastapi.testclient import TestClient
from stock_prediction_api import api
import pandas as pd

client = TestClient(api)

# Corrected URL to access the raw CSV file from GitHub
GITHUB_CSV_URL = "https://raw.githubusercontent.com/DataScientest-Studio/AUG24-BMLOPS-INT-STOCK/main/dataset/AAPL_data.csv"

# Function to download and preprocess stock data from GitHub
def download_and_preprocess_data():
    try:
        # Download the CSV file from the raw GitHub URL
        stock_data = pd.read_csv(GITHUB_CSV_URL)
        
        # Convert the 'Date' column to string format if it exists
        if 'Date' in stock_data.columns:
            stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.strftime('%Y-%m-%d')
        
        # Convert the DataFrame to a dictionary in list format (to send as JSON payload)
        return stock_data.to_dict(orient='records')
    
    except Exception as e:
        print(f"Failed to download or preprocess data: {e}")
        return []

def test_evaluate_model():
    # Download and preprocess the stock data
    stock_data = download_and_preprocess_data()

    # Ensure stock data is downloaded and preprocessed
    assert stock_data, "Stock data should not be empty"

    # Send the preprocessed data to the /evaluate endpoint
    response = client.post("/evaluate", json=stock_data)
    
    # Check if the response is successful
    assert response.status_code == 200
    
    # Verify that key evaluation metrics are present in the response
    evaluation_metrics = response.json()
    
    # Simple checks for some evaluation metrics
    assert "Test Loss" in evaluation_metrics
    assert "MSE" in evaluation_metrics
    assert "RMSE" in evaluation_metrics

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200

    # The response should be in Prometheus text format, so we'll check if it contains typical Prometheus elements
    metrics_text = response.text
    assert "# HELP" in metrics_text  # Check for help comments in Prometheus format
    assert "# TYPE" in metrics_text  # Check for type declaration in Prometheus format
    assert "python_gc_objects_collected_total" in metrics_text  # Example of Prometheus metric
