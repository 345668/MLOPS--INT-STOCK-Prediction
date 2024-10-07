## Stock Prediction API

### Overview

#### This API is built using FastAPI and a pre-trained LSTM model to predict future stock prices based on various stock data features. The model predicts stock prices using features such as Open, High, Low, Close, Adjusted Close, and Volume. The API provides functionalities for downloading stock data, preprocessing input data, evaluating model performance, and returning model metrics.

#### Requirements

To run this API, you need the following dependencies installed:

Python 3.8+
TensorFlow
FastAPI
NumPy
pandas
scikit-learn
joblib
yfinance
Uvicorn
Prometheus client (for monitoring)

You can install the dependencies using the requirements.txt provided in the repository:
```
pip install -r requirements.txt
```

#### Files
lstm_model.h5: Pre-trained LSTM model for stock price prediction.
minmax_scaler.pkl: Scaler used for feature scaling.
lstm_model_metrics.json: JSON file containing model evaluation metrics (MSE, MAE, etc.).

#### Endpoints
/
Method: GET
Description: Basic endpoint to verify if the API is running.
Response:
```
{
  "message": "Welcome to API for Stock predictions. The API is running!"
}
```
/metrics
Method: GET
Description: Returns Prometheus metrics for monitoring.
Response: (Plain Text)
```
# HELP model_evaluations Number of model evaluations
# TYPE model_evaluations counter
model_evaluations 0
...
```
/download_stock_data
Method: GET
Description: Downloads stock data for the given ticker symbol from Yahoo Finance and saves it as a CSV file.
Query Parameter:

ticker (string): Stock ticker symbol (e.g., AAPL, MSFT)
Response:
```
{
  "message": "Data for AAPL downloaded successfully"
}
```
/preprocess
Method: POST
Description: Preprocesses the input stock data by scaling it using the preloaded scaler.
Input (JSON):
```
{
  "open": 150.0,
  "high": 155.0,
  "low": 148.0,
  "close": 152.0,
  "adjusted_close": 152.0,
  "volume": 1000000
}
```
Response:

```
{
  "scaled_data": [[0.7]]  // #Example scaled data // #The API will give you only scaled value for Adjusted Close price.
}
```
/evaluate
Method: GET
Description: Evaluates the model on test data and logs performance metrics.
Response:

```
{
  "Test Loss": 0.025,
  "Test MAE": 2.5,
  "Test MSE": 0.000625
}
```
/predict
Method: POST
Description: Predicts the stock price based on input features.
Input (JSON):

```
{
  "open": 150.0,
  "high": 155.0,
  "low": 148.0,
  "close": 152.0,
  "adjusted_close": 152.0,
  "volume": 1000000
}
```
Response:

```
{
  "prediction": [[153.0]]  // #Example predicted price - Adjusted Close
}
```
/retrain
Method: POST
Description: Retrains the LSTM model using training data.
Response:

```
{
  "message": "Model retrained successfully"
}
```

#### How to Run
Use a virtual environment:

```
python3 -m venv venv
```
```
source venv/bin/activate
```
Clone the repository to your local machine:

```
git clone https://github.com/your-repo/stock-prediction-api.git
```
Navigate to the project directory:

```
cd stock-prediction-api
```
Install the required dependencies:

```
pip install -r requirements.txt
```
Run the FastAPI server using Uvicorn:

```
uvicorn stock_prediction_api:api --host 0.0.0.0 --port 8000 --reload
```
Access the API at http://127.0.0.1:8000 (you can use /docs for the Swagger UI).

#### Docker Setup
To run the API in a Docker container, use the following Dockerfile:

Dockerfile
```
# Use the official Python image
FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "stock_prediction_api:api", "--host", "0.0.0.0", "--port", "8000"]
```
#### Testing the API
Here are the test cases using pytest to ensure that your API endpoints are working correctly.

```
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
    assert isinstance(response.json()["scaled_data"], list)

def test_evaluate_model(mocker):
    mocker.patch('numpy.load', side_effect=[np.array([[1]]), np.array([[1]])])  # Dummy test data
    response = client.get("/evaluate")
    assert response.status_code == 200
    assert "Test Loss" in response.json()

def test_predict():
    response = client.post("/predict", json=sample_stock_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], list)

def test_retrain_model(mocker):
    mocker.patch('numpy.load', side_effect=[np.array([[1]]), np.array([[1]])])  # Dummy train data
    response = client.post("/retrain")
    assert response.status_code == 200
    assert response.json() == {"message": "Model retrained successfully"}

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert isinstance(response.text, str)  # Ensure it's a string
    assert len(response.text) > 0  # Check that there is content
```
