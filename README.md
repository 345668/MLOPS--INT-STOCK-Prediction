# Stock Prediction API

## Overview

This API is built using **FastAPI** and a pre-trained **LSTM model** to predict future stock prices based on stock data features. The model predicts stock prices using features such as Open, High, Low, Close, Adjusted Close and Volume. The API also provides functionalities for downloading stock data, preprocessing input data, evaluating model performance, and returning model metrics.

## Requirements

To run this API, you need the following dependencies installed:

- Python 3.8+
- TensorFlow
- FastAPI
- NumPy
- pandas
- scikit-learn
- joblib
- yfinance
- Uvicorn

You can install the dependencies using the `requirements.txt` provided in the repository:

```bash
pip install -r requirements.txt
```

## Files:
lstm_model.h5: Pre-trained LSTM model for stock price prediction.
minmax_scaler.pkl: Scaler used for feature scaling.
lstm_model_metrics.json: JSON file containing model evaluation metrics (MSE, MAE, etc.).

## Endpoints:

1. /
Method: GET

Description: Basic endpoint to verify if the API is running.

Response:

json
Copy code
{
  "message": "Stock prediction API is running"
}

2. /evaluate
Method: GET

Description: Evaluates the model on test data.

Response:

json
Copy code
{
  "Test Loss": ,
  "Test MAE": ,
  "Test MSE": 
}

3. /download_stock_data
Method: GET

Description: Downloads stock data for the given ticker symbol from Yahoo Finance and saves it as a CSV file.

Query Parameter:

ticker (string): Stock ticker symbol (e.g., AAPL, MSFT)
Response:

json
Copy code
{
  "message": "Data for AAPL downloaded successfully"
}

4. /predict
Method: POST

Description: Predicts the stock price based on input features (open, high, low, close, adjusted_close, volume).

Input (JSON):

json
Copy code
{
  "open": ,
  "high": ,
  "low": ,
  "close": ,
  "adjusted_close": ,
  "volume": 
}
Response:

json
Copy code
{
  "prediction": [[]]
}

5. /metrics
Method: GET

Description: Returns the preloaded model metrics from lstm_model_metrics.json.

Response:

json
Copy code
{
  "MSE": ,
  "RMSE": ,
  "MAE": ,
  "MAPE": ,
  "R²": 
}

6. /preprocess
Method: POST

Description: Preprocesses the input stock data by scaling it using the preloaded scaler.

Input (JSON):

json
Copy code
{
  "open": ,
  "high": ,
  "low": ,
  "close": ,
  "adjusted_close": ,
  "volume": 
}
Response:

json
Copy code
{
  "scaled_data": [[, , , , ]]
}

## How to Run

Use virtual environment:

bash
```
python3 -m venv venv
```
```
source venv/bin/activate
```

Clone the repository to your local machine:

bash
```
git clone https://github.com/your-repo/stock-prediction-api.git
```
Navigate to the project directory:

bash
```
cd stock-prediction-api
```

Install the required dependencies:

bash
```
pip install -r requirements.txt
```
Run the FastAPI server using Uvicorn:

bash
```
uvicorn stock_prediction_api:api --reload
```
Access the API at http://127.0.0.1:8000 (you can use /docs or /redoc).

