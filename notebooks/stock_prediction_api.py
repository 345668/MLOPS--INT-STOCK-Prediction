#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from flask import Flask, request, jsonify
from keras.models import load_model
import joblib
import yfinance as yf
from datetime import datetime
import json

app = Flask(__name__)

# Define folder path to store datasets
folder_path = "/Users/aleksandra.rancic/Desktop/MLOps/dataset"

# Load the trained LSTM model and the MinMaxScaler
model = load_model('lstm_model.h5')
scaler = joblib.load('minmax_scaler.pkl')

# Function to download the dataset
def download_dataset(label, start_date):
    end_date = datetime.now().strftime('%Y-%m-%d')
    stock_data = yf.download(label, start=start_date, end=end_date)
    file_name = f"{label}_stock_data.csv"
    file_path = os.path.join(folder_path, file_name)
    stock_data.to_csv(file_path)
    return stock_data

# Load evaluation metrics from the JSON file
def load_evaluation_metrics():
    with open('lstm_model_metrics.json', 'r') as file:
        metrics = json.load(file)
    return metrics

# Endpoint to check API status
@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "API is running"}), 200

# Endpoint to download stock data and preprocess
@app.route('/download', methods=['POST'])
def download_stock_data():
    data = request.get_json(force=True)
    label = data.get('label', 'AAPL')
    start_date = data.get('start_date', '2014-09-12')
    
    # Download stock data
    stock_data = download_dataset(label, start_date)
    
    return jsonify({"message": f"Stock data for {label} downloaded successfully!"}), 200

# Endpoint to predict stock prices
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Get the input features from the request
    input_data = np.array(data['features']).reshape(-1, 1)

    # Scale the input data using the loaded scaler
    scaled_input = scaler.transform(input_data)

    # Prepare the input data for LSTM
    sequence_length = 60
    x_input = []
    for i in range(len(scaled_input) - sequence_length):
        x_input.append(scaled_input[i:i + sequence_length])

    x_input = np.array(x_input)

    # Make predictions using the LSTM model
    predictions = model.predict(x_input)

    # Inverse transform the predictions to get the actual stock prices
    predicted_values = scaler.inverse_transform(predictions)

    return jsonify({"prediction": predicted_values.tolist()}), 200

# Endpoint to get the model evaluation metrics
@app.route('/evaluation', methods=['GET'])
def evaluation():
    metrics = load_evaluation_metrics()
    return jsonify(metrics), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
