# Use the official Python slim image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt to the working directory
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for the Flask app
EXPOSE 5000

# Set environment variable for Flask (optional)
ENV FLASK_ENV=production

# Copy the model, scaler, and evaluation metrics to the working directory
COPY lstm_model.h5 minmax_scaler.pkl lstm_model_metrics.json /app/

# Copy the rest of your application code to the working directory
COPY . .

# Command to run the application
CMD ["python", "stock_prediction_api.py"]
