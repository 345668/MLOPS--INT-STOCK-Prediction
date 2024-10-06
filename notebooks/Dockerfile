#Base image
FROM python:3.10

#working directory
WORKDIR /app

#copying requirements file and installation of dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

#copy the API code to the container
COPY . /app

#expose the port on which the API will run
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "stock_prediction_api:api", "--host", "0.0.0.0", "--port", "8000"]