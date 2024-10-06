import requests
import time

API_URL = "http://api:8000/evaluate"

def run_evaluation():
    try:
        response = requests.get(API_URL)
        if response.status_code == 200:
            print("Evaluation result:", response.json())
        else:
            print("Failed to evaluate model:", response.content)
    except Exception as e:
        print("Error during evaluation:", e)

if __name__ == "__main__":
    while True:
        run_evaluation()
        time.sleep(1800)  # Sleep for 30 minutes

