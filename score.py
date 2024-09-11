import json
import numpy as np
import joblib

def init():
    global model
    # Load the model from the registered path
    model_path = "model/model.pkl"  # Adjust the path based on your setup
    model = joblib.load(model_path)

def run(data):
    try:
        # Assume the data comes in as a JSON array
        input_data = np.array(json.loads(data)["data"])
        # Perform prediction
        predictions = model.predict(input_data)
        # Return predictions as a JSON response
        return json.dumps({"predictions": predictions.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
