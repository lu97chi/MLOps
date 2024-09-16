# udf_creation.py

def get_create_udf_sql(model_filename):
    return f"""
    CREATE OR REPLACE FUNCTION predict_quality(features ARRAY)
    RETURNS FLOAT
    LANGUAGE PYTHON
    RUNTIME_VERSION = '3.8'
    HANDLER = 'inference_udf'
    IMPORTS = ('@MLOPSTEST.PUBLIC.MY_MODEL_STAGE/{model_filename}')  -- Correctly reference the file from the stage
    PACKAGES = ('pandas', 'scikit-learn')
    AS $$
import sys
import os
import pickle
import pandas as pd
import numpy as np
import gzip

# Use Snowflake's special import directory to locate the file
import_dir = sys._xoptions["snowflake_import_directory"]
model_path = os.path.join(import_dir, '{model_filename}')

# Load the model from the imported file
with gzip.open(model_path, 'rb') as f:
    model = pickle.load(f)

def inference_udf(features):
    # Use the actual feature names from your dataset
    feature_names = [
        'fixed acidity', 
        'volatile acidity', 
        'citric acid', 
        'residual sugar', 
        'CHLORIDES', 
        'free sulfur dioxide', 
        'total sulfur dioxide', 
        'DENSITY', 
        'PH', 
        'SULPHATES', 
        'ALCOHOL'
    ]
    
    # Convert the input list to a DataFrame or appropriate format
    input_df = pd.DataFrame([features], columns=feature_names)
    
    # Perform inference
    prediction = model.predict(input_df)
    
    # Return the prediction
    return float(prediction[0])
    $$;
    """
