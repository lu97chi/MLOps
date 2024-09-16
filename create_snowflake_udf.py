# create_snowflake_udf.py

import snowflake.connector
import os

# Connect to Snowflake
conn = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    account=os.getenv("SNOWFLAKE_ACCOUNT")
)

# Create a UDF in Snowflake for model inference
cursor = conn.cursor()
cursor.execute("""
CREATE OR REPLACE FUNCTION predict_quality(features ARRAY)
RETURNS FLOAT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.8'
HANDLER = 'inference_udf'
IMPORTS = ('@my_model_stage/RandomForestClassifierModel.pkl')
PACKAGES = ('pandas', 'scikit-learn')
AS $$
import pickle
import pandas as pd
import numpy as np

# Load the model from the Snowflake stage
with open('/tmp/RandomForestClassifierModel.pkl', 'rb') as f:
    model = pickle.load(f)

def inference_udf(features):
    # Convert the input list to a DataFrame or appropriate format
    input_df = pd.DataFrame([features], columns=['feature1', 'feature2', ...])  # Replace with actual feature names
    
    # Perform inference
    prediction = model.predict(input_df)
    
    # Return the prediction
    return float(prediction[0])
$$;
""")
print("UDF for inference created in Snowflake.")
