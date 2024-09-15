import os
from snowflake.snowpark import Session
from snowflake.snowpark.functions import udf
import joblib
import pandas as pd

# Snowflake connection parameters
connection_parameters = {
    'account': os.environ.get('SNOWFLAKEACCOUNT'),
    'user': os.environ.get('SNOWFLAKEUSER'),
    'password': os.environ.get('SNOWFLAKEPASSWORD'),
    'role': os.environ.get('SNOWFLAKEROLE'),
    'warehouse': os.environ.get('SNOWFLAKEWAREHOUSE'),
    'database': os.environ.get('SNOWFLAKEDATABASE'),
    'schema': os.environ.get('SNOWFLAKESCHEMA')
}

session = Session.builder.configs(connection_parameters).create()

# Load model
model = joblib.load('model.joblib')

# Define UDF
def predict_udf(*features):
    df = pd.DataFrame([features])
    prediction = model.predict(df)
    return prediction[0]

# Register UDF
session.udf.register(
    func=predict_udf,
    name='predict',
    replace=True,
    is_permanent=True,
    input_types=[FloatType()] * number_of_features,
    return_type=FloatType(),
    packages=['scikit-learn', 'joblib', 'pandas']
)
