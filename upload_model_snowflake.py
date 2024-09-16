# upload_model_to_snowflake.py

import snowflake.connector
import os

# Connect to Snowflake
conn = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKEUSER"),
    password=os.getenv("SNOWFLAKEPASSWORD"),
    account=os.getenv("SNOWFLAKEACCOUNT"),
    warehouse=os.getenv("SNOWFLAKEWAREHOUSE"),  # Make sure to set the warehouse as well
    database=os.getenv("SNOWFLAKEDATABASE"),   # Set the database here
    schema=os.getenv("SNOWFLAKESCHEMA")   
)

# Create a stage in Snowflake and upload the model
cursor = conn.cursor()

cursor.execute(f"USE DATABASE {os.getenv('SNOWFLAKEDATABASE')};")
cursor.execute(f"USE SCHEMA {os.getenv('SNOWFLAKESCHEMA')};")

cursor.execute("CREATE OR REPLACE STAGE my_model_stage;")
cursor.execute("PUT file://models/RandomForestClassifierModel/model/model.pkl @my_model_stage;")
print("Model uploaded to Snowflake stage.")
