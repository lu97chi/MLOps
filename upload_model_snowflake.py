# upload_model_to_snowflake.py

import snowflake.connector
import os

# Connect to Snowflake
conn = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKEUSER"),
    password=os.getenv("SNOWFLAKEPASSWORD"),
    account=os.getenv("SNOWFLAKEACCOUNT")
)

# Create a stage in Snowflake and upload the model
cursor = conn.cursor()
cursor.execute("CREATE OR REPLACE STAGE my_model_stage;")
cursor.execute("PUT file://models/RandomForestClassifierModel.pkl @my_model_stage;")
print("Model uploaded to Snowflake stage.")
