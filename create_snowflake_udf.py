import snowflake.connector
import os
from udf_creation import get_create_udf_sql  # Import the function

# Connect to Snowflake
conn = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKEUSER"),
    password=os.getenv("SNOWFLAKEPASSWORD"),
    account=os.getenv("SNOWFLAKEACCOUNT"),
    warehouse=os.getenv("SNOWFLAKEWAREHOUSE"),  # Set the warehouse
    database=os.getenv("SNOWFLAKEDATABASE"),    # Set the database
    schema=os.getenv("SNOWFLAKESCHEMA")         # Set the schema
)

# Create a cursor object
cursor = conn.cursor()

# Check if the model file exists in the specified stage
try:
    # List files in the stage
    cursor.execute("LIST @MLOPSTEST.PUBLIC.MY_MODEL_STAGE;")
    files = cursor.fetchall()
    
    # Check if the specific file is present
    model_filename = 'model.pkl.gz'
    file_exists = any(model_filename in file[0] for file in files)
    
    if not file_exists:
        raise FileNotFoundError(f"The file '{model_filename}' does not exist in the stage '@MLOPSTEST.PUBLIC.MY_MODEL_STAGE'.")
    
    print(f"File '{model_filename}' found in the stage. Proceeding to create the UDF.")

    # Get the SQL command from the function
    create_udf_sql = get_create_udf_sql(model_filename)
    
    # Execute the SQL command
    cursor.execute(create_udf_sql)

    print("UDF for inference created in Snowflake.")

except Exception as e:
    print(f"An error occurred: {e}")

# Close the cursor and connection
cursor.close()
conn.close()
