import os
import logging
import pandas as pd
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col
from snowflake.connector.pandas_tools import write_pandas

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Snowflake connection parameters (ensure these are set in your environment or GitHub secrets)
SNOWFLAKE_CONFIG = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "role": os.getenv("SNOWFLAKE_ROLE"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA")
}

def fetch_data():
    """
    Load historical data from the local CSV file.
    """
    data_path = "wine-quality.csv"  # Path to your dataset
    try:
        data = pd.read_csv(data_path, sep=";")
        logger.info(f"Data loaded successfully from {data_path}.")
        return data
    except Exception as e:
        logger.exception(f"Error loading data from {data_path}: {e}")
        raise

def main():
    # Fetch and process data
    data = fetch_data()

    # Example of processing: fill missing values, add new features, etc.
    # Adjust the data processing steps based on your use case
    data = data.fillna(0)  # Example: filling missing values with 0

    # Save processed data to a CSV for further use in train_model.py
    processed_data_path = "processed_data.csv"
    data.to_csv(processed_data_path, index=False)
    logger.info(f"Processed data saved to {processed_data_path}.")

if __name__ == "__main__":
    main()
