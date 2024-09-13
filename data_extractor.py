import os
import snowflake.connector
import pandas as pd

# Fetch credentials from environment variables
conn = snowflake.connector.connect(
    user=os.getenv('SNOWFLAKE_USER'),
    password=os.getenv('SNOWFLAKE_PASSWORD'),
    account=os.getenv('SNOWFLAKE_ACCOUNT'),
    warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
    database=os.getenv('SNOWFLAKE_DATABASE'),
    schema=os.getenv('SNOWFLAKE_SCHEMA')
)

# Execute query
query = 'SELECT * FROM WINE;'
df = pd.read_sql(query, conn)

# Save data locally
df.to_csv('data.csv', index=False)

# Close connection
conn.close()
