import os
import snowflake.connector
import pandas as pd

# Fetch credentials from environment variables
conn = snowflake.connector.connect(
    user=os.getenv('SNOWFLAKEUSER'),
    password=os.getenv('SNOWFLAKEPASSWORD'),
    account=os.getenv('SNOWFLAKEACCOUNT'),
    warehouse=os.getenv('SNOWFLAKEWAREHOUSE'),
    database=os.getenv('SNOWFLAKEDATABASE'),
    schema=os.getenv('SNOWFLAKESCHEMA')
)

# Execute query
query = 'SELECT * FROM WINE;'
df = pd.read_sql(query, conn)

# Save data locally
df.to_csv('data.csv', index=False)

# Close connection
conn.close()
