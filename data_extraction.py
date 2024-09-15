import os
import snowflake.connector
import pandas as pd

# Fetch credentials from environment variables
conn = snowflake.connector.connect(
    user=os.environ.get('SNOWFLAKEUSER'),
    password=os.environ.get('SNOWFLAKEPASSWORD'),
    account=os.environ.get('SNOWFLAKEACCOUNT'),
    warehouse=os.environ.get('SNOWFLAKEWAREHOUSE'),
    database=os.environ.get('SNOWFLAKEDATABASE'),
    schema=os.environ.get('SNOWFLAKESCHEMA')
)

# Execute query
query = 'SELECT * FROM WINE;'
df = pd.read_sql(query, conn)

# Save data locally
df.to_csv('data.csv', index=False)

# Close connection
conn.close()
