import os
import snowflake.connector
import pandas as pd

print (os.environ.get('SNOWFLAKEUSER'))
print (os.environ.get('SNOWFLAKEPASSWORD'))
print (os.environ.get('SNOWFLAKEACCOUNT'))
print (os.environ.get('SNOWFLAKEWAREHOUSE'))
print (os.environ.get('SNOWFLAKEDATABASE'))
print (os.environ.get('SNOWFLAKESCHEMA'))
print (os.environ.get('SNOWFLAKEROLE')) 

# Fetch credentials from environment variables
conn = snowflake.connector.connect(
    user=os.environ.get('SNOWFLAKEUSER'),
    password=os.environ.get('SNOWFLAKEPASSWORD'),
    account=os.environ.get('SNOWFLAKEACCOUNT'),
    warehouse=os.environ.get('SNOWFLAKEWAREHOUSE'),
    database=os.environ.get('SNOWFLAKEDATABASE'),
    schema=os.environ.get('SNOWFLAKESCHEMA'),
    role=os.environ.get('SNOWFLAKEROLE'),
    login_timeout=120,
    network_timeout=120,
    ocsp_fail_open=True
)


# Execute query
query = 'SELECT * FROM WINE;'
df = pd.read_sql(query, conn)

# Save data locally
df.to_csv('data.csv', index=False)

# Close connection
conn.close()

