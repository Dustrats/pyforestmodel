#To parquet
import pandas as pd
from data_loader import load_from_postgres

# 1. Define the query
query = "SELECT * FROM employees_final"

# 2. Load the data using your existing tool
print("Fetching data from Postgres...")
df = load_from_postgres(query)

# 3. Save to Parquet
# 'snappy' compression is the default and very fast
df.to_parquet('employees_raw.parquet', engine='pyarrow', compression='snappy')

print(f"✅ Success! Saved {len(df)} rows to employees_raw.parquet")