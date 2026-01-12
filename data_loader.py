import pandas as pd
from sqlalchemy import create_engine

# 1. Define your Query variable (The "What")
# Keeping this outside makes it easy to edit
EMPLOYEE_QUERY = """
SELECT * FROM employees_final;
"""

# 2. Define the Functions (The "How")
def load_from_postgres(sql_query):
    """
    Connects to the database and runs the provided query.
    Returns a pandas dataframe.
    """
    # Connection string for your specific WSL/Docker setup
    engine = create_engine('postgresql://admin:Orion123@127.0.0.1:5433/mydb')
    return pd.read_sql(sql_query, engine)

def load_from_csv(file_path):
    """Fetches data from a local CSV file."""
    return pd.read_csv(file_path)

def load_from_json(file_path):
    """Fetches data from a local JSON file."""
    return pd.read_json(file_path)

def load_from_parquet_sas(lib_table):
    """
    Uses the SAS-Python bridge to load a table from a SAS library.
    Example input: 'PARQUET.employees_raw'
    """
    return SAS.sd2df(lib_table)

#read parquet file in SAS
df = SAS.sd2df("PARQUET.employees_raw")

# 3. Usage:
# Now you simply pass the variable into the function
#df = load_from_postgres(EMPLOYEE_QUERY)

#END statement - Success check
if __name__ == "__main__":
    # This only runs if you type 'python3 data_loader.py'
    print("✅ Data loading module loaded successfully.")