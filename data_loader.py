import pandas as pd
from sqlalchemy import create_engine

# 1. Define your Query variable (The "What")
EMPLOYEE_QUERY = """
SELECT * FROM employees_final;
"""

# 2. Define the Functions (The "How")
def load_from_postgres(sql_query):
    """
    Connects to the database and runs the provided query.
    Returns a pandas dataframe.
    """
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

def load_from_parquet_local(file_path):
    """
    Fetches data from a local Parquet file using pure Python.
    """
    return pd.read_parquet(file_path)


# ---------------------------------------------------------
# 3. Usage & Testing (THE FIX IS HERE)
# ---------------------------------------------------------

# The "if __name__" block acts as a protective shield.
# Code inside this block ONLY runs if you execute this file directly.
# It will NOT run if you import this file into TrainModel.py.

if __name__ == "__main__":
    print("✅ Data loading module loaded successfully.")
    
    # You can put all your "test" runs in here safely.
    # Just uncomment the one you want to test while building!
    
    # df = load_from_postgres(EMPLOYEE_QUERY)
    # df = load_from_parquet_sas("PARQUET.employees_raw")
    # df = load_from_parquet_local("employees_raw.parquet")