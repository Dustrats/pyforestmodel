import saspy
import pandas as pd

# 1. Load your local data
df = pd.read_parquet('employees_raw.parquet')

# 2. Connect
sas = saspy.SASsession(cfgname='viya') 

# 3. Target the PARQUET library specifically
# We use 'parquet' as the libref (case-insensitive in saspy, but good to match)
sas_table = sas.df2sd(df, table='employees_raw', libref='PARQUET')

print("✅ Data pushed successfully to the PARQUET library.")