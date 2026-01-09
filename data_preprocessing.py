#Data Preprocessing
#df included a full read from the employee table in the database
import pandas as pd

def _handle_missing_values(df):
    """Internal helper to fill empty cells."""
    df['salary'] = df['salary'].fillna(df['salary'].median())
    return df

def _encode_categories(df):
    """Internal helper to turn text into numbers."""
    # Example: Convert 'Department' into dummy variables
    return pd.get_dummies(df, columns=['department'], drop_first=True)

def clean_employee_data(df):
    """The 'Master' function you will export to other files."""
    df = df.copy() # Good practice to avoid changing the original raw data
    df = _handle_missing_values(df)
    df = _encode_categories(df)
    # Add any other 31-column specific logic here
    return df

#END statement - Success check
if __name__ == "__main__":
    # This only runs if you type 'python3 data_preprocessing.py'
    print("✅ Preprocessing module loaded successfully.")