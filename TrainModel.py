import sys
import os

# Tell Python where your modular files (.py) are located
project_path = "/export/home/users/swelfr/GitProjects/pyforestmodel/"

if project_path not in sys.path:
    sys.path.append(project_path)

import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from model_evaluation import evaluate_model
# Your modular imports
from config import SELECTED_FEATURES, TARGET_VARIABLE
#from data_loader import load_from_postgres
from data_preprocessing import clean_employee_data

# Step 1: Load
#df_raw = load_from_postgres("SELECT * FROM employees_final")
#From SAS Viya Parquet file
df_raw = SAS.sd2df("PARQUET.employees_raw")
# Step 2: Preprocess
df_clean = clean_employee_data(df_raw)

# Step 3: Slice Data (The "Blueprint" step)
X = df_clean[SELECTED_FEATURES]
y = df_clean[TARGET_VARIABLE]  # Changed to lowercase y for consistency

# 4. Split Data (Using the lowercase y from above)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Initialize and train (Added class_weight='balanced' to help with that recall issue)
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

#Model evaluation moved to model_evaluation.py for modularity
evaluate_model(model, X_test, y_test)

#END statement - Success check
print("✅ Model training complete.")

import joblib

# Define the full path where the model should live
model_destination = "/export/home/users/swelfr/GitProjects/pyforestmodel/random_forest_model.joblib"

# Save the model object to that path
joblib.dump(model, model_destination)

print(f"💾 Model saved successfully to: {model_destination}")