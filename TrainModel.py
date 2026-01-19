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
from config import SELECTED_FEATURES, TARGET_VARIABLE
from data_preprocessing import clean_employee_data

# Step 1: Load
df_raw = SAS.sd2df("PARQUET.employees_raw")

# Step 2: Preprocess
df_clean = clean_employee_data(df_raw)

# Step 3: Slice Data
X = df_clean[SELECTED_FEATURES]
y = df_clean[TARGET_VARIABLE]

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Initialize and train (MOVE THESE TO THE LEFT MARGIN)
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,             # Prevents trees from growing infinitely
    min_samples_leaf=10,      # Ensures each leaf has a minimum amount of data
    random_state=42,
    class_weight='balanced'
)

# --- ADD THIS LINE BACK ---
model.fit(X_train, y_train)

# Model evaluation
evaluate_model(model, X_test, y_test)

# END statement
print("✅ Model training complete.")

import joblib

# Get the local SAS WORK path dynamically
WORK_DIR = os.environ.get('SAS_WORK_DIR', '/tmp')
LOCAL_MODEL_PATH = os.path.join(WORK_DIR, "trained_model.joblib")

# Save here
joblib.dump(model, LOCAL_MODEL_PATH)
print(f"✅ Model saved locally to: {LOCAL_MODEL_PATH}")