import sys
import os
import shutil
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# --- ENGINE TOGGLE SECTION ---
# Toggle these when moving between local and Viya
from sklearn.ensemble import RandomForestClassifier as RF
#from sasviya.ml.tree import ForestClassifier as RF 
# ------------------------------

# --- SMART PATH DETECTION ---
try:
    # Works when running as a standalone .py file (Local VS Code/WSL)
    project_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Works inside SAS Studio (Viya) where __file__ is undefined
    project_path = "/export/home/users/swelfr/GitProjects/pyforestmodel"

if project_path not in sys.path:
    sys.path.append(project_path)
# ----------------------------

# Import specialized modules from your project path
from model_evaluation import evaluate_viya_model, evaluate_sklearn_model
from config import SELECTED_FEATURES, TARGET_VARIABLE
from data_preprocessing import clean_employee_data
from data_loader import load_from_parquet_local, load_from_parquet_sas

# Step 1: Load Data
# Using absolute pathing to ensure it works in SAS Studio
parquet_path = os.path.join(project_path, "employees_raw.parquet")

try:
    print(f"--- Attempting to load data from: {parquet_path} ---")
    df_raw = load_from_parquet_local(parquet_path)
except Exception as e:
    print(f"Local load failed: {e}. Falling back to active session 'df' object.")
    df_raw = df # Fallback if you already have 'df' loaded in memory

# Step 2-3: Prep & Split
df_clean = clean_employee_data(df_raw)
X = df_clean[SELECTED_FEATURES]
y = df_clean[TARGET_VARIABLE]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize & Train
model = RF(n_estimators=100, max_depth=15, min_samples_leaf=10, random_state=42)
model.fit(X_train, y_train)

# Step 7: Specialized Evaluation
if 'sasviya' in model.__class__.__module__:
    evaluate_viya_model(model, X_test, y_test)
else:
    # Use print(df.head()) to see results in SAS Log as we discussed
    print("--- Sklearn Model Preview (First 5 Rows) ---")
    print(X_test.head())
    evaluate_sklearn_model(model, X_test, y_test)

# Step 8: Save to Git Repo & CAS (Only if it's a Viya model)
if 'sasviya' in model.__class__.__module__:
    model_name = "RF"
    filename = f"sas_model_{model_name}.astore"
    git_deploy_path = os.path.join(project_path, filename)

    if os.path.exists(git_deploy_path):
        os.remove(git_deploy_path)
    
    print(f"--- Saving Astore to Git Repo: {git_deploy_path} ---")
    model.save(git_deploy_path)
    
    # Export to Models Caslib for Model Manager
    cas_destination = f"Models.sas_model_{model_name}"
    print(f"--- Exporting to CAS: {cas_destination} ---")
    model.export(cas_destination)
    
    print(f"✅ Viya Model saved and exported successfully.")
else:
    # Local fallback for scikit-learn
    joblib.dump(model, os.path.join(project_path, "local_rf_model.joblib"))
    print(f"✅ Local Scikit-Learn model saved to {project_path}")