import sys
import os
import shutil
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# --- MASTER ENVIRONMENT TOGGLE ---
# True = Runs on laptop (Scikit-Learn, Local Parquet)
# False = Runs on Viya (SAS ML, SAS Library)
RUN_LOCALLY = False
# ==========================================

# --- 1. DYNAMIC IMPORTS BASED ON TOGGLE ---
if RUN_LOCALLY:
    from sklearn.ensemble import RandomForestClassifier as RF
else:
    from sasviya.ml.tree import ForestClassifier as RF 

# --- 2. SMART PATH DETECTION ---
try:
    # Works when running as a standalone .py file (Local VS Code/WSL)
    project_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Works inside SAS Studio (Viya) where __file__ is undefined
    project_path = "/export/home/users/swelfr/GitProjects/pyforestmodel"

if project_path not in sys.path:
    sys.path.append(project_path)

# Import specialized modules from your project path
from model_evaluation import evaluate_viya_model, evaluate_sklearn_model
from config import SELECTED_FEATURES, TARGET_VARIABLE
from data_preprocessing import clean_employee_data
from data_loader import load_from_parquet_local, load_from_parquet_sas

# --- 3. DATA LOADING BASED ON TOGGLE ---
if RUN_LOCALLY:
    print("--- [LOCAL MODE] Loading local parquet file ---")
    parquet_path = os.path.join(project_path, "employees_raw.parquet")
    df_raw = load_from_parquet_local(parquet_path)
else:
    print("--- [VIYA MODE] Loading from SAS Library ---")
    # We pass the magic 'SAS' object directly into the function!
    df_raw = load_from_parquet_sas(SAS, "PARQUET.employees_raw")

# Step 4: Prep & Split
df_clean = clean_employee_data(df_raw)
X = df_clean[SELECTED_FEATURES]
y = df_clean[TARGET_VARIABLE]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize & Train
model = RF(n_estimators=100, max_depth=15, min_samples_leaf=10, random_state=42)
model.fit(X_train, y_train)

# Step 7: Specialized Evaluation
if RUN_LOCALLY:
    print("--- Sklearn Model Preview (First 5 Rows) ---")
    print(X_test.head())
    evaluate_sklearn_model(model, X_test, y_test)
else:
    evaluate_viya_model(model, X_test, y_test)

# Step 8: Save to Git Repo & CAS
if RUN_LOCALLY:
    # Local fallback for scikit-learn
    joblib.dump(model, os.path.join(project_path, "local_rf_model.joblib"))
    print(f"✅ Local Scikit-Learn model saved to {project_path}")
else:
    # Viya Astore & CAS Export
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