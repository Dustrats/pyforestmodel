import sys
import os
import shutil
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# --- ENGINE TOGGLE SECTION ---
# Toggle these when moving between local and Viya
from sklearn.ensemble import RandomForestClassifier as RF
# from sasviya.ml.tree import ForestClassifier as RF 
# ------------------------------

# --- SMART PATH DETECTION ---
# This finds the directory where the current script is actually located
script_dir = os.path.dirname(os.path.abspath(__file__))
project_path = script_dir 

if project_path not in sys.path:
    sys.path.append(project_path)

from model_evaluation import evaluate_viya_model, evaluate_sklearn_model
from config import SELECTED_FEATURES, TARGET_VARIABLE
from data_preprocessing import clean_employee_data
from data_loader import load_from_parquet_local, load_from_parquet_sas

# Step 1: Load Data
# TIP: Use absolute path so Viya doesn't get lost
parquet_path = os.path.join(project_path, "employees_raw.parquet")
df_raw = load_from_parquet_local(parquet_path)

# Step 2-3: Prep
df_clean = clean_employee_data(df_raw)
X = df_clean[SELECTED_FEATURES]
y = df_clean[TARGET_VARIABLE]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train
model = RF(n_estimators=100, max_depth=15, min_samples_leaf=10, random_state=42)
model.fit(X_train, y_train)

# Step 7: Specialized Evaluation
if 'sasviya' in model.__class__.__module__:
    evaluate_viya_model(model, X_test, y_test)
else:
    evaluate_sklearn_model(model, X_test, y_test)

# Step 8: Save to Git Repo & CAS (Only if it's a Viya model)
if 'sasviya' in model.__class__.__module__:
    model_name = "RF"
    filename = f"sas_model_{model_name}.astore"
    git_deploy_path = os.path.join(project_path, filename)

    if os.path.exists(git_deploy_path):
        os.remove(git_deploy_path)
    
    print(f"--- Saving Model to Git Repo: {git_deploy_path} ---")
    model.save(git_deploy_path)
    
    # Export to Models Caslib for Model Manager
    cas_destination = f"Models.sas_model_{model_name}"
    model.export(cas_destination)
    
    print(f"✅ Model saved to Git and exported to CAS: {cas_destination}")
else:
    # Local fallback for scikit-learn
    joblib.dump(model, "local_rf_model.joblib")
    print("✅ Local Scikit-Learn model saved via joblib.")