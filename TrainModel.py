import sys
import os
import pandas as pd
import joblib
import swat
from sklearn.model_selection import train_test_split

# --- ENGINE TOGGLE SECTION ---
# from sklearn.ensemble import RandomForestClassifier as RF
from sasviya.ml.tree import ForestClassifier as RF
# ------------------------------

project_path = "/export/home/users/swelfr/GitProjects/pyforestmodel/"
if project_path not in sys.path:
    sys.path.append(project_path)

# Import the two specialized functions
from model_evaluation import evaluate_viya_model, evaluate_sklearn_model
from config import SELECTED_FEATURES, TARGET_VARIABLE
from data_preprocessing import clean_employee_data

# Step 1-3: Data Prep
df_raw = SAS.sd2df("PARQUET.employees_raw")
df_clean = clean_employee_data(df_raw)
X = df_clean[SELECTED_FEATURES]
y = df_clean[TARGET_VARIABLE]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize & Train
model = RF(n_estimators=100, max_depth=15, min_samples_leaf=10, random_state=42)
model.fit(X_train, y_train)

# Step 7: Specialized Evaluation Branching
if 'sasviya' in model.__class__.__module__:
    evaluate_viya_model(model, X_test, y_test)
else:
    evaluate_sklearn_model(model, X_test, y_test)

# Step 8: Save Joblib
WORK_DIR = os.environ.get('SAS_WORK_DIR', '/tmp')
joblib.dump(model, os.path.join(WORK_DIR, "trained_model.joblib"))

# Step 8: Save Joblib
WORK_DIR = os.environ.get('SAS_WORK_DIR', '/tmp')
joblib.dump(model, os.path.join(WORK_DIR, "trained_model.joblib"))

# Step 9: Save for Deployment
if 'sasviya' in model.__class__.__module__:
    DEPLOY_PATH = os.path.join(project_path, "viya_model_deploy")
    
    # Improved cleanup: handle both files and folders
    import shutil
    if os.path.exists(DEPLOY_PATH):
        if os.path.isdir(DEPLOY_PATH):
            shutil.rmtree(DEPLOY_PATH)
        else:
            os.remove(DEPLOY_PATH)
            
    print(f"--- Saving Model for Deployment to: {DEPLOY_PATH} ---")
    
    # Save call - this creates the ASTORE and metadata files automatically
    model.save(DEPLOY_PATH)
    
    print(f"✅ Deployment package saved to: {DEPLOY_PATH}")