import os
import sys
import joblib
import pandas as pd

# 1. SETUP PROJECT PATH
PROJECT_PATH = "/export/home/users/swelfr/GitProjects/pyforestmodel/"
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

# 2. IMPORTS FROM CONFIG & SASCTL
from config import TARGET_VARIABLE, SELECTED_FEATURES
from sasctl import Session, pzmm
from sasctl.services import model_repository as model_repo

# --- SETTINGS ---
MODEL_NAME = "ForestModel_v1"
PROJECT_NAME = "EmployeeChurn"

# Absolute paths to ensure Viya finds them regardless of the working directory
MODEL_FILE = os.path.join(PROJECT_PATH, "random_forest_model.joblib")
LOCAL_DIR  = os.path.join(PROJECT_PATH, "model_assets")
# -----------------

def register_to_viya():
    # A. Load the trained model from the persistent Git folder
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Error: {MODEL_FILE} not found. Run TrainModel.py first.")
        return
    
    model = joblib.load(MODEL_FILE)
    print(f"✅ Loaded model from {MODEL_FILE}")

    # B. Load data sample from SAS Library using the Bridge
    # This uses your config variables to define the signature
    print("Connecting to SAS Library for schema detection...")
    try:
        # We only need 10 rows to map the data types
        df_sample = SAS.sd2df("PARQUET.employees_raw(obs=10)")
        X = df_sample[SELECTED_FEATURES]
        y_name = TARGET_VARIABLE
    except Exception as e:
        print(f"❌ Error accessing SAS Library: {e}")
        return

    # C. Prepare the local metadata folder
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)

    # D. Generate SAS Metadata (Using Config variables)
    print(f"Generating SAS metadata for target: {y_name}")
    # Pass X, y_name, and path as positional arguments
    pzmm.JSONFiles.write_var_json(X, y_name, LOCAL_DIR)
    # E. Save the model into the assets folder for the push
    pzmm.PickleModel.save_trained_model(model, MODEL_NAME, path=LOCAL_DIR)

    # F. Connect and Upload to Model Manager
    with Session(hostname="https://viya-cauki.unx.sas.com", jupyter_hub=False):
        
        # Ensure Project exists
        project = model_repo.get_project(PROJECT_NAME)
        if project is None:
            print(f"Creating new project: {PROJECT_NAME}")
            project = model_repo.create_project(PROJECT_NAME, 'Classification')

        # G. The Final Push to the Repository
        print(f"Pushing model '{MODEL_NAME}' to SAS Model Manager...")
        pzmm.ImportModel.import_model(
            model_files=LOCAL_DIR,
            model_name=MODEL_NAME,
            project_name=PROJECT_NAME,
            input_data=X,
            predict_method=model.predict,
            overwrite_model=True
        )

    print(f"🚀 Success! Model '{MODEL_NAME}' is now live in the '{PROJECT_NAME}' project.")

if __name__ == "__main__":
    register_to_viya()