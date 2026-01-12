import os
import sys
import joblib
import pandas as pd

# 1. SETUP PROJECT PATH 
# This ensures Python can find your config.py and modular scripts
PROJECT_PATH = "/export/home/users/swelfr/GitProjects/pyforestmodel/"
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

# 2. IMPORTS FROM CONFIG (Your Source of Truth)
from config import TARGET_VARIABLE, SELECTED_FEATURES
from sasctl import Session, pzmm
from sasctl.services import model_repository as model_repo

# --- CONFIGURATION ---
MODEL_NAME = "ForestModel_v1"
PROJECT_NAME = "EmployeeChurn"
LOCAL_DIR = "./model_assets"      # Temp folder for SAS metadata
MODEL_FILE = "random_forest_model.joblib"
# ---------------------

def register_to_viya():
    # A. Load your local model 
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Error: {MODEL_FILE} not found. Did you run TrainModel.py first?")
        return

    model = joblib.load(MODEL_FILE)
    
    # B. Load data sample for schema detection
    # We use the same Parquet file used in training
    try:
        df_sample = pd.read_parquet(f'{PROJECT_PATH}employees_raw.parquet').head(10)
    except FileNotFoundError:
        # Fallback if running inside SAS and need to pull from Library
        import sas7bdat 
        print("Local parquet not found, attempting to pull sample from SAS...")
        # Note: In PROC PYTHON, you could use SAS.sd2df("PARQUET.employees_raw")
    
    # C. Slice data using Config variables
    X = df_sample[SELECTED_FEATURES]
    y_name = TARGET_VARIABLE

    # D. Prepare the local "Asset" folder for SAS Metadata
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)

    # E. Generate SAS Metadata (Using updated snake_case functions)
    print(f"Generating SAS metadata for {MODEL_NAME}...")
    pzmm.JSONFiles.write_var_json(X, target=y_name, path=LOCAL_DIR)
    
    # This creates the .pickle or .joblib inside the assets folder for SAS
    pzmm.PickleModel.save_trained_model(model, MODEL_NAME, path=LOCAL_DIR)

    # F. Connect and Upload to Model Manager
    # hostname: the base URL of your Viya environment
    with Session(hostname="https://viya-cauki.unx.sas.com", jupyter_hub=False):
        
        # Ensure the Project exists in Model Manager
        project = model_repo.get_project(PROJECT_NAME)
        if project is None:
            print(f"Creating new project in Model Manager: {PROJECT_NAME}")
            project = model_repo.create_project(PROJECT_NAME, 'Classification')

        # G. The Final Push
        print(f"Pushing model assets to SAS Model Manager...")
        pzmm.ImportModel.import_model(
            model_files=LOCAL_DIR,
            model_name=MODEL_NAME,
            project_name=PROJECT_NAME,
            input_data=X,
            predict_method=model.predict,
            overwrite_model=True
        )

    print(f"✅ Success! Model '{MODEL_NAME}' is now visible in SAS Model Manager under project '{PROJECT_NAME}'.")

if __name__ == "__main__":
    register_to_viya()