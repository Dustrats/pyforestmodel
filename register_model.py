import os
import sys
import joblib
import pandas as pd

# 1. SETUP PROJECT PATH
PROJECT_PATH = "/export/home/users/swelfr/GitProjects/pyforestmodel/"
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

# 2. IMPORTS FROM CONFIG & SASCTL
#from config import TARGET_VARIABLE, SELECTED_FEATURES
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
    # A. Load model
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Error: {MODEL_FILE} not found.")
        return

    model = joblib.load(MODEL_FILE)
    print(f"✅ Loaded model from {MODEL_FILE}")

    # B. Load data sample
    print("Connecting to SAS Library for schema detection...")
    try:
        df_sample = SAS.sd2df("PARQUET.employees_raw(obs=10)")
        X = df_sample[SELECTED_FEATURES]
        y_name = TARGET_VARIABLE
    except Exception as e:
        print(f"❌ Error accessing SAS Library: {e}")
        return

    # C. Prepare folder
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)

    # D. Generate SAS Metadata (Order: Dataframe, Target Name, Path)
    print(f"Generating SAS metadata for target: {y_name}")
    pzmm.JSONFiles.write_var_json(X, y_name, LOCAL_DIR)

    # E. Save the model into the assets folder 
    # CORRECT ORDER FOR YOUR VERSION: (Name, Model, Path)
    print(f"Pickling model into assets folder...")
    pzmm.PickleModel.pickle_trained_model(MODEL_NAME, model, LOCAL_DIR)

    # F. Connect
    # We remove 'jupyter_hub' as your version doesn't recognize it
    with Session(hostname="https://viya-cauki.unx.sas.com"):

        # G. The Final Push
        print(f"Pushing model '{MODEL_NAME}' to SAS Model Manager...")
        pzmm.ImportModel.import_model(
            model_files=LOCAL_DIR,
            model_name=MODEL_NAME,
            project_name=PROJECT_NAME,
            input_data=X,
            predict_method=model.predict,
            overwrite_model=True,
            model_manager_path=PROJECT_NAME # Extra safety for your version
        )

    print(f"🚀 Success! Model '{MODEL_NAME}' is now live in the '{PROJECT_NAME}' project.")

if __name__ == "__main__":
    register_to_viya()