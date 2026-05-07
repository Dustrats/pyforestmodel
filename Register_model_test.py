print("execution started")

import os, sys, joblib, pandas as pd, time
from sasctl import Session, pzmm
print("imports done")

# 1. SETUP
WORK_DIR = os.environ.get('SAS_WORK_DIR', '/tmp')
MODEL_FILE = os.path.join(WORK_DIR, "trained_model.joblib")
LOCAL_DIR = os.path.join(WORK_DIR, "model_assets")

MODEL_NAME = "ForestModel_v1"
PROJECT_NAME = "EmployeeChurn"
print("Phase 1 done")

# Hardcoded to match your training features exactly
SELECTED_FEATURES = ['salary', 'tenure_months', 'overtime_hours', 'workload_score', 
                     'performance_score', 'satisfaction_score', 'turnover_probability']
TARGET_VARIABLE = 'left_company'

def register_to_viya():
    overall_start = time.time()
    print("🚀 Starting Registration...", flush=True)

    # --- STEP A: LOAD LOCAL MODEL ---
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Error: {MODEL_FILE} not found. Did the training session end?", flush=True)
        return
    
    model = joblib.load(MODEL_FILE)
    print(f"✅ Step A: Model loaded from local storage ({time.time() - overall_start:.2f}s)", flush=True)

    # --- STEP B: SCHEMA CREATION ---
    X = pd.DataFrame(columns=SELECTED_FEATURES)
    X.loc[0] = [0] * len(SELECTED_FEATURES)
    print(f"✅ Step B: Schema created", flush=True)

    # --- STEP C: METADATA & PICKLING ---
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)
    
    pzmm.JSONFiles.write_var_json(X, TARGET_VARIABLE, LOCAL_DIR)
    pzmm.PickleModel.pickle_trained_model(MODEL_NAME, model, LOCAL_DIR)
    print(f"✅ Step C: Metadata generated in {LOCAL_DIR}", flush=True)

    # --- STEP D: CONNECTION & PUSH ---
    print(f"🔗 Connecting to Viya...", flush=True)
    try:
        # Note: No 'jupyter_hub' argument as your version didn't like it
        with Session(hostname="https://viya-cauki.unx.sas.com"):
            print(f"📤 Pushing assets to project '{PROJECT_NAME}'...", flush=True)
            
            # Using positional arguments for safety
            pzmm.ImportModel.import_model(
                LOCAL_DIR,      # Path to assets
                MODEL_NAME,     # Name for Model Manager
                PROJECT_NAME,   # Name of Project in Model Manager
                X,              # Input data for variables
                model.predict,  # Prediction method
                True            # Overwrite if exists
            )
    except Exception as e:
        print(f"❌ Connection/Push failed: {e}", flush=True)
        return

    print(f"\n🏆 SUCCESS! Model '{MODEL_NAME}' is now in Model Manager.", flush=True)
    print(f"Total Registration Time: {time.time() - overall_start:.2f}s", flush=True)

register_to_viya()