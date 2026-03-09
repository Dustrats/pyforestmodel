import os
import sys
from sasctl import register_model
from sasctl.services import model_repository as mr

# --- CONFIGURATION ---
PROJECT_NAME = "EmployeeChurn"
GIT_REPO_PATH = "/export/home/users/swelfr/GitProjects/pyforestmodel/"

def push_astore_to_model_manager(model_id_name):
    filename = f"sas_model_{model_id_name}.astore"
    file_path = os.path.join(GIT_REPO_PATH, filename)

    if not os.path.exists(file_path):
        print(f"❌ Error: File {file_path} not found.")
        return

    # The try block now has its mandatory except handler
    try:
        print(f"--- Checking for Project: {PROJECT_NAME} ---")
        
        # Accessing the service directly inherits your current SAS Studio identity
        project = mr.get_project(PROJECT_NAME)
        if project is None:
            print(f"--- Creating new Project: {PROJECT_NAME} ---")
            project = mr.create_project(PROJECT_NAME, repo='Public')

        print(f"--- Registering {model_id_name} to Model Manager ---")
        model_obj = register_model(
            model=file_path,
            name=f"Forest_{model_id_name}",
            project=PROJECT_NAME,
            force=True
        )
        print(f"✅ Success! Model registered.")

    except Exception as e:
        print(f"❌ Failed to push model: {str(e)}")

# Execution logic
if __name__ == "__main__":
    # Check if a specific name was passed, else default to 'RF'
    if len(sys.argv) > 1 and not sys.argv[1].endswith('.py'):
        target_name = sys.argv[1]
    else:
        target_name = "RF"

    push_astore_to_model_manager(target_name)