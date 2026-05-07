import os

def push_to_mm_final():
    # 1. Configuration
    LOCAL_PATH = "/tmp/sas_model_RF.astore"
    PROJECT_NAME = "EmployeeChurn"
    
    # 2. Local File Check
    if not os.path.exists(LOCAL_PATH):
        print(f"❌ Error: {LOCAL_PATH} not found. Did the SAS copy run?")
        return

    # 3. Late Imports
    from sasctl import Session, register_model
    from sasctl.services import model_repository as mr

    try:
        # Get internal credentials directly from the Pod environment
        # This bypasses the need for sasctl to 'search' for a session
        url = os.environ.get('SAS_SERVICES_URL')
        token = os.environ.get('SAS_CLIENT_TOKEN')
        
        print(f"--- Using Internal URL: {url} ---")

        # Explicitly pass the token to avoid any interactive handshake hangs
        with Session(url, token=token, verify_ssl=False):
            print("--- Session Authenticated ---")
            
            # Check Project
            project = mr.get_project(PROJECT_NAME)
            if not project:
                print(f"--- Creating Project: {PROJECT_NAME} ---")
                project = mr.create_project(PROJECT_NAME, repo='Public')

            # Register
            print("--- Starting Upload to Model Manager ---")
            register_model(
                model=LOCAL_PATH,
                name="Forest_RF",
                project=PROJECT_NAME,
                force=True
            )
            print("✅ SUCCESS: Model is now in Model Manager!")

    except Exception as e:
        print(f"❌ Execution Failed: {str(e)}")

# Execute
push_to_mm_final()