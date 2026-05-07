import subprocess
import sys
import os
import time

# --- SETUP ---
# Automatically find where we are
project_dir = os.path.dirname(os.path.abspath(__file__))

# Define the sequence of standalone scripts to run
# Right now, it's just the main training script, but you can add more later!
PIPELINE_STEPS = [
    "TrainModel.py"
    # Example for later: "deploy_model.py", "generate_report.py"
]

def run_step(script_name):
    """Runs a Python script as a separate process and prints its output."""
    script_path = os.path.join(project_dir, script_name)
    
    print(f"\n{'='*50}")
    print(f"🚀 STARTING STEP: {script_name}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    # Run the script using the exact same Python version currently active
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True
    )
    
    # Print the Standard Output (what usually goes to the console/log)
    if result.stdout:
        print(result.stdout)
        
    # Print Errors if they happened
    if result.stderr:
        print("⚠️ STANDARD ERROR OUTPUT:")
        print(result.stderr)
        
    elapsed_time = time.time() - start_time
    
    # Check if the script crashed
    if result.returncode != 0:
        print(f"\n❌ PIPELINE HALTED: {script_name} failed with exit code {result.returncode}")
        sys.exit(1) # Stop the whole pipeline
    else:
        print(f"✅ STEP COMPLETE: {script_name} (Took {elapsed_time:.2f} seconds)\n")

# --- EXECUTION ---
if __name__ == "__main__":
    print("🌟 INITIALIZING ML PIPELINE 🌟")
    print(f"Working Directory: {project_dir}")
    
    total_start = time.time()
    
    for step in PIPELINE_STEPS:
        run_step(step)
        
    total_time = time.time() - total_start
    print(f"🎉 ENTIRE PIPELINE FINISHED SUCCESSFULLY IN {total_time:.2f} SECONDS 🎉")