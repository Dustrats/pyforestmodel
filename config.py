#Config file for pyforestmodel
#variable selection for model training
SELECTED_FEATURES = ['salary', 'tenure_months', 'overtime_hours', 'workload_score', 
           'performance_score', 'satisfaction_score', 'turnover_probability']
TARGET_VARIABLE = 'left_company'

#END statement - Success check
if __name__ == "__main__":
    # This only runs if you type 'python3 config.py'
    print("✅ Config module loaded successfully.")
