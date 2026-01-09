import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Your modular imports
from config import SELECTED_FEATURES, TARGET_VARIABLE
from data_loader import load_from_postgres
from data_preprocessing import clean_employee_data

# 6. Make predictions and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# 7. Feature Importance
importances = pd.DataFrame({
    'feature': X.columns,  
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\nTop Drivers of Turnover:")
print(importances)