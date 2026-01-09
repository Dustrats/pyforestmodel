# model_evaluation.py
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(model, X_test, y_test):
    """
    Performs full evaluation: Accuracy, Class Report, and Feature Importance.
    """
    # 1. Predictions
    y_pred = model.predict(X_test)
    
    # 2. Basic Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {accuracy:.2%}")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred))

    # 3. Feature Importance 
    # (Note: X_test is used to get the column names)
    importances = pd.DataFrame({
        'feature': X_test.columns,  
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    print("\nTop Drivers of Turnover:")
    print(importances)
    
    return importances

#Success check
if __name__ == "__main__":
    print("✅ Pipeline finished.")