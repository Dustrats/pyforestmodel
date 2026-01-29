# model_evaluation.py
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

def evaluate_viya_model(model, X_test, y_test):
    """Specialized evaluation for SAS Viya Forest models."""
    y_pred = model.predict(X_test)
    print(f"✅ Viya Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred))

    # Align features and importances using zip to handle length mismatches
    imps = np.array(model.feature_importances_).flatten().tolist()
    feats = list(X_test.columns)
    df_imp = pd.DataFrame(list(zip(feats, imps)), columns=['feature', 'importance'])
    
    # Force importance to numeric and drop non-numeric rows (like 'Intercept')
    df_imp['importance'] = pd.to_numeric(df_imp['importance'], errors='coerce')
    df_imp = df_imp.dropna(subset=['importance'])
    
    print("\nTop Drivers (Viya Engine):")
    print(df_imp.sort_values('importance', ascending=False).head(10))

def evaluate_sklearn_model(model, X_test, y_test):
    """Standard evaluation for Scikit-Learn models."""
    y_pred = model.predict(X_test)
    print(f"✅ Sklearn Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred))

    df_imp = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    })
    
    print("\nTop Drivers (Sklearn Engine):")
    print(df_imp.sort_values('importance', ascending=False).head(10))