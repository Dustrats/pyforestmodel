import pandas as pd

from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Connect and Load Data
engine = create_engine('postgresql://admin:Orion123@127.0.0.1:5433/mydb')
query = "SELECT * FROM employees_final;"
df = pd.read_sql(query, engine)

# 2.Preprocessing: Select Features (X) and Target (Y)
# Note: We drop ID and text-heavy columns for simplicity
features = ['salary', 'tenure_months', 'overtime_hours', 'workload_score', 
            'performance_score', 'satisfaction_score', 'turnover_probability']
X = df[features].drop('turnover_probability', axis=1)
Y = df['left_company']

# 3. Split Data into Training and Testing Sets (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 4. Initialize and train the model
# n_estimators is the number of trees in the forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

# 5. Make predictions and evaluate the model
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model Accuracy: {accuracy:.2%}")
print("\nDetailed Report:")
print(classification_report(Y_test, Y_pred))

# 6. Corrected Feature Importance
importances = pd.DataFrame({
    'feature': X.columns,  # Use the columns that were actually trained
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\nTop Drivers of Turnover:")
print(importances)