import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load test data
X_test = pd.read_csv("Data/processed/X_test.csv")
y_test = pd.read_csv("Data/processed/y_test.csv")

# Load trained model
model = joblib.load("models/linear_salary_model.pkl")

# Predict
y_pred = model.predict(X_test)

# Metrics
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R2: {r2}")
