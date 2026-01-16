import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load processed data
X_train = pd.read_csv("Data/processed/X_train.csv")
y_train = pd.read_csv("Data/processed/y_train.csv")

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make sure folder exists
import os
os.makedirs("models", exist_ok=True)

# Save model exactly as declared in DVC
joblib.dump(model, "models/salary_model.pkl")

print("Model trained and saved!")
