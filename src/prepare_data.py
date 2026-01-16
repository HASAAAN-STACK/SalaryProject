import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load dataset
df = pd.read_csv("Data/Salary_Data.csv")

# Features and target
X = df[["YearsExperience"]]
y = df["Salary"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create processed folder
os.makedirs("Data/processed", exist_ok=True)

# Save processed data
X_train.to_csv("Data/processed/X_train.csv", index=False)
X_test.to_csv("Data/processed/X_test.csv", index=False)
y_train.to_csv("Data/processed/y_train.csv", index=False)
y_test.to_csv("Data/processed/y_test.csv", index=False)

print("Data preparation done!")
