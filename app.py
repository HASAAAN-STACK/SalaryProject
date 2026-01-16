# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

# Define input schema
class SalaryInput(BaseModel):
    YearsExperience: float

app = FastAPI(title="Salary Prediction API")

# Load model
MODEL_PATH = "models/salary_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "Salary Prediction API is running."}

@app.post("/predict")
def predict_salary(data: SalaryInput):
    input_value = [[data.YearsExperience]]
    prediction = model.predict(input_value)
    return {"YearsExperience": data.YearsExperience, "PredictedSalary": float(prediction[0])}
