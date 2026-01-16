from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join("models", "salary_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input from form
        years = float(request.form.get("years_experience"))

        # Make prediction
        prediction = model.predict([[years]])  # returns a numpy array
        salary = round(float(prediction[0]), 2)  # convert first element to float and round

        return render_template("index.html", prediction_text=f"Predicted Salary: ${salary}")
    
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
