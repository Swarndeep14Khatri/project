# -*- coding: utf-8 -*-
"""Project_code.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mcEkp03ZZB5ggNi3M40m-vYZWxBLQv-3
"""

import pandas as pd
import numpy as np
import re
import joblib
import logging
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# ✅ Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)  # ✅ Initialize Flask app

# ✅ Load & Validate Dataset
file_path = "final_cleaned_f1_data_fixed.csv"
try:
    f1_data = pd.read_csv(file_path)
    logger.info(f"✅ Dataset '{file_path}' loaded successfully.")
except FileNotFoundError:
    logger.error(f"❌ ERROR: '{file_path}' not found! Make sure the file is in the correct location.")
    f1_data = None

if f1_data is not None:
    required_columns = {"LapNumber", "LapTime", "TyreAge", "TrackTemperature", "Speed"}
    missing_columns = required_columns - set(f1_data.columns)
    if missing_columns:
        raise ValueError(f"❌ ERROR: Missing columns in dataset: {missing_columns}")

    if f1_data.empty:
        raise ValueError("❌ ERROR: Dataset is empty after preprocessing. Please check the input data.")

    # ✅ Feature Selection
    features = ["LapNumber", "TyreAge", "TrackTemperature", "Speed"]
    target = "LapTime"
    X = f1_data[features]
    y = f1_data[target]

    # ✅ Train-Test Split
    if len(X) < 2:
        raise ValueError("❌ ERROR: Not enough data to split into train and test sets. Check dataset size.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Train Model
    model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
    model.fit(X_train, y_train)

    # ✅ Evaluate Model
    y_pred = model.predict(X_test)
    logger.info(f"✅ Model Trained Successfully! MAE: {mean_absolute_error(y_test, y_pred):.3f}")

    # ✅ Save Model
    joblib.dump(model, "f1_model.pkl")
    logger.info("✅ Model saved as 'f1_model.pkl'.")

# ✅ Load Model for Predictions
try:
    model = joblib.load("f1_model.pkl")
    logger.info("✅ Model loaded successfully.")
except FileNotFoundError:
    logger.error("❌ ERROR: 'f1_model.pkl' not found. Make sure the model is trained and saved properly.")
    model = None

# ✅ Health Check Route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🚀 Flask API is running successfully!"})

# ✅ Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "❌ ERROR: Model not found! Train the model first."}), 500
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)[0]
        return jsonify({"predicted_lap_time": round(prediction, 2)})
    except Exception as e:
        logger.error(f"❌ ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 400

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)