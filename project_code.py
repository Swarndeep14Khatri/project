import pandas as pd
import numpy as np
import joblib
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load Enhanced Dataset (use raw string for the file path)
file_path = r"C:\Users\Hp\Downloads\enriched_f1_race_strategy.csv"  # Use raw string (r) to avoid unicode errors
try:
    f1_data = pd.read_csv(file_path)
    logger.info(f"Dataset '{file_path}' loaded successfully.")
except FileNotFoundError:
    logger.error(f"ERROR: '{file_path}' not found! Ensure the file is in the correct location.")
    f1_data = None

if f1_data is not None:
    # Feature Encoding
    f1_data = pd.get_dummies(f1_data, columns=["TyreCompound", "WeatherCondition"], drop_first=True)
    
    # Feature Selection (Check if these columns exist after get_dummies)
    available_columns = f1_data.columns.tolist()
    logger.info(f"Available columns: {available_columns}")
    
    features = [
        "LapNumber", "TyreAge", "TrackTemperature", "Speed", "FuelLoad", "EnginePower", 
        "DownforceLevel", "BrakingEfficiency", "PitStopTime", "CarAerodynamics", "TrackGripLevel",
        "TyreCompound_Medium", "TyreCompound_Soft", "WeatherCondition_Cloudy", "WeatherCondition_Rainy"
    ]
    
    # Check if the selected columns exist in the dataframe after dummies
    missing_features = [feature for feature in features if feature not in available_columns]
    if missing_features:
        logger.error(f"Missing features: {missing_features}")
        features = [feature for feature in features if feature not in missing_features]

    target = "LapTime"
    
    X = f1_data[features]
    y = f1_data[target]
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    model = HistGradientBoostingRegressor(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate Model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Model Performance: MAE={mae:.3f}, MSE={mse:.3f}, RMSE={rmse:.3f}, R^2={r2:.3f}")
    
    # Save Model
    joblib.dump(model, "f1_model.pkl")
    logger.info("Model saved as 'f1_model.pkl'.")
    
    # Visualization - Correlation Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(f1_data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()
    
# Load Model for Predictions
try:
    model = joblib.load("f1_model.pkl")
    logger.info("Model loaded successfully.")
except FileNotFoundError:
    logger.error("ERROR: 'f1_model.pkl' not found. Train the model first.")
    model = None
