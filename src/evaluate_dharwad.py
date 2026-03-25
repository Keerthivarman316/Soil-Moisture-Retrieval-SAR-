import pandas as pd
import numpy as np
import os
import json
import joblib
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_path = os.path.abspath(os.path.join(script_dir, "../data/Raw/Dharwad_Soil_Moisture.csv"))
models_dir = os.path.abspath(os.path.join(script_dir, "../models"))
rf_model_path = os.path.join(models_dir, "rf_global_model.pkl")
ann_model_path = os.path.join(models_dir, "ann_global_model.keras")
scaler_path = os.path.join(models_dir, "global_scaler.pkl")

# Load Dataset
print(f"Loading dataset from: {raw_data_path}")
df = pd.read_csv(raw_data_path)

# 1. Preprocessing - Extract Coordinates
def extract_coords(geo_str):
    try:
        coords = json.loads(geo_str)['coordinates']
        return coords[1], coords[0]   # lat, lon
    except:
        return np.nan, np.nan

print("Preprocessing: Extracting coordinates and engineering features...")
df[['lat', 'lon']] = pd.DataFrame(df['.geo'].apply(extract_coords).tolist(), index=df.index)

# 2. Feature Engineering (Ratios, Indices)
df['VV_VH_ratio'] = df['VV'] / (df['VH'] + 1e-6)
df['NDVI_VV'] = df['NDVI'] * df['VV']
df['SAR_Index'] = (df['VV'] - df['VH']) / (df['VV'] + df['VH'] + 1e-6)
df['VV_VH_diff'] = df['VV'] - df['VH']
df['VV_VH_sum'] = df['VV'] + df['VH']
df['NDVI_VH'] = df['NDVI'] * df['VH']

# Temporal features (Set to global mean since Dharwad dataset lacks date info)
print(f"Loading global scaler from: {scaler_path}")
scaler = joblib.load(scaler_path)

# features_to_scale = ['VV', 'VH', 'NDVI', 'VV_VH_ratio', 'NDVI_VV', 'month', 'DOY', 'lat', 'lon', 'SAR_Index', 'VV_VH_diff', 'VV_VH_sum', 'NDVI_VH']
# month is at index 5, DOY at index 6
df['month'] = scaler.mean_[5]
df['DOY'] = scaler.mean_[6]

# Select and Scale Features
features = [
    'VV', 'VH', 'NDVI', 'VV_VH_ratio', 'NDVI_VV', 
    'month', 'DOY', 'lat', 'lon', 
    'SAR_Index', 'VV_VH_diff', 'VV_VH_sum', 'NDVI_VH'
]

X = scaler.transform(df[features])
y = df['soil_moisture'].values

# --- Random Forest Evaluation ---
print("-" * 40)
print("Evaluating Random Forest Regressor (Global)...")
rf_model = joblib.load(rf_model_path)
y_pred_rf = rf_model.predict(X)

rmse_rf = np.sqrt(mean_squared_error(y, y_pred_rf))
r2_rf = r2_score(y, y_pred_rf)

print(f"RF Results on Dharwad Dataset:")
print(f"RMSE: {rmse_rf:.4f}")
print(f"R2 Score: {r2_rf:.4f}")

# --- ANN Evaluation ---
print("-" * 40)
print("Evaluating Artificial Neural Network (Global)...")
ann_model = tf.keras.models.load_model(ann_model_path)
y_pred_ann = ann_model.predict(X, verbose=0).flatten()

rmse_ann = np.sqrt(mean_squared_error(y, y_pred_ann))
r2_ann = r2_score(y, y_pred_ann)

print(f"ANN Results on Dharwad Dataset:")
print(f"RMSE: {rmse_ann:.4f}")
print(f"R2 Score: {r2_ann:.4f}")
print("-" * 40)
