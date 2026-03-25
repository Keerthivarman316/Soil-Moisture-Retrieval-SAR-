import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.preprocessing import StandardScaler
script_dir = os.path.dirname(os.path.abspath(__file__))
cleaned_path = os.path.join(script_dir, "../data/Cleaned/Combined_Cleaned.csv")
processed_dir = os.path.join(script_dir, "../data/Processed")
models_dir = os.path.join(script_dir, "../models")
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
print("Starting Feature Engineering...")
df = pd.read_csv(cleaned_path)
def extract_coords(geo_str):
    try:
        coords = json.loads(geo_str)['coordinates']
        return coords[1], coords[0]
    except:
        return np.nan, np.nan
print("-> Extracting coordinates...")
df[['lat', 'lon']] = pd.DataFrame(df['.geo'].apply(extract_coords).tolist(), index=df.index)
print("-> Computing SAR indices...")
df['VV_VH_ratio'] = df['VV'] / (df['VH'] + 1e-6)
df['NDVI_VV'] = df['NDVI'] * df['VV']
df['SAR_Index'] = (df['VV'] - df['VH']) / (df['VV'] + df['VH'] + 1e-6)
df['VV_VH_diff'] = df['VV'] - df['VH']
df['VV_VH_sum'] = df['VV'] + df['VH']
df['NDVI_VH'] = df['NDVI'] * df['VH']
features_to_scale = [
    'VV', 'VH', 'NDVI', 'VV_VH_ratio', 'NDVI_VV',
    'lat', 'lon',
    'SAR_Index', 'VV_VH_diff', 'VV_VH_sum', 'NDVI_VH'
]
print(f"-> Normalizing {len(features_to_scale)} features...")
scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale].fillna(df[features_to_scale].mean()))
joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
df_final = df[features_to_scale + ['soil_moisture']]
df_final.to_csv(os.path.join(processed_dir, "Final_Dataset.csv"), index=False)
print("-" * 30)
print("Feature Engineering Complete!")
print(f"Dataset shape: {df_final.shape}")
print(f"Processed data saved to: data/Processed/Final_Dataset.csv")
print(f"Scaler saved to: models/scaler.pkl")
