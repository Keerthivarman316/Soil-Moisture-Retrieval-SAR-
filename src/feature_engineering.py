import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler
import joblib

def extract_geo(geo_str):
    try:
        coords = json.loads(geo_str)['coordinates']
        return coords[1], coords[0]
    except:
        return np.nan, np.nan

def engineer_features_v2():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../data/Cleaned/V2_Combined.csv")
    models_dir = os.path.join(script_dir, "../models")
    processed_dir = os.path.join(script_dir, "../data/Processed")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    
    if '.geo' in df.columns:
        df[['lat', 'lon']] = pd.DataFrame(df['.geo'].apply(extract_geo).tolist(), index=df.index)
        df.drop(columns=['.geo'], inplace=True)
    elif 'lat' not in df.columns:
        print("Warning: .geo not found and lat/lon not present.")

    df['VV_VH_ratio'] = df['VV'] / (df['VH'] + 1e-6)
    df['SAR_Index'] = (df['VV'] - df['VH']) / (df['VV'] + df['VH'] + 1e-6)
    df['VV_VH_diff'] = df['VV'] - df['VH']
    df['VV_VH_sum'] = df['VV'] + df['VH']
    df['NDVI_VH'] = df['NDVI'] * df['VH']
    df['NDWI_VH'] = df['NDWI'] * df['VH']
    df['NDVI_NDWI_ratio'] = df['NDVI'] / (df['NDWI'] + 1e-6)

    features = [
        'Elevation', 'LST', 'NDVI', 'NDWI', 'Rainfall', 'lat', 'lon',
        'VH', 'VV', 'VV_VH_ratio', 'SAR_Index', 'VV_VH_diff', 
        'VV_VH_sum', 'NDVI_VH', 'NDWI_VH', 'NDVI_NDWI_ratio'
    ]
    
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    joblib.dump(scaler, os.path.join(models_dir, "scaler_v2.pkl"))
    df.to_csv(os.path.join(processed_dir, "V2_Final.csv"), index=False)
    print(f"Feature Engineering Complete! Scaler saved and processed data saved.")

if __name__ == "__main__":
    engineer_features_v2()
