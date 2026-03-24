import pandas as pd
import os
import numpy as np
import json
import joblib
from sklearn.preprocessing import StandardScaler

def extract_coords(geo_str):
    try:
        coords = json.loads(geo_str)['coordinates']
        return coords[1], coords[0]
    except:
        return np.nan, np.nan

def engineer_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['system:index'].str.split('_').str[0], format='%Y%m%d')
    df['month'] = df['date'].dt.month
    df['DOY'] = df['date'].dt.dayofyear
    df[['lat', 'lon']] = pd.DataFrame(df['.geo'].apply(extract_coords).tolist(), index=df.index)
    df['VV_VH_ratio'] = df['VV'] / (df['VH'] + 1e-6)
    df['NDVI_VV'] = df['NDVI'] * df['VV']
    df['SAR_Index'] = (df['VV'] - df['VH']) / (df['VV'] + df['VH'] + 1e-6)
    df['VV_VH_diff'] = df['VV'] - df['VH']
    df['VV_VH_sum'] = df['VV'] + df['VH']
    df['NDVI_VH'] = df['NDVI'] * df['VH']
    df = df.drop(columns=['system:index', '.geo', 'date'])
    return df

script_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(script_dir, "../data/Raw")
processed_dir = os.path.join(script_dir, "../data/Processed")
models_dir = os.path.join(script_dir, "../models")
os.makedirs(processed_dir, exist_ok=True)

kaveri_raw = pd.read_csv(os.path.join(raw_dir, "Kaveri_Delta_Soil_Moisture.csv"))
punjab_raw = pd.read_csv(os.path.join(raw_dir, "Punjab_Soil_Moisture.csv"))

def clean(df):
    df = df[['system:index', '.geo', 'VV', 'VH', 'NDVI', 'soil_moisture']].dropna()
    return df[(df['soil_moisture'] >= 0) & (df['soil_moisture'] <= 1)].copy()

kaveri_clean = clean(kaveri_raw)
punjab_clean = clean(punjab_raw)

combined_raw = pd.concat([kaveri_clean, punjab_clean], axis=0).reset_index(drop=True)
combined_engineered = engineer_features(combined_raw)

features_to_scale = ['VV', 'VH', 'NDVI', 'VV_VH_ratio', 'NDVI_VV', 'month', 'DOY', 'lat', 'lon', 'SAR_Index', 'VV_VH_diff', 'VV_VH_sum', 'NDVI_VH']

global_scaler = StandardScaler()
combined_engineered[features_to_scale] = global_scaler.fit_transform(combined_engineered[features_to_scale])

joblib.dump(global_scaler, os.path.join(models_dir, "global_scaler.pkl"))

global_path = os.path.join(processed_dir, "Global_Soil_Moisture.csv")
combined_engineered.to_csv(global_path, index=False)
print(f"Global dataset saved with {len(combined_engineered)} samples and a fresh unified scaler!")
