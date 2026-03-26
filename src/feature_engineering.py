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

from sklearn.preprocessing import RobustScaler, MinMaxScaler, LabelEncoder
import joblib

def engineer_features_v3():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../data/Cleaned/V3_Combined.csv")
    models_dir = os.path.join(script_dir, "../models")
    processed_dir = os.path.join(script_dir, "../data/Processed")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    
    # 1. Coordinate Extraction
    if '.geo' in df.columns:
        df[['lat', 'lon']] = pd.DataFrame(df['.geo'].apply(extract_geo).tolist(), index=df.index)
        df.drop(columns=['.geo'], inplace=True)

    # 2. Log Transformation for skewed features
    print("Applying log transformation to Rainfall...")
    df['Rainfall'] = np.log1p(df['Rainfall'])

    # 3. Derived SAR Features
    df['VV_VH_ratio'] = df['VV'] / (df['VH'] + 1e-6)
    df['SAR_Index'] = (df['VV'] - df['VH']) / (df['VV'] + df['VH'] + 1e-6)
    df['VV_VH_diff'] = df['VV'] - df['VH']
    df['VV_VH_sum'] = df['VV'] + df['VH']
    df['NDVI_VH'] = df['NDVI'] * df['VH']
    df['NDWI_VH'] = df['NDWI'] * df['VH']
    df['NDVI_NDWI_ratio'] = df['NDVI'] / (df['NDWI'] + 1e-6)

    features_to_scale = [
        'Elevation', 'LST', 'NDVI', 'NDWI', 'Rainfall',
        'VH', 'VV', 'VV_VH_ratio', 'SAR_Index', 'VV_VH_diff', 
        'VV_VH_sum', 'NDVI_VH', 'NDWI_VH', 'NDVI_NDWI_ratio'
    ]
    
    # 4. Global Scaling for Coordinates (Preserve relative spatial information)
    print("Applying global scaling for coordinates...")
    coord_scaler = MinMaxScaler()
    df[['lat', 'lon']] = coord_scaler.fit_transform(df[['lat', 'lon']])
    joblib.dump(coord_scaler, os.path.join(models_dir, "coord_scaler_v3.pkl"))

    # 5. Region-wise Robust Normalization
    print("Applying region-wise Robust scaling (Median/IQR)...")
    scalers = {}
    for region in df['region'].unique():
        scaler = RobustScaler()
        mask = df['region'] == region
        df.loc[mask, features_to_scale] = scaler.fit_transform(df.loc[mask, features_to_scale])
        scalers[region] = scaler
    
    # Save the dictionary of scalers
    joblib.dump(scalers, os.path.join(models_dir, "scalers_v3.pkl"))

    # 6. Label Encoding for Region
    le = LabelEncoder()
    df['region_encoded'] = le.fit_transform(df['region'])
    joblib.dump(le, os.path.join(models_dir, "label_encoder_v3.pkl"))

    # Select final features for training
    final_features = features_to_scale + ['lat', 'lon', 'region_encoded', 'soil_moisture']
    df_final = df[final_features]

    df_final.to_csv(os.path.join(processed_dir, "V3_Final.csv"), index=False)
    print(f"V3.1 Feature Engineering Complete! Processed data saved ({len(df_final)} samples).")

if __name__ == "__main__":
    engineer_features_v3()
