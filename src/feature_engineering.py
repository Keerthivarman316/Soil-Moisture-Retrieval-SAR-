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

def engineer_features_v4():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../data/Cleaned/V3_Combined.csv")
    models_dir = os.path.join(script_dir, "../models")
    processed_dir = os.path.join(script_dir, "../data/Processed")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    if '.geo' in df.columns:
        df[['lat', 'lon']] = pd.DataFrame(df['.geo'].apply(extract_geo).tolist(), index=df.index)
        df.drop(columns=['.geo'], inplace=True)

    df['Rainfall'] = np.log1p(df['Rainfall'])

    print("Adding V4 interaction features...")
    df['VV_VH_ratio'] = df['VV'] / (df['VH'] + 1e-6)
    df['SAR_Index'] = (df['VV'] - df['VH']) / (df['VV'] + df['VH'] + 1e-6)
    df['NDWI_Rain'] = df['NDWI'] * df['Rainfall']
    df['LST_NDVI'] = df['LST'] * df['NDVI']
    df['VV_NDWI'] = df['VV'] * df['NDWI']
    df['VH_NDVI'] = df['VH'] * df['NDVI']
    df['Rain_LST'] = df['Rainfall'] / (df['LST'] + 1.0)

    features_to_scale = [
        'Elevation', 'LST', 'NDVI', 'NDWI', 'Rainfall',
        'VH', 'VV', 'VV_VH_ratio', 'SAR_Index', 
        'NDWI_Rain', 'LST_NDVI', 'VV_NDWI', 'VH_NDVI', 'Rain_LST'
    ]
    
    coord_scaler = MinMaxScaler()
    df[['lat', 'lon']] = coord_scaler.fit_transform(df[['lat', 'lon']])
    joblib.dump(coord_scaler, os.path.join(models_dir, "coord_scaler_v4.pkl"))

    print("Applying region-wise Robust scaling...")
    scalers = {}
    for region in df['region'].unique():
        region_mask = df['region'] == region
        region_scaler = RobustScaler()
        df.loc[region_mask, features_to_scale] = region_scaler.fit_transform(df.loc[region_mask, features_to_scale])
        scalers[region] = region_scaler
    
    joblib.dump(scalers, os.path.join(models_dir, "scalers_v4.pkl"))

    print("Applying One-Hot Encoding for Regions...")
    df_encoded = pd.get_dummies(df, columns=['region'], prefix='region', drop_first=False)
    region_cols = [c for c in df_encoded.columns if c.startswith('region_')]
    joblib.dump(region_cols, os.path.join(models_dir, "region_cols_v4.pkl"))

    final_features = features_to_scale + ['lat', 'lon'] + region_cols + ['region', 'soil_moisture']
    df_final = df_encoded[final_features]

    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df_final, test_size=0.2, random_state=42)
    
    df_final.to_csv(os.path.join(processed_dir, "V4_Final.csv"), index=False)
    df_test.to_csv(os.path.join(processed_dir, "Test_Split_V4.csv"), index=False)
    print(f"V4 Feature engineering complete. Region-Aware features saved.")

if __name__ == "__main__":
    engineer_features_v4()
