import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
cleaned_dir = os.path.join(script_dir, "../data/Cleaned")
cleaned_data_path = os.path.join(cleaned_dir, "Kaveri_Delta_Soil_Moisture.csv")

processed_dir = os.path.join(script_dir, "../data/Processed")
processed_data_path = os.path.join(processed_dir, "Kaveri_Delta_Soil_Moisture.csv")

os.makedirs(processed_dir, exist_ok=True)
os.makedirs(cleaned_dir, exist_ok=True)

df = pd.read_csv(cleaned_data_path)

df['date'] = pd.to_datetime(df['system:index'].str.split('_').str[0], format='%Y%m%d')
df['month'] = df['date'].dt.month
df['DOY'] = df['date'].dt.dayofyear

def extract_coords(geo_str):
    try:
        coords = json.loads(geo_str)['coordinates']
        return coords[1], coords[0] 
    except:
        return np.nan, np.nan

df[['lat', 'lon']] = pd.DataFrame(df['.geo'].apply(extract_coords).tolist(), index=df.index)

df['VV_VH_ratio'] = df['VV'] / (df['VH'] + 1e-6)
df['NDVI_VV'] = df['NDVI'] * df['VV']
df['SAR_Index'] = (df['VV'] - df['VH']) / (df['VV'] + df['VH'] + 1e-6)
df['VV_VH_diff'] = df['VV'] - df['VH']
df['VV_VH_sum'] = df['VV'] + df['VH']
df['NDVI_VH'] = df['NDVI'] * df['VH']

features_to_scale = [
    'VV', 'VH', 'NDVI', 'VV_VH_ratio', 'NDVI_VV', 
    'month', 'DOY', 'lat', 'lon', 
    'SAR_Index', 'VV_VH_diff', 'VV_VH_sum', 'NDVI_VH'
]

scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

df = df.drop(columns=['system:index', '.geo', 'date'])

df.to_csv(processed_data_path, index=False)
print("Feature Engineering completed")