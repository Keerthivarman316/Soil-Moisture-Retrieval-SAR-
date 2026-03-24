import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

script_dir = os.path.dirname(os.path.abspath(__file__))
cleaned_dir = os.path.join(script_dir, "../data/Cleaned")
cleaned_data_path = os.path.join(cleaned_dir, "Kaveri_Delta_Soil_Moisture.csv")

processed_dir = os.path.join(script_dir, "../data/Processed")
processed_data_path = os.path.join(processed_dir, "Kaveri_Delta_Soil_Moisture.csv")

os.makedirs(processed_dir, exist_ok=True)
os.makedirs(cleaned_dir, exist_ok=True)

df = pd.read_csv(cleaned_data_path)

df['VV_VH_ratio'] = df['VV'] / (df['VH'] + 1e-6)
df['NDVI_VV'] = df['NDVI'] * df['VV']

features_to_scale = ['VV', 'VH', 'NDVI', 'VV_VH_ratio', 'NDVI_VV']
scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

df.to_csv(processed_data_path, index=False)
print("Feature Engineering completed")