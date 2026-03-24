import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_path = os.path.join(script_dir, "../data/Raw/Kaveri_Delta_Soil_Moisture.csv")
processed_dir = os.path.join(script_dir, "../data/Processed")
processed_data_path = os.path.join(processed_dir, "Kaveri_Delta_Soil_Moisture.csv")

os.makedirs(processed_dir, exist_ok=True)

df = pd.read_csv(raw_data_path)
df = df[['VV', 'VH', 'NDVI', 'soil_moisture']]
df = df.dropna()
df = df[(df['soil_moisture'] >= 0) & (df['soil_moisture'] <= 1)]
df.to_csv(processed_data_path, index=False)
print("Preprocessing completed")