import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_path = os.path.join(script_dir, "../data/Raw/Kaveri_Delta_Soil_Moisture.csv")
cleaned_dir = os.path.join(script_dir, "../data/Cleaned")
cleaned_data_path = os.path.join(cleaned_dir, "Kaveri_Delta_Soil_Moisture.csv")

os.makedirs(cleaned_dir, exist_ok=True)

df = pd.read_csv(raw_data_path)
df = df[['system:index', '.geo', 'VV', 'VH', 'NDVI', 'soil_moisture']]
df = df.dropna()
df = df[(df['soil_moisture'] >= 0) & (df['soil_moisture'] <= 1)]
df.to_csv(cleaned_data_path, index=False)
print("Preprocessing completed")