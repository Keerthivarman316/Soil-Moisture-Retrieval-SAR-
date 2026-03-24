import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../data/Processed/Kaveri_Delta_Soil_Moisture.csv")
models_dir = os.path.join(script_dir, "../models")
model_path = os.path.join(models_dir, "random_forest_model.pkl")

os.makedirs(models_dir, exist_ok=True)

df = pd.read_csv(data_path)
x = df[['VV', 'VH', 'NDVI', 'VV_VH_ratio', 'NDVI_VV', 'month', 'DOY', 'lat', 'lon', 'SAR_Index', 'VV_VH_diff', 'VV_VH_sum', 'NDVI_VH']]
y = df['soil_moisture']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, model_path)
print("Model trained and saved successfully")