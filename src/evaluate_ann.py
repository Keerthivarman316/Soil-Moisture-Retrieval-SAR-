import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../data/Processed/Global_Soil_Moisture.csv")
model_path = os.path.join(script_dir, "../models/ann_global_model.keras")

df = pd.read_csv(data_path)
features = ['VV', 'VH', 'NDVI', 'VV_VH_ratio', 'NDVI_VV', 'month', 'DOY', 'lat', 'lon', 'SAR_Index', 'VV_VH_diff', 'VV_VH_sum', 'NDVI_VH']

X = df[features]
y = df['soil_moisture']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.models.load_model(model_path)
y_pred = model.predict(X_test, verbose=0).flatten()

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "="*40)
print("ANN GLOBAL EVALUATION RESULTS")
print("="*40)
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
print("="*40)
