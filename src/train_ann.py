import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../data/Processed/Global_Soil_Moisture.csv")

df = pd.read_csv(data_path)
features = ['VV', 'VH', 'NDVI', 'VV_VH_ratio', 'NDVI_VV', 'month', 'DOY', 'lat', 'lon', 'SAR_Index', 'VV_VH_diff', 'VV_VH_sum', 'NDVI_VH']

X = df[features]
y = df['soil_moisture']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Input(shape=(len(features),)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("Training Global ANN...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

models_dir = os.path.join(script_dir, "../models")
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "ann_global_model.keras")
model.save(model_path)
print(f"Global ANN model saved to {model_path}!")
