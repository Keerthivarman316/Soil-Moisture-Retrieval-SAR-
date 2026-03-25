import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras import layers, models
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_path = os.path.join(script_dir, "../data/Processed/Final_Dataset.csv")
models_dir = os.path.join(script_dir, "../models")
os.makedirs(models_dir, exist_ok=True)
print("Starting Model Training...")
df = pd.read_csv(processed_path)
X = df.drop(columns=['soil_moisture'])
y = df['soil_moisture']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_csv(os.path.join(script_dir, "../data/Processed/Test_Split.csv"), index=False)
print("-> Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, os.path.join(models_dir, "rf_model.pkl"))
print("-> Training ANN...")
ann_model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
ann_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
ann_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
ann_model.save(os.path.join(models_dir, "ann_model.keras"))
print("-" * 30)
print("Model Training Complete!")
print("Random Forest saved to: models/rf_model.pkl")
print("ANN saved to: models/ann_model.keras")
print("Test split saved for evaluation: data/Processed/Test_Split.csv")
