import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def train_v2():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../data/Processed/V2_Final.csv")
    models_dir = os.path.join(script_dir, "../models")
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    X = df.drop(columns=['soil_moisture'])
    y = df['soil_moisture']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Nested split for stacking meta-learner
    X_base, X_meta, y_base, y_meta = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print(f"Training Base Models (Samples: {len(X_base)})...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_base, y_base)

    ann_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    ann_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    ann_model.fit(X_base, y_base, epochs=50, batch_size=32, verbose=0, validation_split=0.1)

    print("Training Meta-Learner (Stacked Ensemble)...")
    pred_rf = rf_model.predict(X_meta)
    pred_ann = ann_model.predict(X_meta, verbose=0).flatten()
    
    meta_features = np.column_stack((pred_rf, pred_ann))
    meta_learner = LinearRegression()
    meta_learner.fit(meta_features, y_meta)

    joblib.dump(rf_model, os.path.join(models_dir, "rf_model_v2.pkl"))
    ann_model.save(os.path.join(models_dir, "ann_model_v2.keras"))
    joblib.dump(meta_learner, os.path.join(models_dir, "meta_learner_v2.pkl"))
    
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv(os.path.join(script_dir, "../data/Processed/Test_Split_V2.csv"), index=False)
    
    print("V2 Training Complete! Models and Test split saved.")

if __name__ == "__main__":
    train_v2()
