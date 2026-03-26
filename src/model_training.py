import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, r2_score

def train_v3():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../data/Processed/V3_Final.csv")
    models_dir = os.path.join(script_dir, "../models")
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    # Re-map region names for LORO visibility
    le = joblib.load(os.path.join(models_dir, "label_encoder_v3.pkl"))
    
    X = df.drop(columns=['soil_moisture'])
    y = df['soil_moisture']
    groups = df['region_encoded']

    print(f"Starting V3 Training with {len(df)} samples...")

    # 1. Leave-One-Region-Out (LORO) Validation
    logo = LeaveOneGroupOut()
    loro_results = []

    print("\n--- Leave-One-Region-Out (LORO) CV ---")
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        region_name = le.inverse_transform([groups.iloc[test_idx].iloc[0]])[0]

        # Use a simple RF for LORO check
        rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        loro_results.append({"Region": region_name, "RMSE": rmse, "R2": r2})
        print(f"Held out {region_name}: RMSE={rmse:.4f}, R2={r2:.4f}")

    print("\nLORO Summary Metrics:")
    print(pd.DataFrame(loro_results).to_string(index=False))

    # 2. Final Model Training (Stacked Ensemble)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_base, X_meta, y_base, y_meta = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print("\nTraining Upgraded Base Models...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_base, y_base)

    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    gb_model.fit(X_base, y_base)

    ann_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    ann_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    ann_model.fit(X_base, y_base, epochs=50, batch_size=32, verbose=0)

    print("Training Meta-Learner (Stacked V3)...")
    pred_rf = rf_model.predict(X_meta)
    pred_gb = gb_model.predict(X_meta)
    pred_ann = ann_model.predict(X_meta, verbose=0).flatten()
    
    meta_features = np.column_stack((pred_rf, pred_gb, pred_ann))
    meta_learner = LinearRegression()
    meta_learner.fit(meta_features, y_meta)

    # 3. Save V3 Models
    joblib.dump(rf_model, os.path.join(models_dir, "rf_model_v3.pkl"))
    joblib.dump(gb_model, os.path.join(models_dir, "gb_model_v3.pkl"))
    ann_model.save(os.path.join(models_dir, "ann_model_v3.keras"))
    joblib.dump(meta_learner, os.path.join(models_dir, "meta_learner_v3.pkl"))
    
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv(os.path.join(script_dir, "../data/Processed/Test_Split_V3.csv"), index=False)
    
    print("\nV3 Training Complete! Models and Test split saved.")

if __name__ == "__main__":
    train_v3()
