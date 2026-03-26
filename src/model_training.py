import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, r2_score

def train_v4():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../data/Processed/V4_Final.csv")
    models_dir = os.path.join(script_dir, "../models")
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    
    # Target and Groups for LORO
    y = df['soil_moisture']
    groups = df['region']
    
    # Drop non-feature columns for X
    X = df.drop(columns=['soil_moisture', 'region'])

    print(f"Starting V4 Training with {len(df)} samples...")
    print(f"Features count: {X.shape[1]}")

    # 1. Leave-One-Region-Out (LORO) Validation
    logo = LeaveOneGroupOut()
    loro_results = []

    print("\n--- V4 Leave-One-Region-Out (LORO) CV ---")
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
        region_name = groups.iloc[test_idx].iloc[0]

        # Use XGBoost for LORO benchmark (Region-Aware via One-Hot features)
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
        model.fit(X_train_cv, y_train_cv)
        preds = model.predict(X_test_cv)
        
        rmse = np.sqrt(mean_squared_error(y_test_cv, preds))
        r2 = r2_score(y_test_cv, preds)
        loro_results.append({"Region": region_name, "RMSE": rmse, "R2": r2})
        print(f"Held out {region_name}: RMSE={rmse:.4f}, R2={r2:.4f}")

    print("\nV4 LORO Summary Metrics:")
    print(pd.DataFrame(loro_results).to_string(index=False))

    # 2. Final Model Training (Stacked V4 Ensemble)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Split training for meta-learner training (preventing leakage)
    X_base, X_meta, y_base, y_meta = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print("\nTraining V4 Base Models (XGB, RF, LGBM)...")
    
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_base, y_base)

    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
    xgb_model.fit(X_base, y_base)

    lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1)
    lgb_model.fit(X_base, y_base)

    print("Training Ridge Meta-Learner (Stacked V4)...")
    pred_rf = rf_model.predict(X_meta)
    pred_xgb = xgb_model.predict(X_meta)
    pred_lgb = lgb_model.predict(X_meta)
    
    meta_features = np.column_stack((pred_rf, pred_xgb, pred_lgb))
    meta_learner = Ridge(alpha=1.0)
    meta_learner.fit(meta_features, y_meta)

    # 3. Save V4 Models
    joblib.dump(rf_model, os.path.join(models_dir, "rf_model_v4.pkl"))
    joblib.dump(xgb_model, os.path.join(models_dir, "xgb_model_v4.pkl"))
    joblib.dump(lgb_model, os.path.join(models_dir, "lgb_model_v4.pkl"))
    joblib.dump(meta_learner, os.path.join(models_dir, "meta_learner_v4.pkl"))
    
    # Save test split for evaluation
    test_data = X_test.copy()
    test_data['soil_moisture'] = y_test
    test_data.to_csv(os.path.join(script_dir, "../data/Processed/Test_Split_V4.csv"), index=False)

    print(f"V4 Training Complete! Models and Test split saved.")

if __name__ == "__main__":
    train_v4()
