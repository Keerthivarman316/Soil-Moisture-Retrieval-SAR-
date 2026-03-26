import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

def train_v4():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../data/Processed/V4_Final.csv")
    models_dir = os.path.join(script_dir, "../models")
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    
    regions = df['region'].unique()
    print(f"Starting V4 LORO Cross-Validation (Regions: {regions})")

    X = df.drop(columns=['soil_moisture', 'region'])
    y = df['soil_moisture']

    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    xgb_m = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, random_state=42)
    lgb_m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42)

    print("Training Final V4 Ensemble (RF, XGB, LGBM)...")
    rf.fit(X, y)
    xgb_m.fit(X, y)
    lgb_m.fit(X, y)

    print("Creating meta-features for Stacking...")
    p_rf = rf.predict(X)
    p_xgb = xgb_m.predict(X)
    p_lgb = lgb_m.predict(X)
    
    meta_features = np.column_stack((p_rf, p_xgb, p_lgb))
    
    meta_learner = Ridge(alpha=1.0)
    meta_learner.fit(meta_features, y)

    joblib.dump(rf, os.path.join(models_dir, "rf_model_v4.pkl"))
    joblib.dump(xgb_m, os.path.join(models_dir, "xgb_model_v4.pkl"))
    joblib.dump(lgb_m, os.path.join(models_dir, "lgb_model_v4.pkl"))
    joblib.dump(meta_learner, os.path.join(models_dir, "meta_learner_v4.pkl"))

    print("V4 Stacking Assembly Complete (XGB+RF+LGBM -> Ridge).")

if __name__ == "__main__":
    train_v4()
