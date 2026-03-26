import pandas as pd
import numpy as np
import os
import json
import joblib
import xgboost as xgb
import lightgbm as lgb
import sys
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_external_v4(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    name = os.path.basename(file_path)
    region_name = name.split('_')[0]
    print(f"=== V4 External Evaluation: {name} [Region: {region_name}] ===")

    df = pd.read_csv(file_path)
    
    # 1. Preprocessing (V4 Columns)
    cols_v4 = ['Elevation', 'LST', 'NDVI', 'NDWI', 'Rainfall', 'VH', 'VV', 'soil_moisture', '.geo']
    df = df[cols_v4].dropna()

    def extract_geo(geo_str):
        try:
            coords = json.loads(geo_str)['coordinates']
            return coords[1], coords[0]
        except:
            return np.nan, np.nan

    df[['lat', 'lon']] = pd.DataFrame(df['.geo'].apply(extract_geo).tolist(), index=df.index)
    
    # 2. V4 Feature Engineering
    df['Rainfall'] = np.log1p(df['Rainfall'])
    df['VV_VH_ratio'] = df['VV'] / (df['VH'] + 1e-6)
    df['SAR_Index'] = (df['VV'] - df['VH']) / (df['VV'] + df['VH'] + 1e-6)
    df['NDWI_Rain'] = df['NDWI'] * df['Rainfall']
    df['LST_NDVI'] = df['LST'] * df['NDVI']
    df['VV_NDWI'] = df['VV'] * df['NDWI']
    df['VH_NDVI'] = df['VH'] * df['NDVI']
    df['Rain_LST'] = df['Rainfall'] / (df['LST'] + 1.0)

    features_to_scale = [
        'Elevation', 'LST', 'NDVI', 'NDWI', 'Rainfall',
        'VH', 'VV', 'VV_VH_ratio', 'SAR_Index', 
        'NDWI_Rain', 'LST_NDVI', 'VV_NDWI', 'VH_NDVI', 'Rain_LST'
    ]
    
    # 3. Scaling & Modeling Context
    models_dir = os.path.join(os.path.dirname(__file__), "../models")
    scalers_v4 = joblib.load(os.path.join(models_dir, "scalers_v4.pkl"))
    coord_scaler = joblib.load(os.path.join(models_dir, "coord_scaler_v4.pkl"))
    region_cols = joblib.load(os.path.join(models_dir, "region_cols_v4.pkl"))

    # Global Coordinate Scaling
    df[['lat', 'lon']] = coord_scaler.transform(df[['lat', 'lon']])

    if region_name in scalers_v4:
        print(f"-> [INFO] Using known Robust scaler for {region_name}")
        X_scaled = scalers_v4[region_name].transform(df[features_to_scale])
    else:
        print(f"-> [WARNING] Region '{region_name}' unknown. Using first available scaler for baseline.")
        fallback_region = list(scalers_v4.keys())[0] 
        X_scaled = scalers_v4[fallback_region].transform(df[features_to_scale])

    X_final = pd.DataFrame(X_scaled, columns=features_to_scale)
    X_final[['lat', 'lon']] = df[['lat', 'lon']].values
    
    # 4. V4 One-Hot Encoding (Region-Aware Logic)
    for col in region_cols:
        target_name = f"region_{region_name}"
        X_final[col] = 1 if col == target_name else 0

    y_true = df['soil_moisture'].values

    # 5. Load V4 Ensemble
    rf_model = joblib.load(os.path.join(models_dir, "rf_model_v4.pkl"))
    xgb_model = joblib.load(os.path.join(models_dir, "xgb_model_v4.pkl"))
    lgb_model = joblib.load(os.path.join(models_dir, "lgb_model_v4.pkl"))
    meta_learner = joblib.load(os.path.join(models_dir, "meta_learner_v4.pkl"))
    
    y_pred_rf = rf_model.predict(X_final)
    y_pred_xgb = xgb_model.predict(X_final)
    y_pred_lgb = lgb_model.predict(X_final)
    
    meta_features = np.column_stack((y_pred_rf, y_pred_xgb, y_pred_lgb))
    y_pred_stacked = meta_learner.predict(meta_features)

    # 6. Metrics & Visualization
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_stacked))
    r2 = r2_score(y_true, y_pred_stacked)
    mae = mean_absolute_error(y_true, y_pred_stacked)

    print(f"\nV4 Stacked Results for {region_name}:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
    print(f"MAE:  {mae:.4f}")

    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_true, y=y_pred_stacked, scatter_kws={'alpha': 0.4, 'color': 'purple'})
    plt.title(f"V4 Final Expansion: {region_name}\nR2: {r2:.4f}, RMSE: {rmse:.4f}")
    plt.xlabel("Ground Truth Soil Moisture")
    plt.ylabel("V4 Ensemble Prediction")
    
    results_dir = os.path.join(os.path.dirname(__file__), "../results")
    plt.savefig(os.path.join(results_dir, f"{region_name}_V4_Results.png"))
    print(f"Plot saved to results/{region_name}_V4_Results.png")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/Raw 2/WestBengal_TEST.csv"
    evaluate_external_v4(path)
