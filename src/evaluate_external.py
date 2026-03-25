import pandas as pd
import numpy as np
import os
import json
import joblib
import tensorflow as tf
import sys
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_external_v2(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    name = os.path.basename(file_path)
    print(f"=== Evaluating External Dataset (V2): {name} ===")

    df = pd.read_csv(file_path)
    
    # 1. Preprocessing
    cols_v2 = ['Elevation', 'LST', 'NDVI', 'NDWI', 'Rainfall', 'VH', 'VV', 'soil_moisture', '.geo']
    df = df[cols_v2].dropna()

    # Unify scale to 0-1
    if df['soil_moisture'].max() > 1.1:
        print("-> [INFO] Scaling soil_moisture 0-100 to 0-1 range.")
        df['soil_moisture'] = df['soil_moisture'] / 100.0

    # 2. Feature Engineering
    def extract_geo(geo_str):
        try:
            coords = json.loads(geo_str)['coordinates']
            return coords[1], coords[0]
        except:
            return np.nan, np.nan

    df[['lat', 'lon']] = pd.DataFrame(df['.geo'].apply(extract_geo).tolist(), index=df.index)
    df['VV_VH_ratio'] = df['VV'] / (df['VH'] + 1e-6)
    df['SAR_Index'] = (df['VV'] - df['VH']) / (df['VV'] + df['VH'] + 1e-6)
    df['VV_VH_diff'] = df['VV'] - df['VH']
    df['VV_VH_sum'] = df['VV'] + df['VH']
    df['NDVI_VH'] = df['NDVI'] * df['VH']
    df['NDWI_VH'] = df['NDWI'] * df['VH']
    df['NDVI_NDWI_ratio'] = df['NDVI'] / (df['NDWI'] + 1e-6)

    features = [
        'Elevation', 'LST', 'NDVI', 'NDWI', 'Rainfall', 'lat', 'lon',
        'VH', 'VV', 'VV_VH_ratio', 'SAR_Index', 'VV_VH_diff', 
        'VV_VH_sum', 'NDVI_VH', 'NDWI_VH', 'NDVI_NDWI_ratio'
    ]
    
    # 3. Scaling
    models_dir = os.path.join(os.path.dirname(__file__), "../models")
    scaler = joblib.load(os.path.join(models_dir, "scaler_v2.pkl"))
    X_scaled = scaler.transform(df[features])
    y = df['soil_moisture'].values

    # 4. Load Models and Predict
    rf_model = joblib.load(os.path.join(models_dir, "rf_model_v2.pkl"))
    ann_model = tf.keras.models.load_model(os.path.join(models_dir, "ann_model_v2.keras"))
    meta_learner = joblib.load(os.path.join(models_dir, "meta_learner_v2.pkl"))
    
    y_pred_rf = rf_model.predict(X_scaled)
    y_pred_ann = ann_model.predict(X_scaled, verbose=0).flatten()
    
    meta_features = np.column_stack((y_pred_rf, y_pred_ann))
    y_pred_stacked = meta_learner.predict(meta_features)

    # 5. Metrics
    def get_metrics(y_true, y_pred, model_name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return {"Model": model_name, "RMSE": round(rmse, 4), "R2": round(r2, 4), "MAE": round(mae, 4)}

    results = [
        get_metrics(y, y_pred_rf, "Random Forest (V2)"),
        get_metrics(y, y_pred_ann, "ANN (V2)"),
        get_metrics(y, y_pred_stacked, "Stacked Ensemble")
    ]
    
    print("\nEvaluation Results (Odisha - V2):")
    print(pd.DataFrame(results).to_string(index=False))

    results_dir = os.path.join(os.path.dirname(__file__), "../results")
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y, y=y_pred_stacked, scatter_kws={'alpha': 0.3, 'color': 'green'}, line_kws={'color': 'black'})
    plt.title(f"Odisha V2 Test: Actual vs Predicted (Stacked Model)\nRMSE: {results[2]['RMSE']}, R2: {results[2]['R2']}")
    plt.savefig(os.path.join(results_dir, "Odisha_V2_Results.png"))
    print(f"\nPlot saved to results/Odisha_V2_Results.png")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/Raw 2/Odisha_TEST.csv"
    evaluate_external_v2(path)
