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
def evaluate_external(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return
    name = os.path.basename(file_path)
    print(f"=== Evaluating External Dataset: {name} ===")
    print("-> Loading and Preprocessing...")
    df = pd.read_csv(file_path)
    required_cols = ['system:index', '.geo', 'VV', 'VH', 'NDVI', 'soil_moisture']
    df = df[required_cols].dropna()
    if df['soil_moisture'].max() > 1.1:
        print(f"-> Scaling soil_moisture from 0-100 to 0-1 range.")
        df['soil_moisture'] = df['soil_moisture'] / 100.0
    df = df[(df['soil_moisture'] >= 0) & (df['soil_moisture'] <= 1)].copy()
    print("-> Engineering features...")
    def extract_coords(geo_str):
        try:
            coords = json.loads(geo_str)['coordinates']
            return coords[1], coords[0]
        except:
            return np.nan, np.nan
    df[['lat', 'lon']] = pd.DataFrame(df['.geo'].apply(extract_coords).tolist(), index=df.index)
    df['VV_VH_ratio'] = df['VV'] / (df['VH'] + 1e-6)
    df['NDVI_VV'] = df['NDVI'] * df['VV']
    df['SAR_Index'] = (df['VV'] - df['VH']) / (df['VV'] + df['VH'] + 1e-6)
    df['VV_VH_diff'] = df['VV'] - df['VH']
    df['VV_VH_sum'] = df['VV'] + df['VH']
    df['NDVI_VH'] = df['NDVI'] * df['VH']
    features = [
        'VV', 'VH', 'NDVI', 'VV_VH_ratio', 'NDVI_VV',
        'lat', 'lon',
        'SAR_Index', 'VV_VH_diff', 'VV_VH_sum', 'NDVI_VH'
    ]
    X = df[features]
    y = df['soil_moisture'].values
    print("-> Loading scaler and normalizing features...")
    models_dir = os.path.join(os.path.dirname(__file__), "../models")
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    X_scaled = scaler.transform(X)
    print("-> Loading models and predicting...")
    rf_model = joblib.load(os.path.join(models_dir, "rf_model.pkl"))
    ann_model = tf.keras.models.load_model(os.path.join(models_dir, "ann_model.keras"))
    y_pred_rf = rf_model.predict(X_scaled)
    y_pred_ann = ann_model.predict(X_scaled, verbose=0).flatten()
    y_pred_hybrid = (y_pred_rf + y_pred_ann) / 2.0
    def get_metrics(y_true, y_pred, model_name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return {"Model": model_name, "RMSE": round(rmse, 4), "R2": round(r2, 4), "MAE": round(mae, 4)}
    results = [
        get_metrics(y, y_pred_rf, "Random Forest"),
        get_metrics(y, y_pred_ann, "ANN"),
        get_metrics(y, y_pred_hybrid, "Hybrid (RF+ANN)")
    ]
    results_df = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print(results_df.to_string(index=False))
    results_dir = os.path.join(os.path.dirname(__file__), "../results")
    os.makedirs(results_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y, y=y_pred_hybrid, scatter_kws={'alpha': 0.3, 'color': 'purple'}, line_kws={'color': 'red'})
    plt.title(f"Maharashtra Test: Actual vs Predicted (Hybrid Model)\nRMSE: {results[2]['RMSE']}, R2: {results[2]['R2']}")
    plt.xlabel("Actual Soil Moisture")
    plt.ylabel("Predicted Soil Moisture")
    plt.savefig(os.path.join(results_dir, f"{name.replace('.csv','')}_Results.png"))
    print(f"\nPlot saved to: results/{name.replace('.csv','')}_Results.png")
    print("=" * 40)
if __name__ == "__main__":
    if len(sys.argv) > 1:
        evaluate_external(sys.argv[1])
    else:
        evaluate_external("data/Raw/Maharashtra_TEST.csv")
