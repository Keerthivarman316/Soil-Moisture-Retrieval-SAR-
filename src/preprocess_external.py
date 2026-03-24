import pandas as pd
import os
import json
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

def extract_coords(geo_str):
    try:
        coords = json.loads(geo_str)['coordinates']
        return coords[1], coords[0] 
    except:
        return np.nan, np.nan

def process_and_evaluate(raw_data_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scaler_path = os.path.join(script_dir, "../models/scaler.pkl")
    model_path = os.path.join(script_dir, "../models/random_forest_model.pkl")
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please re-run feature_engineering.py first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        
    print(f"Loading raw external dataset from: {raw_data_path}")
    raw_df = pd.read_csv(raw_data_path)
    
    print("Preprocessing and filtering...")
    df = raw_df[['system:index', '.geo', 'VV', 'VH', 'NDVI', 'soil_moisture']]
    df = df.dropna()
    df = df[(df['soil_moisture'] >= 0) & (df['soil_moisture'] <= 1)].copy()
    
    print("Feature Engineering...")
    df['date'] = pd.to_datetime(df['system:index'].str.split('_').str[0], format='%Y%m%d')
    df['month'] = df['date'].dt.month
    df['DOY'] = df['date'].dt.dayofyear
    
    df[['lat', 'lon']] = pd.DataFrame(df['.geo'].apply(extract_coords).tolist(), index=df.index)
    
    df['VV_VH_ratio'] = df['VV'] / (df['VH'] + 1e-6)
    df['NDVI_VV'] = df['NDVI'] * df['VV']
    df['SAR_Index'] = (df['VV'] - df['VH']) / (df['VV'] + df['VH'] + 1e-6)
    df['VV_VH_diff'] = df['VV'] - df['VH']
    df['VV_VH_sum'] = df['VV'] + df['VH']
    df['NDVI_VH'] = df['NDVI'] * df['VH']
    
    features_to_scale = [
        'VV', 'VH', 'NDVI', 'VV_VH_ratio', 'NDVI_VV', 
        'month', 'DOY', 'lat', 'lon', 
        'SAR_Index', 'VV_VH_diff', 'VV_VH_sum', 'NDVI_VH'
    ]
    
    print("Applying saved standard scaler...")
    scaler = joblib.load(scaler_path)
    df[features_to_scale] = scaler.transform(df[features_to_scale])
    
    processed_path = raw_data_path.replace("Raw", "Processed")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df[features_to_scale + ['soil_moisture']].to_csv(processed_path, index=False)
    print(f"Saved perfectly scaled external dataset to: {processed_path}")
    
    X = df[features_to_scale]
    y_true = df['soil_moisture']
    
    print(f"Loading model and running predictions on {len(X)} samples...")
    model = joblib.load(model_path)
    y_pred = model.predict(X)
    
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        
    r2 = r2_score(y_true, y_pred)
    
    print("\n" + "="*40)
    print("TESTING ON EXTERNAL DATASET RESULTS")
    print("="*40)
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print("="*40)
    
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Replace 'External_Data.csv' with your actual file name
    raw_data_path = os.path.join(script_dir, "../data/Raw/Punjab_Soil_Moisture.csv")
    
    process_and_evaluate(raw_data_path)
