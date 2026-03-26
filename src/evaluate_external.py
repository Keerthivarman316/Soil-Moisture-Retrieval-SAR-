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

import pandas as pd
import numpy as np
import os
import json
import joblib
import tensorflow as tf
import sys
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_external_v3(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    name = os.path.basename(file_path)
    region_name = name.split('_')[0]
    print(f"=== Evaluating External Dataset (V3): {name} [Region: {region_name}] ===")

    df = pd.read_csv(file_path)
    
    # 1. Preprocessing (V3 Columns)
    cols_v3 = ['Elevation', 'LST', 'NDVI', 'NDWI', 'Rainfall', 'VH', 'VV', 'soil_moisture', '.geo']
    df = df[cols_v3].dropna()

    def extract_geo(geo_str):
        try:
            coords = json.loads(geo_str)['coordinates']
            return coords[1], coords[0]
        except:
            return np.nan, np.nan

    df[['lat', 'lon']] = pd.DataFrame(df['.geo'].apply(extract_geo).tolist(), index=df.index)
    
    # 2. Derived SAR Features
    df['Rainfall'] = np.log1p(df['Rainfall'])
    df['VV_VH_ratio'] = df['VV'] / (df['VH'] + 1e-6)
    df['SAR_Index'] = (df['VV'] - df['VH']) / (df['VV'] + df['VH'] + 1e-6)
    df['VV_VH_diff'] = df['VV'] - df['VH']
    df['VV_VH_sum'] = df['VV'] + df['VH']
    df['NDVI_VH'] = df['NDVI'] * df['VH']
    df['NDWI_VH'] = df['NDWI'] * df['VH']
    df['NDVI_NDWI_ratio'] = df['NDVI'] / (df['NDWI'] + 1e-6)

    features_to_scale = [
        'Elevation', 'LST', 'NDVI', 'NDWI', 'Rainfall',
        'VH', 'VV', 'VV_VH_ratio', 'SAR_Index', 'VV_VH_diff', 
        'VV_VH_sum', 'NDVI_VH', 'NDWI_VH', 'NDVI_NDWI_ratio'
    ]
    
    # 3. Scaling & Encoding
    models_dir = os.path.join(os.path.dirname(__file__), "../models")
    scalers_v3 = joblib.load(os.path.join(models_dir, "scalers_v3.pkl"))
    coord_scaler = joblib.load(os.path.join(models_dir, "coord_scaler_v3.pkl"))
    le_v3 = joblib.load(os.path.join(models_dir, "label_encoder_v3.pkl"))

    # Global Coordinate Scaling
    df[['lat', 'lon']] = coord_scaler.transform(df[['lat', 'lon']])

    if region_name in scalers_v3:
        print(f"-> [INFO] Using pre-trained Robust scaler for {region_name}")
        X_scaled = scalers_v3[region_name].transform(df[features_to_scale])
        region_encoded = le_v3.transform([region_name])[0]
    else:
        print(f"-> [WARNING] Region '{region_name}' not in training set. Calculating new Robust statistics...")
        new_scaler = RobustScaler()
        X_scaled = new_scaler.fit_transform(df[features_to_scale])
        try:
            region_encoded = le_v3.transform(['Kaveri'])[0]
        except:
            region_encoded = 0

    X_final = pd.DataFrame(X_scaled, columns=features_to_scale)
    X_final[['lat', 'lon']] = df[['lat', 'lon']].values
    X_final['region_encoded'] = region_encoded
    y = df['soil_moisture'].values

    # 4. Load Models and Predict
    rf_model = joblib.load(os.path.join(models_dir, "rf_model_v3.pkl"))
    gb_model = joblib.load(os.path.join(models_dir, "gb_model_v3.pkl"))
    ann_model = tf.keras.models.load_model(os.path.join(models_dir, "ann_model_v3.keras"))
    meta_learner = joblib.load(os.path.join(models_dir, "meta_learner_v3.pkl"))
    
    y_pred_rf = rf_model.predict(X_final)
    y_pred_gb = gb_model.predict(X_final)
    y_pred_ann = ann_model.predict(X_final, verbose=0).flatten()
    
    meta_features = np.column_stack((y_pred_rf, y_pred_gb, y_pred_ann))
    y_pred_stacked = meta_learner.predict(meta_features)

    # 5. Metrics
    def get_metrics(y_true, y_pred, model_name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return {"Model": model_name, "RMSE": round(rmse, 4), "R2": round(r2, 4), "MAE": round(mae, 4)}

    results = [
        get_metrics(y, y_pred_rf, "Random Forest (V3)"),
        get_metrics(y, y_pred_gb, "Gradient Boosting (V3)"),
        get_metrics(y, y_pred_ann, "ANN (V3)"),
        get_metrics(y, y_pred_stacked, "Stacked Ensemble (V3)")
    ]
    
    print(f"\nV3 Evaluation Results for {region_name}:")
    print(pd.DataFrame(results).to_string(index=False))

    results_dir = os.path.join(os.path.dirname(__file__), "../results")
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y, y=y_pred_stacked, scatter_kws={'alpha': 0.3, 'color': 'purple'}, line_kws={'color': 'black'})
    plt.title(f"{region_name} V3 Test: Actual vs Predicted (Stacked Model)\nRMSE: {results[3]['RMSE']}, R2: {results[3]['R2']}")
    plt.savefig(os.path.join(results_dir, f"{region_name}_V3_Results.png"))
    print(f"\nPlot saved to results/{region_name}_V3_Results.png")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/Raw 2/WestBengal_TEST.csv"
    evaluate_external_v3(path)
