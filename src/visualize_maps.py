import pandas as pd
import numpy as np
import os
import joblib
import json
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata

def visualize_maps_v4(file_path, grid_resolution=100):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    name = os.path.basename(file_path)
    region_name = name.split('_')[0]
    print(f"=== V4 Spatial Mapping: {region_name} ===")

    df = pd.read_csv(file_path)
    cols_needed = ['Elevation', 'LST', 'NDVI', 'NDWI', 'Rainfall', 'VH', 'VV', 'soil_moisture', '.geo']
    df = df[cols_needed].dropna()

    def extract_geo(geo_str):
        try:
            coords = json.loads(geo_str)['coordinates']
            return coords[1], coords[0]
        except:
            return np.nan, np.nan

    df[['lat', 'lon']] = pd.DataFrame(df['.geo'].apply(extract_geo).tolist(), index=df.index)
    
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    lon_min, lon_max = df['lon'].min(), df['lon'].max()
    
    grid_lat, grid_lon = np.mgrid[lat_min:lat_max:complex(grid_resolution), 
                                  lon_min:lon_max:complex(grid_resolution)]
    
    print("Interpolating spatial grids...")
    points = df[['lat', 'lon']].values
    
    grid_truth = griddata(points, df['soil_moisture'].values, (grid_lat, grid_lon), method='cubic')
    grid_vv = griddata(points, df['VV'].values, (grid_lat, grid_lon), method='cubic')
    grid_vh = griddata(points, df['VH'].values, (grid_lat, grid_lon), method='cubic')
    
    grid_features = {}
    for col in ['Elevation', 'LST', 'NDVI', 'NDWI', 'Rainfall', 'VH', 'VV']:
        grid_features[col] = griddata(points, df[col].values, (grid_lat, grid_lon), method='cubic')
    
    flat_features = pd.DataFrame({col: grid_features[col].flatten() for col in grid_features})
    mask = ~flat_features.isna().any(axis=1)
    df_grid = flat_features[mask].copy()
    
    df_grid['lat'] = grid_lat.flatten()[mask]
    df_grid['lon'] = grid_lon.flatten()[mask]

    print("Engineering features on the grid...")
    df_grid['Rainfall'] = np.log1p(df_grid['Rainfall'])
    df_grid['VV_VH_ratio'] = df_grid['VV'] / (df_grid['VH'] + 1e-6)
    df_grid['SAR_Index'] = (df_grid['VV'] - df_grid['VH']) / (df_grid['VV'] + df_grid['VH'] + 1e-6)
    df_grid['NDWI_Rain'] = df_grid['NDWI'] * df_grid['Rainfall']
    df_grid['LST_NDVI'] = df_grid['LST'] * df_grid['NDVI']
    df_grid['VV_NDWI'] = df_grid['VV'] * df_grid['NDWI']
    df_grid['VH_NDVI'] = df_grid['VH'] * df_grid['NDVI']
    df_grid['Rain_LST'] = df_grid['Rainfall'] / (df_grid['LST'] + 1.0)

    features_to_scale = [
        'Elevation', 'LST', 'NDVI', 'NDWI', 'Rainfall',
        'VH', 'VV', 'VV_VH_ratio', 'SAR_Index', 
        'NDWI_Rain', 'LST_NDVI', 'VV_NDWI', 'VH_NDVI', 'Rain_LST'
    ]

    models_dir = os.path.join(os.path.dirname(__file__), "../models")
    scalers_v4 = joblib.load(os.path.join(models_dir, "scalers_v4.pkl"))
    coord_scaler = joblib.load(os.path.join(models_dir, "coord_scaler_v4.pkl"))
    region_cols = joblib.load(os.path.join(models_dir, "region_cols_v4.pkl"))
    
    fallback_region = list(scalers_v4.keys())[0]
    scaler = scalers_v4.get(region_name, scalers_v4[fallback_region])
    
    X_scaled = scaler.transform(df_grid[features_to_scale])
    X_final = pd.DataFrame(X_scaled, columns=features_to_scale)
    
    coords_scaled = coord_scaler.transform(df_grid[['lat', 'lon']])
    X_final['lat'] = coords_scaled[:, 0]
    X_final['lon'] = coords_scaled[:, 1]
    
    for col in region_cols:
        target_name = f"region_{region_name}"
        X_final[col] = 1 if col == target_name else 0

    print("Running ensemble inference on spatial grid...")
    rf = joblib.load(os.path.join(models_dir, "rf_model_v4.pkl"))
    xgb_m = joblib.load(os.path.join(models_dir, "xgb_model_v4.pkl"))
    lgb_m = joblib.load(os.path.join(models_dir, "lgb_model_v4.pkl"))
    meta = joblib.load(os.path.join(models_dir, "meta_learner_v4.pkl"))
    
    X_final = X_final[features_to_scale + ['lat', 'lon'] + region_cols]

    p_rf = rf.predict(X_final)
    p_xgb = xgb_m.predict(X_final)
    p_lgb = lgb_m.predict(X_final)
    
    p_stacked = meta.predict(np.column_stack((p_rf, p_xgb, p_lgb)))
    
    grid_pred = np.full(grid_lat.shape, np.nan)
    grid_pred.ravel()[mask] = p_stacked

    print("Generating 3-panel visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    def norm(x): 
        x_min, x_max = np.nanmin(x), np.nanmax(x)
        return (x - x_min) / (x_max - x_min + 1e-6)
    
    sar_rgb = np.stack([norm(grid_vv), norm(grid_vh), norm(grid_vv/(grid_vh + 1e-6))], axis=-1)
    axes[0].imshow(sar_rgb, origin='lower')
    axes[0].set_title(f"Raw Satellite (SAR Composite)\nRegion: {region_name}")
    axes[0].axis('off')

    im2 = axes[1].imshow(grid_pred, origin='lower', cmap='YlGnBu')
    plt.colorbar(im2, ax=axes[1], label="Moisture (%)")
    axes[1].set_title("V4 Predicted Moisture Map")
    axes[1].axis('off')

    im3 = axes[2].imshow(grid_truth, origin='lower', cmap='YlGnBu')
    plt.colorbar(im3, ax=axes[2], label="Moisture (%)")
    axes[2].set_title("Ground Truth Moisture Map")
    axes[2].axis('off')

    plt.tight_layout()
    results_dir = os.path.join(os.path.dirname(__file__), "../results")
    out_path = os.path.join(results_dir, f"{region_name}_V4_Map_Comparison.png")
    plt.savefig(out_path, dpi=300)
    print(f"Success! Map saved to {out_path}")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/Raw 2/WestBengal_TEST.csv"
    visualize_maps_v4(path)
