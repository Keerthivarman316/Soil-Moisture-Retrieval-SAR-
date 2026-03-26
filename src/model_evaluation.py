import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_v4():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(script_dir, "../data/Processed/Test_Split_V4.csv")
    models_dir = os.path.join(script_dir, "../models")
    results_dir = os.path.join(script_dir, "../results")
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(test_path):
        print(f"Error: {test_path} not found.")
        return

    df_test = pd.read_csv(test_path)
    X_test = df_test.drop(columns=['soil_moisture', 'region'])
    y_test = df_test['soil_moisture']

    print("Loading V4 Models (RF, XGB, LGBM, Ridge)...")
    rf_model = joblib.load(os.path.join(models_dir, "rf_model_v4.pkl"))
    xgb_model = joblib.load(os.path.join(models_dir, "xgb_model_v4.pkl"))
    lgb_model = joblib.load(os.path.join(models_dir, "lgb_model_v4.pkl"))
    meta_learner = joblib.load(os.path.join(models_dir, "meta_learner_v4.pkl"))

    print("Generating V4 Ensemble Predictions...")
    y_pred_rf = rf_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_lgb = lgb_model.predict(X_test)
    
    meta_features = np.column_stack((y_pred_rf, y_pred_xgb, y_pred_lgb))
    y_pred_stacked = meta_learner.predict(meta_features)

    def get_metrics(y_true, y_pred, model_name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return {"Model": model_name, "RMSE": rmse, "R2": r2, "MAE": mae}

    results = [
        get_metrics(y_test, y_pred_rf, "Random Forest (V4)"),
        get_metrics(y_test, y_pred_xgb, "XGBoost (V4)"),
        get_metrics(y_test, y_pred_lgb, "LightGBM (V4)"),
        get_metrics(y_test, y_pred_stacked, "Stacked Ensemble (V4)")
    ]

    results_df = pd.DataFrame(results)
    print("\nV4 Metrics Comparison (Held-out Test Split):")
    print(results_df.to_string(index=False))

    with open(os.path.join(results_dir, "v4_results.txt"), "w") as f:
        f.write("V4 Model Evaluation Results (Region-Aware)\n")
        f.write("=========================================\n")
        f.write(results_df.to_string(index=False))

    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_test, y=y_pred_stacked, scatter_kws={'alpha': 0.3, 'color': 'green'}, line_kws={'color': 'black'})
    plt.title(f"V4 Stacked Model (XGB+RF+LGBM): Actual vs Predicted\nRMSE: {results[3]['RMSE']:.4f}, R2: {results[3]['R2']:.4f}")
    plt.xlabel("Actual Soil Moisture")
    plt.ylabel("Predicted Soil Moisture")
    plt.savefig(os.path.join(results_dir, "V4_Stacked_Plot.png"))
    print(f"\nPlot saved to results/V4_Stacked_Plot.png")

if __name__ == "__main__":
    evaluate_v4()
