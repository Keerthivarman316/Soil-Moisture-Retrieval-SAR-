import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_v3():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(script_dir, "../data/Processed/Test_Split_V3.csv")
    models_dir = os.path.join(script_dir, "../models")
    results_dir = os.path.join(script_dir, "../results")
    os.makedirs(results_dir, exist_ok=True)

    df_test = pd.read_csv(test_path)
    X_test = df_test.drop(columns=['soil_moisture'])
    y_test = df_test['soil_moisture']

    # Load All V3 Models
    rf_model = joblib.load(os.path.join(models_dir, "rf_model_v3.pkl"))
    gb_model = joblib.load(os.path.join(models_dir, "gb_model_v3.pkl"))
    ann_model = tf.keras.models.load_model(os.path.join(models_dir, "ann_model_v3.keras"))
    meta_learner = joblib.load(os.path.join(models_dir, "meta_learner_v3.pkl"))

    print("Generating V3 Ensemble Predictions...")
    y_pred_rf = rf_model.predict(X_test)
    y_pred_gb = gb_model.predict(X_test)
    y_pred_ann = ann_model.predict(X_test, verbose=0).flatten()
    
    # Stacked Model Prediction
    meta_features = np.column_stack((y_pred_rf, y_pred_gb, y_pred_ann))
    y_pred_stacked = meta_learner.predict(meta_features)

    def get_metrics(y_true, y_pred, model_name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return {"Model": model_name, "RMSE": rmse, "R2": r2, "MAE": mae}

    results = [
        get_metrics(y_test, y_pred_rf, "Random Forest (V3)"),
        get_metrics(y_test, y_pred_gb, "Gradient Boosting (V3)"),
        get_metrics(y_test, y_pred_ann, "ANN (V3)"),
        get_metrics(y_test, y_pred_stacked, "Stacked Ensemble (V3)")
    ]

    results_df = pd.DataFrame(results)
    print("\nV3 Metrics Comparison (Held-out Test Split):")
    print(results_df.to_string(index=False))

    # Save results to text file
    with open(os.path.join(results_dir, "v3_results.txt"), "w") as f:
        f.write("V3 Model Evaluation Results\n")
        f.write("===========================\n")
        f.write(results_df.to_string(index=False))

    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_test, y=y_pred_stacked, scatter_kws={'alpha': 0.3, 'color': 'blue'}, line_kws={'color': 'black'})
    plt.title(f"V3 Stacked Model: Actual vs Predicted\nRMSE: {results[3]['RMSE']:.4f}, R2: {results[3]['R2']:.4f}")
    plt.xlabel("Actual Soil Moisture")
    plt.ylabel("Predicted Soil Moisture")
    plt.savefig(os.path.join(results_dir, "V3_Stacked_Plot.png"))
    print(f"\nPlot saved to results/V3_Stacked_Plot.png")

if __name__ == "__main__":
    evaluate_v3()
