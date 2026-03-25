import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.join(script_dir, "../data/Processed/Test_Split.csv")
models_dir = os.path.join(script_dir, "../models")
results_dir = os.path.join(script_dir, "../results")
os.makedirs(results_dir, exist_ok=True)

print("Starting Model Evaluation...")
df_test = pd.read_csv(test_path)
X_test = df_test.drop(columns=['soil_moisture'])
y_test = df_test['soil_moisture']

# Load Models
rf_model = joblib.load(os.path.join(models_dir, "rf_model.pkl"))
ann_model = tf.keras.models.load_model(os.path.join(models_dir, "ann_model.keras"))

# Predictions
print("-> Generating predictions...")
y_pred_rf = rf_model.predict(X_test)
y_pred_ann = ann_model.predict(X_test, verbose=0).flatten()
# Hybrid Model: Simple Average Ensemble
y_pred_hybrid = (y_pred_rf + y_pred_ann) / 2.0

def calculate_metrics(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {"Model": model_name, "RMSE": rmse, "R2": r2, "MAE": mae}

# Collect Results
results = []
results.append(calculate_metrics(y_test, y_pred_rf, "Random Forest"))
results.append(calculate_metrics(y_test, y_pred_ann, "ANN"))
results.append(calculate_metrics(y_test, y_pred_hybrid, "Hybrid (RF+ANN)"))

results_df = pd.DataFrame(results)
print("\nEvaluation Results Table:")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv(os.path.join(results_dir, "Evaluation_Comparison.csv"), index=False)

# Visualization
print("\n-> Generating comparison plots...")
plt.figure(figsize=(15, 5))

# Subplot 1: Actual vs Predicted (Hybrid)
plt.subplot(1, 2, 1)
sns.regplot(x=y_test, y=y_pred_hybrid, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
plt.title("Actual vs Predicted Soil Moisture (Hybrid Model)")
plt.xlabel("Actual")
plt.ylabel("Predicted")

# Subplot 2: Metric Comparison Bar Chart
plt.subplot(1, 2, 2)
melted_df = results_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
# Filter to just RMSE and MAE for clearer bar chart (R2 is often on a different scale)
bar_df = melted_df[melted_df['Metric'].isin(['RMSE', 'MAE'])]
sns.barplot(data=bar_df, x='Metric', y='Value', hue='Model')
plt.title("Model Performance Comparison (RMSE & MAE)")

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Model_Comparison_Plot.png"))
print(f"Plots saved to: {results_dir}/Model_Comparison_Plot.png")

print("-" * 30)
print("Model Evaluation Complete!")
