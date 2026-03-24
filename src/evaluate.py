import pandas as pd
import joblib
import os
import warnings
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split 
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../data/Processed/Kaveri_Delta_Soil_Moisture.csv")
model_path = os.path.join(script_dir, "../models/random_forest_model.pkl")

df = pd.read_csv(data_path)
x = df[['VV', 'VH', 'NDVI', 'VV_VH_ratio', 'NDVI_VV']]
y = df['soil_moisture']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = joblib.load(model_path)
y_pred = model.predict(X_test)
try:
    rmse = mean_squared_error(y_test, y_pred, squared=False)
except TypeError:
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)