# Soil Moisture Retrieval Using Dual-Frequency SAR

## 📌 Overview
This project focuses on estimating soil moisture under vegetated conditions using dual-frequency SAR data (VV, VH) and vegetation indices.

We use satellite data from:
- Sentinel-1 (SAR)
- Sentinel-2 (NDVI)
- SMAP (Soil Moisture)

## 🚀 Workflow

GEE → Data Collection → CSV Export  
→ Preprocessing → Feature Engineering  
→ Random Forest Model → Evaluation  
→ (Optional) ANN Comparison  

## 📂 Project Structure

- `data/` → raw and processed datasets  
- `gee/` → Google Earth Engine scripts  
- `src/` → ML pipeline code  
- `notebooks/` → experiments and visualization  
- `results/` → graphs and metrics  
- `docs/` → report and methodology  

## 🧠 Models Used

- Random Forest (Primary)
- Artificial Neural Network (Optional comparison)

## 📊 Evaluation Metrics

- RMSE
- R² Score

## ⚙️ Tech Stack

- Python (NumPy, Pandas, Scikit-learn)
- Google Earth Engine
- Matplotlib / Seaborn

## 📈 Future Work

- Deep Learning models (CNN/LSTM)
- Multi-frequency SAR analysis
- Larger geographic coverage

## 👤 Author

S K Keerthi Varman
