# 🌍 Soil Moisture Retrieval using SAR & Machine Learning

A multi-region soil moisture prediction system using Sentinel-1 SAR data, optical indices, meteorological features, and machine learning models. This project focuses on **cross-regional generalization** and **data fusion** for high-resolution soil moisture estimation.

---

## 📌 Project Overview

Soil moisture is a critical parameter for agriculture, hydrology, and climate studies. Traditional satellite products like SMAP provide coarse-resolution data (~9 km), which is not suitable for field-level applications.

This project aims to:

- Use **Sentinel-1 SAR (VV, VH)** for high-resolution inputs  
- Fuse data from **Sentinel-2, MODIS, and GPM**  
- Train ML models to predict soil moisture  
- Evaluate **generalization across unseen regions**

---

## 🧠 Key Features

- 🌐 Multi-region dataset (Bihar, Kerala, Punjab, Rajasthan, Maharashtra, Dharwad, Kaveri Delta)
- 📡 Satellite data fusion:
  - Sentinel-1 (SAR)
  - Sentinel-2 (NDVI, NDWI)
  - MODIS (LST)
  - GPM (Rainfall)
  - SMAP (Ground truth)
- ⚙️ Advanced feature engineering:
  - NDVI, NDWI
  - SAR ratios (VV/VH)
  - Interaction features:
    - NDWI × Rainfall
    - LST × NDVI
- 🧪 Multiple ML models:
  - Random Forest
  - XGBoost
  - LightGBM
  - Stacked Ensemble (Ridge)
- 🌍 Cross-region evaluation (unseen data testing)


## ⚙️ Pipeline Workflow

### 1️⃣ Data Collection (Google Earth Engine)
- Extract satellite features:
  - VV, VH (Sentinel-1)
  - NDVI, NDWI (Sentinel-2)
  - LST (MODIS)
  - Rainfall (GPM)
  - Soil Moisture (SMAP)

---

### 2️⃣ Preprocessing
- Merge all regional datasets
- Handle missing values
- Filter noisy samples

---

### 3️⃣ Feature Engineering
- Generate SAR-based features:
  - VV/VH ratio
  - VV - VH
- Add interaction features:
  - NDWI × Rainfall
  - LST × NDVI
- Apply normalization:
  - RobustScaler
  - Log transform on rainfall

---

### 4️⃣ Model Training
- Train multiple models:
  - Random Forest
  - XGBoost
  - LightGBM
- Build stacked ensemble using Ridge regression

---

### 5️⃣ Evaluation
- Internal validation (train/test split)
- External validation on unseen region (e.g., West Bengal)

---

## 📊 Results Summary

### 🔹 Internal Performance

| Model | R² Score |
|------|--------|
| Random Forest | ~0.88 |
| XGBoost | ~0.86 |
| LightGBM | ~0.85 |
| Stacked Ensemble | ~0.88 |

---

### 🌍 External Generalization (Unseen Region)

| Version | R² Score |
|--------|--------|
| Initial Model | -1.86 ❌ |
| After Scaling Fix | +0.22 ✅ |
| Final Model | **+0.24** ✅ |

---

## ⚠️ Limitations

- High noise in satellite-derived data  
- Resolution mismatch:
  - Sentinel-1 (~10m) vs SMAP (~9km)  
- Rainfall variability introduces uncertainty  
- Domain shift across regions  

---

## 🧠 Key Insights

- Proper **scaling and normalization** is critical  
- **Feature engineering is more important than model complexity**  
- Cross-region generalization is the main challenge  
- Performance is limited by **data quality, not model capacity**

---

## 🛠️ Installation

```bash
git clone https://github.com/Keerthivarman316/Soil-Moisture-Retrieval-SAR-
cd Soil-Moisture-Retrieval-SAR-
pip install -r requirements.txt
```
## Usage
```bash
python src/preprocessing.py
python src/feature_engineering.py
python src/train_models.py
python src/evaluate.py
```
External evaluation:
```bash
python src/evaluate_external.py
