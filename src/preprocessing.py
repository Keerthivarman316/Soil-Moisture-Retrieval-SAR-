import pandas as pd
import os
import glob
script_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(script_dir, "../data/Raw")
cleaned_dir = os.path.join(script_dir, "../data/Cleaned")
os.makedirs(cleaned_dir, exist_ok=True)
def preprocess_dataset(file_path):
    name = os.path.basename(file_path)
    df = pd.read_csv(file_path)
    required_cols = ['system:index', '.geo', 'VV', 'VH', 'NDVI', 'soil_moisture']
    df = df[required_cols].dropna()
    sm_max = df['soil_moisture'].max()
    if sm_max > 1.1:
        print(f"-> Scaling {name}: Max value {sm_max:.2f} detected. Converting to 0-1 range.")
        df['soil_moisture'] = df['soil_moisture'] / 100.0
    df = df[(df['soil_moisture'] >= 0) & (df['soil_moisture'] <= 1)].copy()
    clean_name = name.replace(".csv", "_cleaned.csv")
    df.to_csv(os.path.join(cleaned_dir, clean_name), index=False)
    return df
print("Starting Preprocessing...")
all_files = glob.glob(os.path.join(raw_dir, "*.csv"))
cleaned_dfs = []
for f in all_files:
    print(f"Processing: {os.path.basename(f)}")
    df_clean = preprocess_dataset(f)
    cleaned_dfs.append(df_clean)
combined_df = pd.concat(cleaned_dfs, axis=0).reset_index(drop=True)
combined_df.to_csv(os.path.join(cleaned_dir, "Combined_Cleaned.csv"), index=False)
print("-" * 30)
print(f"Preprocessing Complete!")
print(f"Total Samples: {len(combined_df)}")
print(f"Combined data saved to: data/Cleaned/Combined_Cleaned.csv")
