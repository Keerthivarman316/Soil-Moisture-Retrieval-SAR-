import pandas as pd
import os
import glob

def preprocess_v3():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(script_dir, "../data/Raw 2")
    cleaned_dir = os.path.join(script_dir, "../data/Cleaned")
    os.makedirs(cleaned_dir, exist_ok=True)

    all_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    combined_data = []

    # Features strictly required for V3 Stacked Model
    cols_v3 = ['Elevation', 'LST', 'NDVI', 'NDWI', 'Rainfall', 'VH', 'VV', 'soil_moisture', '.geo']

    print(f"Checking {len(all_files)} datasets for V3 columns...")

    for file_path in all_files:
        name = os.path.basename(file_path)
        # Extract region name (e.g., 'Bihar' from 'Bihar_FINAL.csv')
        region_name = name.split('_')[0]
        
        df = pd.read_csv(file_path)
        
        # Check if all required columns exist
        if all(col in df.columns for col in cols_v3):
            df = df[cols_v3].dropna()
            df['region'] = region_name
            combined_data.append(df)
            print(f"-> [KEEP] {name} as Region '{region_name}': {len(df)} samples")
        else:
            missing = [col for col in cols_v3 if col not in df.columns]
            print(f"-> [SKIP] {name}: Missing {missing}")

    if combined_data:
        final_df = pd.concat(combined_data, ignore_index=True)
        out_path = os.path.join(cleaned_dir, "V3_Combined.csv")
        final_df.to_csv(out_path, index=False)
        print(f"Combined data saved to: {out_path} ({len(final_df)} samples)")
    else:
        print("Error: No datasets matched the V3 criteria.")

if __name__ == "__main__":
    preprocess_v3()
