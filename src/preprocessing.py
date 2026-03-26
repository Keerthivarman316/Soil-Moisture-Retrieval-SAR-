import pandas as pd
import numpy as np
import os
import glob

def preprocess_v4():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(script_dir, "../data/Raw 2")
    output_path = os.path.join(script_dir, "../data/Cleaned/V3_Combined.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    datasets = []

    for f in all_files:
        name = os.path.basename(f)
        region = name.split('_')[0]
        if "TEST" in name.upper() or "VAL" in name.upper():
            continue
            
        print(f"Loading {name} for region: {region}")
        df = pd.read_csv(f)
        df['region'] = region
        datasets.append(df)

    if not datasets:
        print("No training datasets found!")
        return

    full_df = pd.concat(datasets, ignore_index=True)
    full_df.to_csv(output_path, index=False)
    print(f"Combined V4 dataset saved to {output_path} ({len(full_df)} rows)")

if __name__ == "__main__":
    preprocess_v4()
