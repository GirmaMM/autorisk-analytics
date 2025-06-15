# scripts/data_cleaning.py

import pandas as pd
import numpy as np
import os

def load_raw_data(filepath):
    df = pd.read_csv(filepath, sep="|", low_memory=False)
    print(f"Loaded data with shape: {df.shape}")
    return df

def preprocess_data(df):
    # Convert dates
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')

    # Drop columns with >90% missing or no variance
    drop_cols = [col for col in df.columns if df[col].isna().mean() > 0.9 or df[col].nunique() <= 1]
    df.drop(columns=drop_cols, inplace=True)

    # Flag negative premium and claims
    df['HasNegativePremium'] = df['TotalPremium'] < 0
    df['HasNegativeClaims'] = df['TotalClaims'] < 0

    # Adjust negatives: Optional—replace negatives with NaN or absolute values
    df.loc[df['TotalPremium'] < 0, 'TotalPremium'] = np.nan
    df.loc[df['TotalClaims'] < 0, 'TotalClaims'] = np.nan  # Could also impute 0

    # Handle missing values in key fields
    df['CustomValueEstimate'] = df.groupby('VehicleType')['CustomValueEstimate'].transform(lambda x: x.fillna(x.median()))
    df['Gender'] = df['Gender'].fillna("Unknown")

    # Loss Ratio calculation (avoid divide-by-zero errors)
    df['LossRatio'] = df['TotalClaims'] / (df['TotalPremium'] + 1e-9)  # Small offset to prevent infinity
    df['LossRatio'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Log-transform key financial values for better distribution
    for col in ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']:
        df[f'log_{col}'] = np.log1p(df[col])  # log1p(x) = log(1 + x) to handle zeros
    
    print(f"Remaining columns after cleaning: {df.shape[1]}")
    return df

def save_interim(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")