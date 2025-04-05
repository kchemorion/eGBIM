"""
Script to generate synthetic datasets for experiments with different missingness mechanisms.
"""

import os
import numpy as np
import pandas as pd
from src.data_generator import (
    generate_synthetic_data,
    introduce_mcar_missingness,
    introduce_mar_missingness,
    introduce_mnar_missingness,
    create_dataset_with_mixed_missingness
)

# Create data directory if it doesn't exist
os.makedirs('/home/ubuntu/gbim_project/data', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Generate base synthetic dataset
print("Generating base synthetic dataset...")
df_complete = generate_synthetic_data(n_samples=10000, n_features=50, random_state=42)
df_complete.to_csv('/home/ubuntu/gbim_project/data/synthetic_complete.csv', index=False)
print(f"Complete dataset shape: {df_complete.shape}")

# Generate datasets with MCAR missingness at different rates
for missing_rate in [0.1, 0.3, 0.5]:
    print(f"Generating MCAR dataset with {missing_rate*100}% missingness...")
    df_missing, mask = introduce_mcar_missingness(
        df_complete, missing_rate=missing_rate, random_state=42
    )
    df_missing.to_csv(f'/home/ubuntu/gbim_project/data/synthetic_mcar_{int(missing_rate*100)}.csv', index=False)
    mask.to_csv(f'/home/ubuntu/gbim_project/data/synthetic_mcar_{int(missing_rate*100)}_mask.csv', index=False)
    print(f"MCAR {missing_rate*100}% dataset missing values: {df_missing.isna().sum().sum()}")

# Generate datasets with MAR missingness
print("Generating MAR dataset...")
df_missing, mask = introduce_mar_missingness(
    df_complete, missing_rate=0.3, n_causing_cols=3, random_state=42
)
df_missing.to_csv('/home/ubuntu/gbim_project/data/synthetic_mar_30.csv', index=False)
mask.to_csv('/home/ubuntu/gbim_project/data/synthetic_mar_30_mask.csv', index=False)
print(f"MAR dataset missing values: {df_missing.isna().sum().sum()}")

# Generate datasets with MNAR missingness
print("Generating MNAR dataset...")
df_missing, mask = introduce_mnar_missingness(
    df_complete, missing_rate=0.3, threshold_percentile=70, random_state=42
)
df_missing.to_csv('/home/ubuntu/gbim_project/data/synthetic_mnar_30.csv', index=False)
mask.to_csv('/home/ubuntu/gbim_project/data/synthetic_mnar_30_mask.csv', index=False)
print(f"MNAR dataset missing values: {df_missing.isna().sum().sum()}")

# Generate dataset with mixed missingness mechanisms
print("Generating mixed missingness dataset...")
df_complete, df_missing, mask = create_dataset_with_mixed_missingness(
    n_samples=10000, n_features=50, missing_rate=0.3, random_state=42
)
df_complete.to_csv('/home/ubuntu/gbim_project/data/synthetic_mixed_complete.csv', index=False)
df_missing.to_csv('/home/ubuntu/gbim_project/data/synthetic_mixed_30.csv', index=False)
mask.to_csv('/home/ubuntu/gbim_project/data/synthetic_mixed_30_mask.csv', index=False)
print(f"Mixed missingness dataset missing values: {df_missing.isna().sum().sum()}")

print("All synthetic datasets generated successfully!")
