"""
Script to create publication-quality plots from experiment results.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from src.visualization import (
    plot_imputation_error_comparison,
    plot_downstream_performance,
    plot_runtime_comparison,
    plot_missingness_rate_comparison,
    plot_missingness_mechanism_comparison,
    create_summary_table
)

# Create figures directory if it doesn't exist
os.makedirs('/home/ubuntu/gbim_project/figures', exist_ok=True)

# Check if results file exists
results_file = '/home/ubuntu/gbim_project/results/imputation_results.csv'
if not os.path.exists(results_file):
    print(f"Results file not found: {results_file}")
    print("Please run the experiments first.")
    exit(1)

# Load results
print("Loading experiment results...")
results_df = pd.read_csv(results_file)

# Print summary statistics
print("\nSummary of Results:")
summary_table = create_summary_table(results_df)
print(summary_table.to_string(index=False))

# Save summary table
summary_table.to_csv('/home/ubuntu/gbim_project/results/summary_table.csv', index=False)
print("Summary table saved to: /home/ubuntu/gbim_project/results/summary_table.csv")

# Create plots
print("\nCreating publication-quality plots...")

# 1. Imputation error comparison (RMSE)
print("Creating imputation error comparison plot (RMSE)...")
plot_imputation_error_comparison(
    results_df, 
    metric='Test_RMSE',
    save_path='/home/ubuntu/gbim_project/figures/imputation_error_rmse.png'
)

# 2. Imputation error comparison (MAE)
print("Creating imputation error comparison plot (MAE)...")
plot_imputation_error_comparison(
    results_df, 
    metric='Test_MAE',
    save_path='/home/ubuntu/gbim_project/figures/imputation_error_mae.png'
)

# 3. Downstream task performance
print("Creating downstream task performance plot...")
plot_downstream_performance(
    results_df,
    metric='Downstream_R2',
    save_path='/home/ubuntu/gbim_project/figures/downstream_performance_r2.png'
)

# 4. Runtime comparison
print("Creating runtime comparison plot...")
plot_runtime_comparison(
    results_df,
    save_path='/home/ubuntu/gbim_project/figures/runtime_comparison.png'
)

# 5. Missingness rate comparison (MCAR at different rates)
print("Creating missingness rate comparison plot...")
plot_missingness_rate_comparison(
    results_df,
    metric='Test_RMSE',
    save_path='/home/ubuntu/gbim_project/figures/missingness_rate_comparison.png'
)

# 6. Missingness mechanism comparison (MCAR vs MAR vs MNAR)
print("Creating missingness mechanism comparison plot...")
plot_missingness_mechanism_comparison(
    results_df,
    metric='Test_RMSE',
    save_path='/home/ubuntu/gbim_project/figures/missingness_mechanism_comparison.png'
)

print("\nAll plots created successfully!")
print("Plots saved to: /home/ubuntu/gbim_project/figures/")
