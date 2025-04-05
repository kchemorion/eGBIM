"""
Script to compare different boosting algorithms for imputation with missingness modeling.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm
import time

from src.boosting_models import MultiBoostingGBImputer

# Create results directory if it doesn't exist
os.makedirs('/home/ubuntu/gbim_project/results', exist_ok=True)

# Create figures directory if it doesn't exist
os.makedirs('/home/ubuntu/gbim_project/figures', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Set publication-quality plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.title_fontsize'] = 14

# Load a small subset of data for quick comparison
print("Loading data for boosting algorithm comparison...")
df_complete = pd.read_csv('/home/ubuntu/gbim_project/data/synthetic_complete.csv')
df_missing = pd.read_csv('/home/ubuntu/gbim_project/data/synthetic_mcar_30.csv')
mask = pd.read_csv('/home/ubuntu/gbim_project/data/synthetic_mcar_30_mask.csv')

# Use only first 1000 rows for quick comparison
df_complete = df_complete.iloc[:1000]
df_missing = df_missing.iloc[:1000]
mask = mask.iloc[:1000]

# Split data into features and target
X_complete = df_complete.drop('target', axis=1)
X_missing = df_missing.drop('target', axis=1)
y = df_missing['target']

# Split data into train and test sets
X_train_missing, X_test_missing, X_train_complete, X_test_complete, y_train, y_test, mask_train, mask_test = train_test_split(
    X_missing, X_complete, y, mask, test_size=0.2, random_state=42
)

# Define boosting algorithms to compare
boosting_algorithms = ['lightgbm', 'xgboost', 'histgbr']

# Define missingness approaches to compare
missingness_approaches = ['standard', 'indicators', 'pattern_clustering', 'embedding']

# Results storage
results = []

# Run comparison
print("\nComparing different boosting algorithms with various missingness approaches...")

for boosting_algorithm in boosting_algorithms:
    for missingness_approach in missingness_approaches:
        print(f"\nEvaluating {boosting_algorithm} with {missingness_approach} approach...")
        
        # Fit and transform
        model = MultiBoostingGBImputer(
            boosting_model=boosting_algorithm,
            missingness_approach=missingness_approach,
            max_iter=2,
            verbose=True
        )
        
        # Measure runtime
        start_time = time.time()
        X_train_imputed = model.fit_transform(X_train_missing)
        X_test_imputed = model.transform(X_test_missing)
        runtime = time.time() - start_time
        
        # Evaluate imputation accuracy
        train_rmse = np.sqrt(mean_squared_error(
            X_train_complete.values[mask_train.astype(bool).values], 
            X_train_imputed.values[mask_train.astype(bool).values]
        ))
        
        test_rmse = np.sqrt(mean_squared_error(
            X_test_complete.values[mask_test.astype(bool).values], 
            X_test_imputed.values[mask_test.astype(bool).values]
        ))
        
        train_mae = mean_absolute_error(
            X_train_complete.values[mask_train.astype(bool).values], 
            X_train_imputed.values[mask_train.astype(bool).values]
        )
        
        test_mae = mean_absolute_error(
            X_test_complete.values[mask_test.astype(bool).values], 
            X_test_imputed.values[mask_test.astype(bool).values]
        )
        
        # Evaluate downstream task performance
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train_imputed, y_train)
        y_pred = rf.predict(X_test_imputed)
        downstream_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        downstream_r2 = r2_score(y_test, y_pred)
        
        # Store results
        results.append({
            'Boosting_Algorithm': boosting_algorithm,
            'Missingness_Approach': missingness_approach,
            'Train_RMSE': train_rmse,
            'Test_RMSE': test_rmse,
            'Train_MAE': train_mae,
            'Test_MAE': test_mae,
            'Downstream_RMSE': downstream_rmse,
            'Downstream_R2': downstream_r2,
            'Runtime': runtime
        })
        
        # Print results
        print(f"Train RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
        print(f"Downstream RMSE: {downstream_rmse:.4f}, R2: {downstream_r2:.4f}")
        print(f"Runtime: {runtime:.2f} seconds")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results
results_df.to_csv('/home/ubuntu/gbim_project/results/boosting_comparison_results.csv', index=False)

print("\nBoosting Algorithm Comparison Summary:")
print(results_df.to_string(index=False))

# Create plots
print("\nCreating boosting algorithm comparison plots...")

# 1. Imputation Error Comparison (RMSE)
plt.figure(figsize=(14, 10))
ax = sns.barplot(x='Boosting_Algorithm', y='Test_RMSE', hue='Missingness_Approach', data=results_df)
plt.title('Imputation Error Comparison (RMSE) for Different Boosting Algorithms')
plt.xlabel('Boosting Algorithm')
plt.ylabel('Test RMSE')
plt.legend(title='Missingness Approach')
ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
for container in ax.containers:
    ax.bar_label(container, fmt='%.4f', fontsize=10)
plt.tight_layout()
plt.savefig('/home/ubuntu/gbim_project/figures/boosting_comparison_rmse.png', bbox_inches='tight', dpi=300)

# 2. Downstream Task Performance
plt.figure(figsize=(14, 10))
ax = sns.barplot(x='Boosting_Algorithm', y='Downstream_R2', hue='Missingness_Approach', data=results_df)
plt.title('Downstream Task Performance (R²) for Different Boosting Algorithms')
plt.xlabel('Boosting Algorithm')
plt.ylabel('R²')
plt.legend(title='Missingness Approach')
ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
for container in ax.containers:
    ax.bar_label(container, fmt='%.4f', fontsize=10)
plt.tight_layout()
plt.savefig('/home/ubuntu/gbim_project/figures/boosting_comparison_r2.png', bbox_inches='tight', dpi=300)

# 3. Runtime Comparison
plt.figure(figsize=(14, 10))
ax = sns.barplot(x='Boosting_Algorithm', y='Runtime', hue='Missingness_Approach', data=results_df)
plt.title('Runtime Comparison for Different Boosting Algorithms')
plt.xlabel('Boosting Algorithm')
plt.ylabel('Runtime (seconds)')
plt.legend(title='Missingness Approach')
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', fontsize=10)
plt.tight_layout()
plt.savefig('/home/ubuntu/gbim_project/figures/boosting_comparison_runtime.png', bbox_inches='tight', dpi=300)

# 4. Heatmap of algorithm-approach combinations
# Create a pivot table for the heatmap
heatmap_data = pd.pivot_table(
    results_df,
    values='Test_RMSE',
    index='Missingness_Approach',
    columns='Boosting_Algorithm'
)

plt.figure(figsize=(12, 8))
ax = sns.heatmap(
    heatmap_data,
    annot=True,
    fmt='.4f',
    cmap='viridis_r',  # Reversed viridis (lower values = better = darker)
    cbar_kws={'label': 'Test RMSE (lower is better)'}
)
plt.title('Missingness Approach vs Boosting Algorithm Performance')
plt.tight_layout()
plt.savefig('/home/ubuntu/gbim_project/figures/approach_algorithm_heatmap_rmse.png', bbox_inches='tight', dpi=300)

# 5. Heatmap for downstream performance
# Create a pivot table for the heatmap
heatmap_data_r2 = pd.pivot_table(
    results_df,
    values='Downstream_R2',
    index='Missingness_Approach',
    columns='Boosting_Algorithm'
)

plt.figure(figsize=(12, 8))
ax = sns.heatmap(
    heatmap_data_r2,
    annot=True,
    fmt='.4f',
    cmap='viridis',  # Higher values = better = darker
    cbar_kws={'label': 'Downstream R² (higher is better)'}
)
plt.title('Downstream Performance: Missingness Approach vs Boosting Algorithm')
plt.tight_layout()
plt.savefig('/home/ubuntu/gbim_project/figures/approach_algorithm_heatmap_r2.png', bbox_inches='tight', dpi=300)

print("Boosting algorithm comparison plots created successfully!")
print("Results saved to: /home/ubuntu/gbim_project/results/boosting_comparison_results.csv")
print("Plots saved to: /home/ubuntu/gbim_project/figures/")
