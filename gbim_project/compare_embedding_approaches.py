"""
Script to compare different embedding approaches for missingness modeling.
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

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import different embedding approaches
from src.embedding_models import (
    PCAEmbeddingGBImputer,
    SimplifiedTransformerGBImputer,
    HybridEmbeddingGBImputer
)

# Import transformer-based approach if PyTorch is available
if TORCH_AVAILABLE:
    from src.embedding_models import TransformerEmbeddingGBImputer

# Create results directory if it doesn't exist
os.makedirs('gbim_project/results', exist_ok=True)

# Create figures directory if it doesn't exist
os.makedirs('gbim_project/figures', exist_ok=True)

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

# Load a subset of data for comparison
print("Loading data for embedding comparison...")
df_complete = pd.read_csv('gbim_project/data/synthetic_complete.csv')
df_missing = pd.read_csv('gbim_project/data/synthetic_mcar_30.csv')
mask = pd.read_csv('gbim_project/data/synthetic_mcar_30_mask.csv')

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

# Define embedding models to compare
embedding_models = [
    {
        'name': 'PCA (dim=8)',
        'model': PCAEmbeddingGBImputer(embedding_dim=8, max_iter=2, verbose=True)
    },
    {
        'name': 'PCA (dim=16)',
        'model': PCAEmbeddingGBImputer(embedding_dim=16, max_iter=2, verbose=True)
    },
    {
        'name': 'PCA (dim=32)',
        'model': PCAEmbeddingGBImputer(embedding_dim=32, max_iter=2, verbose=True)
    },
    {
        'name': 'Simplified Transformer',
        'model': SimplifiedTransformerGBImputer(embedding_dim=16, n_components=3, max_iter=2, verbose=True)
    },
    {
        'name': 'Hybrid Embedding',
        'model': HybridEmbeddingGBImputer(embedding_dim=16, max_iter=2, verbose=True)
    }
]

# Add transformer-based model if PyTorch is available - just one for testing
if TORCH_AVAILABLE:
    transformer_models = [
        {
            'name': 'BERT-style',
            'model': TransformerEmbeddingGBImputer(embedding_dim=16, transformer_type='bert', 
                                                 num_epochs=3, batch_size=32, max_iter=2, verbose=True)
        }
    ]
    embedding_models.extend(transformer_models)

# Results storage
results = []

# Run comparison
print("\nComparing different embedding approaches...")

for model_config in embedding_models:
    print(f"\nEvaluating {model_config['name']}...")
    
    # Fit and transform
    model = model_config['model']
    X_train_imputed = model.fit_transform(X_train_missing)
    X_test_imputed = model.transform(X_test_missing)
    
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
        'Model': model_config['name'],
        'Train_RMSE': train_rmse,
        'Test_RMSE': test_rmse,
        'Train_MAE': train_mae,
        'Test_MAE': test_mae,
        'Downstream_RMSE': downstream_rmse,
        'Downstream_R2': downstream_r2
    })
    
    # Print results
    print(f"Train RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
    print(f"Downstream RMSE: {downstream_rmse:.4f}, R2: {downstream_r2:.4f}")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results
results_df.to_csv('gbim_project/results/embedding_comparison_results.csv', index=False)

print("\nEmbedding Comparison Summary:")
print(results_df.to_string(index=False))

# Create plots
print("\nCreating embedding comparison plots...")

# 1. Imputation Error Comparison (RMSE)
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Model', y='Test_RMSE', data=results_df)
plt.title('Imputation Error Comparison (RMSE) for Different Embedding Approaches')
plt.xlabel('Embedding Approach')
plt.ylabel('Test RMSE')
plt.xticks(rotation=45)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
for container in ax.containers:
    ax.bar_label(container, fmt='%.4f', fontsize=10)
plt.tight_layout()
plt.savefig('gbim_project/figures/embedding_approach_comparison.png', bbox_inches='tight', dpi=300)

# 2. Downstream Task Performance
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Model', y='Downstream_R2', data=results_df)
plt.title('Downstream Task Performance (R²) for Different Embedding Approaches')
plt.xlabel('Embedding Approach')
plt.ylabel('R²')
plt.xticks(rotation=45)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
for container in ax.containers:
    ax.bar_label(container, fmt='%.4f', fontsize=10)
plt.tight_layout()
plt.savefig('gbim_project/figures/downstream_performance_r2.png', bbox_inches='tight', dpi=300)

# 3. Combined metrics plot
plt.figure(figsize=(14, 10))
metrics = ['Test_RMSE', 'Test_MAE', 'Downstream_R2']
results_melted = pd.melt(results_df, id_vars=['Model'], value_vars=metrics, 
                        var_name='Metric', value_name='Value')

# Normalize values for comparison (min-max scaling)
for metric in metrics:
    min_val = results_df[metric].min()
    max_val = results_df[metric].max()
    if max_val > min_val:
        results_df[f'{metric}_Normalized'] = (results_df[metric] - min_val) / (max_val - min_val)
        # For RMSE and MAE, lower is better, so invert
        if metric in ['Test_RMSE', 'Test_MAE']:
            results_df[f'{metric}_Normalized'] = 1 - results_df[f'{metric}_Normalized']
    else:
        results_df[f'{metric}_Normalized'] = 0.5  # Default if all values are the same

# Melt the normalized metrics
normalized_metrics = [f'{m}_Normalized' for m in metrics]
results_normalized = pd.melt(results_df, id_vars=['Model'], value_vars=normalized_metrics,
                           var_name='Metric', value_name='Normalized_Value')
results_normalized['Metric'] = results_normalized['Metric'].str.replace('_Normalized', '')

# Create heatmap of normalized metrics
pivot_table = results_normalized.pivot(index='Model', columns='Metric', values='Normalized_Value')
plt.figure(figsize=(12, 8))
ax = sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.2f', cbar_kws={'label': 'Performance (higher is better)'})
plt.title('Normalized Performance Metrics for Different Embedding Approaches')
plt.tight_layout()
plt.savefig('gbim_project/figures/embedding_comparison_heatmap.png', bbox_inches='tight', dpi=300)

print("Embedding comparison plots created successfully!")
print("Results saved to: gbim_project/results/embedding_comparison_results.csv")
print("Plots saved to: gbim_project/figures/")