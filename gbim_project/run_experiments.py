"""
Script to run experiments with different imputation models on synthetic datasets.
"""

import os
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.imputation_models import (
    StandardGBImputer,
    IndicatorGBImputer,
    PatternClusterGBImputer,
    SimplifiedEmbeddingGBImputer
)

# Create results directory if it doesn't exist
os.makedirs('/home/ubuntu/gbim_project/results', exist_ok=True)

# Create figures directory if it doesn't exist
os.makedirs('/home/ubuntu/gbim_project/figures', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Define experiment configurations
experiment_configs = [
    {
        'name': 'mcar_10',
        'data_file': '/home/ubuntu/gbim_project/data/synthetic_mcar_10.csv',
        'mask_file': '/home/ubuntu/gbim_project/data/synthetic_mcar_10_mask.csv',
        'description': 'MCAR 10% missingness'
    },
    {
        'name': 'mcar_30',
        'data_file': '/home/ubuntu/gbim_project/data/synthetic_mcar_30.csv',
        'mask_file': '/home/ubuntu/gbim_project/data/synthetic_mcar_30_mask.csv',
        'description': 'MCAR 30% missingness'
    },
    {
        'name': 'mcar_50',
        'data_file': '/home/ubuntu/gbim_project/data/synthetic_mcar_50.csv',
        'mask_file': '/home/ubuntu/gbim_project/data/synthetic_mcar_50_mask.csv',
        'description': 'MCAR 50% missingness'
    },
    {
        'name': 'mar_30',
        'data_file': '/home/ubuntu/gbim_project/data/synthetic_mar_30.csv',
        'mask_file': '/home/ubuntu/gbim_project/data/synthetic_mar_30_mask.csv',
        'description': 'MAR 30% missingness'
    },
    {
        'name': 'mnar_30',
        'data_file': '/home/ubuntu/gbim_project/data/synthetic_mnar_30.csv',
        'mask_file': '/home/ubuntu/gbim_project/data/synthetic_mnar_30_mask.csv',
        'description': 'MNAR 30% missingness'
    },
    {
        'name': 'mixed_30',
        'data_file': '/home/ubuntu/gbim_project/data/synthetic_mixed_30.csv',
        'mask_file': '/home/ubuntu/gbim_project/data/synthetic_mixed_30_mask.csv',
        'description': 'Mixed 30% missingness'
    }
]

# Define imputation models to test
imputation_models = [
    {
        'name': 'Standard GBIM',
        'model': StandardGBImputer(boosting_model='lightgbm', max_iter=3, verbose=True)
    },
    {
        'name': 'Indicator GBIM',
        'model': IndicatorGBImputer(boosting_model='lightgbm', max_iter=3, verbose=True)
    },
    {
        'name': 'Pattern Cluster GBIM',
        'model': PatternClusterGBImputer(n_clusters=10, boosting_model='lightgbm', max_iter=3, verbose=True)
    },
    {
        'name': 'Simplified Embedding GBIM',
        'model': SimplifiedEmbeddingGBImputer(embedding_dim=16, boosting_model='lightgbm', max_iter=3, verbose=True)
    }
]

# Function to evaluate downstream task performance
def evaluate_downstream_task(X_train, X_test, y_train, y_test):
    """
    Evaluate downstream task performance using a random forest regressor.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features.
    X_test : pandas.DataFrame
        Test features.
    y_train : pandas.Series
        Training target.
    y_test : pandas.Series
        Test target.
        
    Returns:
    --------
    dict
        Dictionary with performance metrics.
    """
    # Train a random forest regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

# Results storage
results = []

# Run experiments
for config in experiment_configs:
    print(f"\n{'='*80}")
    print(f"Running experiment: {config['description']}")
    print(f"{'='*80}")
    
    # Load data
    df_missing = pd.read_csv(config['data_file'])
    mask = pd.read_csv(config['mask_file'])
    
    # Load complete data (ground truth)
    df_complete = pd.read_csv('/home/ubuntu/gbim_project/data/synthetic_complete.csv')
    
    # Split data into features and target
    X_missing = df_missing.drop('target', axis=1)
    y = df_missing['target']
    X_complete = df_complete.drop('target', axis=1)
    
    # Split data into train and test sets
    X_train_missing, X_test_missing, y_train, y_test, mask_train, mask_test = train_test_split(
        X_missing, y, mask, test_size=0.2, random_state=42
    )
    
    X_train_complete, X_test_complete = train_test_split(
        X_complete, test_size=0.2, random_state=42
    )
    
    # Run each imputation model
    for model_config in imputation_models:
        print(f"\nRunning {model_config['name']}...")
        
        # Start timer
        start_time = time.time()
        
        # Fit and transform
        imputer = model_config['model']
        X_train_imputed = imputer.fit_transform(X_train_missing)
        X_test_imputed = imputer.transform(X_test_missing)
        
        # End timer
        end_time = time.time()
        runtime = end_time - start_time
        
        # Evaluate imputation accuracy
        train_metrics = {
            'RMSE': np.sqrt(mean_squared_error(
                X_train_complete.values[mask_train.astype(bool).values], 
                X_train_imputed.values[mask_train.astype(bool).values]
            )),
            'MAE': mean_absolute_error(
                X_train_complete.values[mask_train.astype(bool).values], 
                X_train_imputed.values[mask_train.astype(bool).values]
            )
        }
        
        test_metrics = {
            'RMSE': np.sqrt(mean_squared_error(
                X_test_complete.values[mask_test.astype(bool).values], 
                X_test_imputed.values[mask_test.astype(bool).values]
            )),
            'MAE': mean_absolute_error(
                X_test_complete.values[mask_test.astype(bool).values], 
                X_test_imputed.values[mask_test.astype(bool).values]
            )
        }
        
        # Evaluate downstream task
        downstream_metrics = evaluate_downstream_task(
            X_train_imputed, X_test_imputed, y_train, y_test
        )
        
        # Store results
        result = {
            'Experiment': config['name'],
            'Description': config['description'],
            'Model': model_config['name'],
            'Train_RMSE': train_metrics['RMSE'],
            'Train_MAE': train_metrics['MAE'],
            'Test_RMSE': test_metrics['RMSE'],
            'Test_MAE': test_metrics['MAE'],
            'Downstream_RMSE': downstream_metrics['RMSE'],
            'Downstream_MAE': downstream_metrics['MAE'],
            'Downstream_R2': downstream_metrics['R2'],
            'Runtime': runtime
        }
        
        results.append(result)
        
        # Print results
        print(f"Train RMSE: {train_metrics['RMSE']:.4f}, MAE: {train_metrics['MAE']:.4f}")
        print(f"Test RMSE: {test_metrics['RMSE']:.4f}, MAE: {test_metrics['MAE']:.4f}")
        print(f"Downstream RMSE: {downstream_metrics['RMSE']:.4f}, R2: {downstream_metrics['R2']:.4f}")
        print(f"Runtime: {runtime:.2f} seconds")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results
results_df.to_csv('/home/ubuntu/gbim_project/results/imputation_results.csv', index=False)

print("\nAll experiments completed successfully!")
print(f"Results saved to: /home/ubuntu/gbim_project/results/imputation_results.csv")
