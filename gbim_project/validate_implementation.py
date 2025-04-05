"""
Script to validate the implementation against the paper's methodology.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.imputation_models import (
    StandardGBImputer,
    IndicatorGBImputer,
    PatternClusterGBImputer,
    SimplifiedEmbeddingGBImputer
)

# Create validation directory if it doesn't exist
os.makedirs('/home/ubuntu/gbim_project/validation', exist_ok=True)

# Load a small subset of data for quick validation
print("Loading data for validation...")
df_complete = pd.read_csv('/home/ubuntu/gbim_project/data/synthetic_complete.csv')
df_missing = pd.read_csv('/home/ubuntu/gbim_project/data/synthetic_mcar_30.csv')
mask = pd.read_csv('/home/ubuntu/gbim_project/data/synthetic_mcar_30_mask.csv')

# Use only first 1000 rows for quick validation
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

# Validation checks
validation_results = []

print("\nValidating implementation against paper's methodology...")

# 1. Validate StandardGBImputer
print("\n1. Validating StandardGBImputer...")
model = StandardGBImputer(boosting_model='lightgbm', max_iter=2, verbose=True)
X_train_imputed = model.fit_transform(X_train_missing)

# Check if imputation filled all missing values
missing_after = X_train_imputed.isna().sum().sum()
print(f"Missing values after imputation: {missing_after}")
assert missing_after == 0, "Imputation failed to fill all missing values"

# Check if feature importances are stored
assert hasattr(model, 'feature_importances_'), "Feature importances not stored"
print("Feature importances stored correctly")

# Check if models are stored for each feature
assert hasattr(model, 'models_'), "Models not stored"
print(f"Number of models created: {len(model.models_)}")

# 2. Validate IndicatorGBImputer
print("\n2. Validating IndicatorGBImputer...")
model = IndicatorGBImputer(boosting_model='lightgbm', max_iter=2, verbose=True)
X_train_imputed = model.fit_transform(X_train_missing)

# Check if indicator features are used
# We can't directly check the internal data, but we can verify the model works
missing_after = X_train_imputed.isna().sum().sum()
print(f"Missing values after imputation: {missing_after}")
assert missing_after == 0, "Imputation failed to fill all missing values"

# 3. Validate PatternClusterGBImputer
print("\n3. Validating PatternClusterGBImputer...")
model = PatternClusterGBImputer(n_clusters=5, boosting_model='lightgbm', max_iter=2, verbose=True)
X_train_imputed = model.fit_transform(X_train_missing)

# Check if clustering is performed
assert hasattr(model, 'kmeans_'), "KMeans clustering not performed"
assert hasattr(model, 'cluster_labels_'), "Cluster labels not stored"
print(f"Number of clusters: {model.n_clusters}")
print(f"Cluster distribution: {np.bincount(model.cluster_labels_)}")

# 4. Validate SimplifiedEmbeddingGBImputer
print("\n4. Validating SimplifiedEmbeddingGBImputer...")
model = SimplifiedEmbeddingGBImputer(embedding_dim=8, boosting_model='lightgbm', max_iter=2, verbose=True)
X_train_imputed = model.fit_transform(X_train_missing)

# Check if PCA embedding is performed
assert hasattr(model, 'pca_'), "PCA embedding not performed"
assert hasattr(model, 'embeddings_'), "Embeddings not stored"
print(f"Embedding dimension: {model.embedding_dim}")
print(f"Explained variance ratio: {np.sum(model.pca_.explained_variance_ratio_):.4f}")

# 5. Compare imputation performance
print("\n5. Comparing imputation performance...")

# Initialize models
models = {
    'StandardGBIM': StandardGBImputer(boosting_model='lightgbm', max_iter=2, verbose=False),
    'IndicatorGBIM': IndicatorGBImputer(boosting_model='lightgbm', max_iter=2, verbose=False),
    'PatternClusterGBIM': PatternClusterGBImputer(n_clusters=5, boosting_model='lightgbm', max_iter=2, verbose=False),
    'SimplifiedEmbeddingGBIM': SimplifiedEmbeddingGBImputer(embedding_dim=8, boosting_model='lightgbm', max_iter=2, verbose=False)
}

# Run each model and evaluate
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    
    # Fit and transform
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
    
    # Store results
    validation_results.append({
        'Model': name,
        'Train_RMSE': train_rmse,
        'Test_RMSE': test_rmse
    })
    
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")

# Convert results to DataFrame
validation_df = pd.DataFrame(validation_results)

# Save validation results
validation_df.to_csv('/home/ubuntu/gbim_project/validation/validation_results.csv', index=False)

print("\nValidation Summary:")
print(validation_df.to_string(index=False))

# Check if enhanced models improve over baseline
baseline_rmse = validation_df[validation_df['Model'] == 'StandardGBIM']['Test_RMSE'].values[0]
for model in ['IndicatorGBIM', 'PatternClusterGBIM', 'SimplifiedEmbeddingGBIM']:
    model_rmse = validation_df[validation_df['Model'] == model]['Test_RMSE'].values[0]
    improvement = (baseline_rmse - model_rmse) / baseline_rmse * 100
    print(f"{model} improvement over baseline: {improvement:.2f}%")

print("\nValidation completed successfully!")
print("Results saved to: /home/ubuntu/gbim_project/validation/validation_results.csv")
