"""
Utility functions for the Gradient Boosting Imputation with Missingness Modeling project.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

def create_missingness_indicators(X):
    """
    Create binary indicators for missing values.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        The input data with missing values.
        
    Returns:
    --------
    pandas.DataFrame or numpy.ndarray
        Binary indicators for missing values (1 if missing, 0 if present).
    """
    if isinstance(X, pd.DataFrame):
        return X.isna().astype(int)
    else:
        return np.isnan(X).astype(int)

def create_missingness_pattern_clusters(X, n_clusters=10, random_state=42):
    """
    Create cluster assignments for missingness patterns.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        The input data with missing values.
    n_clusters : int, default=10
        Number of clusters to create.
    random_state : int, default=42
        Random seed for KMeans clustering.
        
    Returns:
    --------
    numpy.ndarray
        Cluster assignments for each sample.
    """
    # Create binary missingness mask
    if isinstance(X, pd.DataFrame):
        mask = X.isna().astype(int).values
    else:
        mask = np.isnan(X).astype(int)
    
    # Apply KMeans clustering to the missingness patterns
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(mask)
    
    return clusters

def preprocess_data(X, standardize=True):
    """
    Preprocess the data by handling initial missing values and standardizing.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        The input data with missing values.
    standardize : bool, default=True
        Whether to standardize the data.
        
    Returns:
    --------
    pandas.DataFrame or numpy.ndarray
        Preprocessed data with initial missing values filled.
    """
    if isinstance(X, pd.DataFrame):
        # Fill missing values with mean for initial preprocessing
        X_filled = X.fillna(X.mean())
        
        if standardize:
            scaler = StandardScaler()
            X_filled = pd.DataFrame(
                scaler.fit_transform(X_filled),
                columns=X.columns,
                index=X.index
            )
    else:
        # Fill missing values with column means
        col_means = np.nanmean(X, axis=0)
        X_filled = X.copy()
        for i, mean_val in enumerate(col_means):
            mask = np.isnan(X[:, i])
            X_filled[mask, i] = mean_val
            
        if standardize:
            scaler = StandardScaler()
            X_filled = scaler.fit_transform(X_filled)
    
    return X_filled

def split_data(X, test_size=0.2, val_size=0.25, random_state=42):
    """
    Split data into training, validation, and test sets.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        The input data.
    test_size : float, default=0.2
        Proportion of data to use for testing.
    val_size : float, default=0.25
        Proportion of training data to use for validation.
    random_state : int, default=42
        Random seed for reproducibility.
        
    Returns:
    --------
    tuple
        (X_train, X_val, X_test) split datasets.
    """
    # First split into train+val and test
    X_train_val, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
    
    # Then split train+val into train and val
    X_train, X_val = train_test_split(X_train_val, test_size=val_size, random_state=random_state)
    
    return X_train, X_val, X_test

def evaluate_imputation(X_true, X_imputed, mask=None):
    """
    Evaluate imputation performance using RMSE and MAE.
    
    Parameters:
    -----------
    X_true : pandas.DataFrame or numpy.ndarray
        The ground truth data without missing values.
    X_imputed : pandas.DataFrame or numpy.ndarray
        The imputed data.
    mask : pandas.DataFrame or numpy.ndarray, optional
        Binary mask indicating which values were missing (1 if missing, 0 if present).
        If None, all values are compared.
        
    Returns:
    --------
    dict
        Dictionary with RMSE and MAE metrics.
    """
    if isinstance(X_true, pd.DataFrame):
        X_true = X_true.values
    if isinstance(X_imputed, pd.DataFrame):
        X_imputed = X_imputed.values
    
    if mask is None:
        # Compare all values
        diff = X_true - X_imputed
    else:
        if isinstance(mask, pd.DataFrame):
            mask = mask.values
        
        # Only compare values that were missing
        diff = np.zeros_like(X_true)
        diff[mask.astype(bool)] = X_true[mask.astype(bool)] - X_imputed[mask.astype(bool)]
    
    # Calculate metrics
    rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(diff))
    
    return {
        'RMSE': rmse,
        'MAE': mae
    }

def create_logger(log_file=None):
    """
    Create a simple logger for experiment tracking.
    
    Parameters:
    -----------
    log_file : str, optional
        Path to log file. If None, only print to console.
        
    Returns:
    --------
    function
        A logging function that can be called with messages.
    """
    def log(message):
        print(message)
        if log_file:
            with open(log_file, 'a') as f:
                f.write(message + '\n')
    
    return log
