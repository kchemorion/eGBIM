"""
Data generator for creating synthetic datasets with controlled missingness mechanisms.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

def generate_synthetic_data(n_samples=10000, n_features=50, random_state=42):
    """
    Generate synthetic dataset with correlated features.
    
    Parameters:
    -----------
    n_samples : int, default=10000
        Number of samples to generate.
    n_features : int, default=50
        Number of features to generate.
    random_state : int, default=42
        Random seed for reproducibility.
        
    Returns:
    --------
    pandas.DataFrame
        Synthetic dataset with correlated features.
    """
    # Generate synthetic data with correlations
    X, y = make_regression(
        n_samples=n_samples, 
        n_features=n_features,
        n_informative=int(n_features * 0.8),  # 80% of features are informative
        random_state=random_state
    )
    
    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert to DataFrame with feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add target variable for downstream tasks
    df['target'] = y
    
    return df

def introduce_mcar_missingness(df, missing_rate=0.3, random_state=42):
    """
    Introduce Missing Completely At Random (MCAR) missingness.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data without missing values.
    missing_rate : float, default=0.3
        Proportion of values to set as missing.
    random_state : int, default=42
        Random seed for reproducibility.
        
    Returns:
    --------
    pandas.DataFrame
        Data with MCAR missingness.
    tuple
        (data with missingness, mask of missing values)
    """
    np.random.seed(random_state)
    
    # Create a copy of the data
    df_missing = df.copy()
    
    # Exclude target variable from missingness
    features = df.columns[df.columns != 'target']
    
    # Create a mask for missing values
    mask = np.random.random(size=(df.shape[0], len(features))) < missing_rate
    mask_df = pd.DataFrame(mask, columns=features, index=df.index)
    
    # Set values as missing according to the mask
    for col in features:
        df_missing.loc[mask_df[col], col] = np.nan
    
    return df_missing, mask_df

def introduce_mar_missingness(df, missing_rate=0.3, n_causing_cols=3, random_state=42):
    """
    Introduce Missing At Random (MAR) missingness where missingness depends on observed values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data without missing values.
    missing_rate : float, default=0.3
        Average proportion of values to set as missing.
    n_causing_cols : int, default=3
        Number of columns that influence missingness in other columns.
    random_state : int, default=42
        Random seed for reproducibility.
        
    Returns:
    --------
    tuple
        (data with missingness, mask of missing values)
    """
    np.random.seed(random_state)
    
    # Create a copy of the data
    df_missing = df.copy()
    
    # Exclude target variable from missingness
    features = df.columns[df.columns != 'target'].tolist()
    
    # Select columns that will cause missingness in other columns
    causing_cols = np.random.choice(features, size=n_causing_cols, replace=False)
    
    # Select columns that will have missing values
    affected_cols = [col for col in features if col not in causing_cols]
    
    # Create empty mask DataFrame
    mask_df = pd.DataFrame(False, index=df.index, columns=features)
    
    # For each affected column, create missingness based on values in causing columns
    for col in affected_cols:
        # Create a probability of missingness based on values in causing columns
        # Higher values in causing columns lead to higher probability of missingness
        prob_missing = np.zeros(len(df))
        
        for causing_col in causing_cols:
            # Normalize the causing column to [0, 1] range
            normalized_values = (df[causing_col] - df[causing_col].min()) / (df[causing_col].max() - df[causing_col].min())
            # Add contribution to missingness probability
            prob_missing += normalized_values / n_causing_cols
        
        # Adjust probabilities to achieve desired missing rate on average
        prob_missing = prob_missing * missing_rate * 2  # Multiply by 2 to get some higher probabilities
        prob_missing = np.clip(prob_missing, 0, 0.9)  # Cap at 90% to avoid complete missingness
        
        # Generate missing mask based on probabilities
        col_mask = np.random.random(size=len(df)) < prob_missing
        mask_df[col] = col_mask
        
        # Set values as missing
        df_missing.loc[col_mask, col] = np.nan
    
    return df_missing, mask_df

def introduce_mnar_missingness(df, missing_rate=0.3, threshold_percentile=70, random_state=42):
    """
    Introduce Missing Not At Random (MNAR) missingness where missingness depends on the values themselves.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data without missing values.
    missing_rate : float, default=0.3
        Average proportion of values to set as missing.
    threshold_percentile : int, default=70
        Percentile threshold above which values are more likely to be missing.
    random_state : int, default=42
        Random seed for reproducibility.
        
    Returns:
    --------
    tuple
        (data with missingness, mask of missing values)
    """
    np.random.seed(random_state)
    
    # Create a copy of the data
    df_missing = df.copy()
    
    # Exclude target variable from missingness
    features = df.columns[df.columns != 'target'].tolist()
    
    # Create empty mask DataFrame
    mask_df = pd.DataFrame(False, index=df.index, columns=features)
    
    # For each feature, introduce MNAR missingness
    for col in features:
        # Calculate threshold value based on percentile
        threshold = np.percentile(df[col], threshold_percentile)
        
        # Higher values have higher probability of being missing
        high_values = df[col] > threshold
        
        # Set base probabilities
        prob_missing = np.zeros(len(df))
        prob_missing[high_values] = missing_rate * 2  # Higher probability for values above threshold
        prob_missing[~high_values] = missing_rate * 0.5  # Lower probability for values below threshold
        
        # Generate missing mask based on probabilities
        col_mask = np.random.random(size=len(df)) < prob_missing
        mask_df[col] = col_mask
        
        # Set values as missing
        df_missing.loc[col_mask, col] = np.nan
    
    return df_missing, mask_df

def create_dataset_with_mixed_missingness(n_samples=10000, n_features=50, 
                                         mcar_cols=None, mar_cols=None, mnar_cols=None,
                                         missing_rate=0.3, random_state=42):
    """
    Create a dataset with different types of missingness in different columns.
    
    Parameters:
    -----------
    n_samples : int, default=10000
        Number of samples to generate.
    n_features : int, default=50
        Number of features to generate.
    mcar_cols : list, optional
        List of column indices to have MCAR missingness. If None, one third of columns.
    mar_cols : list, optional
        List of column indices to have MAR missingness. If None, one third of columns.
    mnar_cols : list, optional
        List of column indices to have MNAR missingness. If None, one third of columns.
    missing_rate : float, default=0.3
        Proportion of values to set as missing.
    random_state : int, default=42
        Random seed for reproducibility.
        
    Returns:
    --------
    tuple
        (original data, data with missingness, mask of missing values)
    """
    np.random.seed(random_state)
    
    # Generate complete data
    df_complete = generate_synthetic_data(n_samples, n_features, random_state)
    
    # Exclude target from missingness
    features = df_complete.columns[df_complete.columns != 'target'].tolist()
    
    # If column assignments not provided, split columns evenly
    if mcar_cols is None and mar_cols is None and mnar_cols is None:
        all_cols = list(range(n_features))
        np.random.shuffle(all_cols)
        third = n_features // 3
        mcar_cols = all_cols[:third]
        mar_cols = all_cols[third:2*third]
        mnar_cols = all_cols[2*third:]
    
    # Create a copy for introducing missingness
    df_missing = df_complete.copy()
    
    # Create empty mask DataFrame
    mask_df = pd.DataFrame(False, index=df_complete.index, columns=features)
    
    # Introduce MCAR missingness
    if mcar_cols:
        mcar_features = [features[i] for i in mcar_cols]
        for col in mcar_features:
            # Generate random mask
            col_mask = np.random.random(size=len(df_complete)) < missing_rate
            mask_df[col] = col_mask
            # Set values as missing
            df_missing.loc[col_mask, col] = np.nan
    
    # Introduce MAR missingness
    if mar_cols:
        mar_features = [features[i] for i in mar_cols]
        # Select a few columns that will cause missingness
        causing_cols = np.random.choice([f for f in features if f not in mar_features], 
                                        size=min(3, len(features)-len(mar_features)), 
                                        replace=False)
        
        for col in mar_features:
            # Create probability based on causing columns
            prob_missing = np.zeros(len(df_complete))
            for causing_col in causing_cols:
                normalized_values = (df_complete[causing_col] - df_complete[causing_col].min()) / \
                                   (df_complete[causing_col].max() - df_complete[causing_col].min())
                prob_missing += normalized_values / len(causing_cols)
            
            # Adjust to get desired missing rate
            prob_missing = prob_missing * missing_rate * 2
            prob_missing = np.clip(prob_missing, 0, 0.9)
            
            # Generate missing mask
            col_mask = np.random.random(size=len(df_complete)) < prob_missing
            mask_df[col] = col_mask
            
            # Set values as missing
            df_missing.loc[col_mask, col] = np.nan
    
    # Introduce MNAR missingness
    if mnar_cols:
        mnar_features = [features[i] for i in mnar_cols]
        for col in mnar_features:
            # Calculate threshold
            threshold = np.percentile(df_complete[col], 70)
            
            # Higher values have higher probability of being missing
            high_values = df_complete[col] > threshold
            
            # Set probabilities
            prob_missing = np.zeros(len(df_complete))
            prob_missing[high_values] = missing_rate * 2
            prob_missing[~high_values] = missing_rate * 0.5
            
            # Generate missing mask
            col_mask = np.random.random(size=len(df_complete)) < prob_missing
            mask_df[col] = col_mask
            
            # Set values as missing
            df_missing.loc[col_mask, col] = np.nan
    
    return df_complete, df_missing, mask_df
