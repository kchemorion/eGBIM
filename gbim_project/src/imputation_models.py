"""
Implementation of Gradient Boosting Imputation methods with different missingness modeling strategies.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm

class BaseGBImputer(BaseEstimator, TransformerMixin):
    """
    Base class for Gradient Boosting Imputation.
    """
    
    def __init__(self, boosting_model='lightgbm', max_iter=5, tol=1e-3, random_state=42,
                 model_params=None, verbose=True):
        """
        Initialize the imputer.
        
        Parameters:
        -----------
        boosting_model : str, default='lightgbm'
            Type of boosting model to use ('lightgbm' or 'xgboost').
        max_iter : int, default=5
            Maximum number of imputation iterations.
        tol : float, default=1e-3
            Tolerance for convergence.
        random_state : int, default=42
            Random seed for reproducibility.
        model_params : dict, optional
            Parameters for the boosting model.
        verbose : bool, default=True
            Whether to print progress information.
        """
        self.boosting_model = boosting_model
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.model_params = model_params
        self.verbose = verbose
        self.models_ = {}
        self.feature_importances_ = {}
        
        # Set default model parameters if not provided
        if self.model_params is None:
            if self.boosting_model == 'lightgbm':
                self.model_params = {
                    'objective': 'regression',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'n_estimators': 500,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': self.random_state,
                    'verbose': -1
                }
            elif self.boosting_model == 'xgboost':
                self.model_params = {
                    'objective': 'reg:squarederror',
                    'learning_rate': 0.05,
                    'n_estimators': 500,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': self.random_state,
                    'verbosity': 0
                }
    
    def _get_model(self):
        """Get the appropriate boosting model."""
        if self.boosting_model == 'lightgbm':
            return lgb.LGBMRegressor(**self.model_params)
        elif self.boosting_model == 'xgboost':
            return xgb.XGBRegressor(**self.model_params)
        else:
            raise ValueError(f"Unsupported boosting model: {self.boosting_model}")
    
    def _prepare_data_for_imputation(self, X, feature_to_impute):
        """
        Prepare data for imputation of a specific feature.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def fit_transform(self, X, y=None):
        """
        Fit the imputer and transform the data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with missing values.
        y : None
            Ignored.
            
        Returns:
        --------
        pandas.DataFrame
            Imputed data.
        """
        # Initialize with simple imputation (mean)
        X_imputed = X.copy()
        
        # Get initial mean values for each column
        self.initial_values_ = X.mean()
        
        # Fill missing values with column means
        X_imputed = X_imputed.fillna(self.initial_values_)
        
        # Store column names and missing mask
        self.columns_ = X.columns
        self.missing_mask_ = X.isna()
        
        # Identify columns with missing values
        self.missing_columns_ = [col for col in X.columns if X[col].isna().any()]
        
        # Iterative imputation
        prev_X = X_imputed.copy()
        
        for iteration in range(self.max_iter):
            if self.verbose:
                print(f"Imputation iteration {iteration+1}/{self.max_iter}")
            
            # For each column with missing values, train a model and impute
            for col in tqdm(self.missing_columns_, disable=not self.verbose):
                # Get indices of missing values in this column
                missing_idx = self.missing_mask_[col]
                
                if missing_idx.sum() == 0:
                    continue
                
                # Prepare data for this column's imputation
                X_train, y_train = self._prepare_data_for_imputation(X_imputed, col)
                
                # Train model
                model = self._get_model()
                model.fit(X_train[~missing_idx], y_train[~missing_idx])
                
                # Store model and feature importances
                self.models_[col] = model
                
                if self.boosting_model == 'lightgbm':
                    self.feature_importances_[col] = model.feature_importances_
                elif self.boosting_model == 'xgboost':
                    self.feature_importances_[col] = model.feature_importances_
                
                # Predict missing values
                X_imputed.loc[missing_idx, col] = model.predict(X_train[missing_idx])
            
            # Check convergence
            diff = np.mean((X_imputed - prev_X)**2)
            if self.verbose:
                print(f"Mean squared difference: {diff:.6f}")
            
            if diff < self.tol:
                if self.verbose:
                    print(f"Converged after {iteration+1} iterations")
                break
            
            prev_X = X_imputed.copy()
        
        return X_imputed
    
    def transform(self, X):
        """
        Transform new data using the fitted imputer.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with missing values.
            
        Returns:
        --------
        pandas.DataFrame
            Imputed data.
        """
        if not hasattr(self, 'models_'):
            raise ValueError("This instance is not fitted yet. Call 'fit_transform' first.")
        
        # Initialize with simple imputation (mean)
        X_imputed = X.copy()
        
        # Fill missing values with column means from training data
        for col in X.columns:
            if col in self.initial_values_:
                X_imputed[col] = X_imputed[col].fillna(self.initial_values_[col])
        
        # Get missing mask
        missing_mask = X.isna()
        
        # For each column with missing values, impute using the trained model
        for col in self.missing_columns_:
            if col not in X.columns or not missing_mask[col].any():
                continue
            
            # Get indices of missing values in this column
            missing_idx = missing_mask[col]
            
            # Prepare data for this column's imputation
            X_pred, _ = self._prepare_data_for_imputation(X_imputed, col)
            
            # Predict missing values
            X_imputed.loc[missing_idx, col] = self.models_[col].predict(X_pred[missing_idx])
        
        return X_imputed


class StandardGBImputer(BaseGBImputer):
    """
    Standard Gradient Boosting Imputation without special missingness modeling.
    """
    
    def _prepare_data_for_imputation(self, X, feature_to_impute):
        """
        Prepare data for imputation of a specific feature.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data.
        feature_to_impute : str
            Name of the feature to impute.
            
        Returns:
        --------
        tuple
            (X_train, y_train) for model training.
        """
        # Use all other features as predictors
        predictor_cols = [col for col in X.columns if col != feature_to_impute]
        
        X_train = X[predictor_cols]
        y_train = X[feature_to_impute]
        
        return X_train, y_train


class IndicatorGBImputer(BaseGBImputer):
    """
    Gradient Boosting Imputation with missingness indicator features.
    """
    
    def _prepare_data_for_imputation(self, X, feature_to_impute):
        """
        Prepare data for imputation of a specific feature, including missingness indicators.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data.
        feature_to_impute : str
            Name of the feature to impute.
            
        Returns:
        --------
        tuple
            (X_train, y_train) for model training.
        """
        # Use all other features as predictors
        predictor_cols = [col for col in X.columns if col != feature_to_impute]
        
        # Create missingness indicators for all features except the target
        indicators = pd.DataFrame(index=X.index)
        for col in predictor_cols:
            indicators[f"{col}_missing"] = self.missing_mask_[col].astype(int)
        
        # Combine original features with indicators
        X_train = pd.concat([X[predictor_cols], indicators], axis=1)
        y_train = X[feature_to_impute]
        
        return X_train, y_train


class PatternClusterGBImputer(BaseGBImputer):
    """
    Gradient Boosting Imputation with missingness pattern clustering.
    """
    
    def __init__(self, n_clusters=10, **kwargs):
        """
        Initialize the imputer.
        
        Parameters:
        -----------
        n_clusters : int, default=10
            Number of clusters for missingness patterns.
        **kwargs : dict
            Additional parameters for BaseGBImputer.
        """
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        
    def fit_transform(self, X, y=None):
        """
        Fit the imputer and transform the data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with missing values.
        y : None
            Ignored.
            
        Returns:
        --------
        pandas.DataFrame
            Imputed data.
        """
        # Store column names and missing mask
        self.columns_ = X.columns
        self.missing_mask_ = X.isna()
        
        # Cluster missingness patterns
        self._fit_pattern_clusters(X)
        
        # Continue with standard imputation process
        return super().fit_transform(X, y)
    
    def _fit_pattern_clusters(self, X):
        """
        Fit KMeans clustering on missingness patterns.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with missing values.
        """
        # Create binary missingness mask
        mask_matrix = X.isna().astype(int).values
        
        # Apply KMeans clustering
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.cluster_labels_ = self.kmeans_.fit_predict(mask_matrix)
        
        # Create one-hot encoder for cluster labels
        self.cluster_encoder_ = OneHotEncoder(sparse_output=False)
        self.cluster_onehot_ = self.cluster_encoder_.fit_transform(self.cluster_labels_.reshape(-1, 1))
    
    def _prepare_data_for_imputation(self, X, feature_to_impute):
        """
        Prepare data for imputation of a specific feature, including pattern clusters.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data.
        feature_to_impute : str
            Name of the feature to impute.
            
        Returns:
        --------
        tuple
            (X_train, y_train) for model training.
        """
        # Use all other features as predictors
        predictor_cols = [col for col in X.columns if col != feature_to_impute]
        
        # Get cluster assignments for current data
        if not hasattr(self, 'cluster_labels_'):
            # For new data in transform, predict clusters
            mask_matrix = X.isna().astype(int).values
            cluster_labels = self.kmeans_.predict(mask_matrix)
            cluster_onehot = self.cluster_encoder_.transform(cluster_labels.reshape(-1, 1))
        else:
            # Use pre-computed clusters from fit_transform
            cluster_onehot = self.cluster_onehot_
        
        # Create cluster feature DataFrame
        cluster_df = pd.DataFrame(
            cluster_onehot,
            columns=[f'pattern_cluster_{i}' for i in range(self.n_clusters)],
            index=X.index
        )
        
        # Combine original features with cluster features
        X_train = pd.concat([X[predictor_cols], cluster_df], axis=1)
        y_train = X[feature_to_impute]
        
        return X_train, y_train
    
    def transform(self, X):
        """
        Transform new data using the fitted imputer.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with missing values.
            
        Returns:
        --------
        pandas.DataFrame
            Imputed data.
        """
        # For new data, we need to assign clusters based on missingness patterns
        mask_matrix = X.isna().astype(int).values
        self.cluster_labels_ = self.kmeans_.predict(mask_matrix)
        self.cluster_onehot_ = self.cluster_encoder_.transform(self.cluster_labels_.reshape(-1, 1))
        
        # Continue with standard transform process
        return super().transform(X)


class SimplifiedEmbeddingGBImputer(BaseGBImputer):
    """
    Gradient Boosting Imputation with a simplified embedding approach using PCA.
    
    Since we can't use Transformer models due to space constraints, this class
    uses PCA to create a lower-dimensional representation of the missingness patterns.
    """
    
    def __init__(self, embedding_dim=16, include_values=True, **kwargs):
        """
        Initialize the imputer.
        
        Parameters:
        -----------
        embedding_dim : int, default=16
            Dimension of the embedding vector.
        include_values : bool, default=True
            Whether to include observed values in the embedding computation.
        **kwargs : dict
            Additional parameters for BaseGBImputer.
        """
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.include_values = include_values
    
    def fit_transform(self, X, y=None):
        """
        Fit the imputer and transform the data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with missing values.
        y : None
            Ignored.
            
        Returns:
        --------
        pandas.DataFrame
            Imputed data.
        """
        # Store column names and missing mask
        self.columns_ = X.columns
        self.missing_mask_ = X.isna()
        
        # Create embeddings
        self._fit_embeddings(X)
        
        # Continue with standard imputation process
        return super().fit_transform(X, y)
    
    def _fit_embeddings(self, X):
        """
        Fit PCA to create embeddings of missingness patterns.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with missing values.
        """
        # Create binary missingness mask
        mask_matrix = X.isna().astype(int).values
        
        if self.include_values:
            # Fill missing values with mean for initial representation
            X_filled = X.fillna(X.mean())
            
            # Standardize values
            X_std = (X_filled - X_filled.mean()) / X_filled.std()
            
            # Replace NaN with 0 after standardization (in case of constant columns)
            X_std = X_std.fillna(0)
            
            # Combine mask with standardized values
            # For missing values, we'll use the mask (1) and 0 for the value
            # For present values, we'll use the mask (0) and the standardized value
            combined_matrix = np.hstack([mask_matrix, X_std.values])
        else:
            # Use only the mask
            combined_matrix = mask_matrix
        
        # Apply PCA to create embeddings
        self.pca_ = PCA(n_components=self.embedding_dim, random_state=self.random_state)
        self.embeddings_ = self.pca_.fit_transform(combined_matrix)
    
    def _prepare_data_for_imputation(self, X, feature_to_impute):
        """
        Prepare data for imputation of a specific feature, including embeddings.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data.
        feature_to_impute : str
            Name of the feature to impute.
            
        Returns:
        --------
        tuple
            (X_train, y_train) for model training.
        """
        # Use all other features as predictors
        predictor_cols = [col for col in X.columns if col != feature_to_impute]
        
        # Get embeddings for current data
        if not hasattr(self, 'embeddings_') or X.shape[0] != self.embeddings_.shape[0]:
            # For new data in transform, compute embeddings
            mask_matrix = X.isna().astype(int).values
            
            if self.include_values:
                # Fill missing values with mean from training
                X_filled = X.copy()
                for col in X.columns:
                    if col in self.initial_values_:
                        X_filled[col] = X_filled[col].fillna(self.initial_values_[col])
                
                # Standardize using training means and stds
                X_std = X_filled.copy()
                for col in X_std.columns:
                    if col in self.columns_:
                        X_std[col] = (X_std[col] - X_filled[col].mean()) / X_filled[col].std()
                
                # Replace NaN with 0 after standardization
                X_std = X_std.fillna(0)
                
                # Combine mask with standardized values
                combined_matrix = np.hstack([mask_matrix, X_std.values])
            else:
                # Use only the mask
                combined_matrix = mask_matrix
            
            # Transform using pre-trained PCA
            embeddings = self.pca_.transform(combined_matrix)
        else:
            # Use pre-computed embeddings from fit_transform
            embeddings = self.embeddings_
        
        # Create embedding feature DataFrame
        embedding_df = pd.DataFrame(
            embeddings,
            columns=[f'embedding_{i}' for i in range(self.embedding_dim)],
            index=X.index
        )
        
        # Combine original features with embedding features
        X_train = pd.concat([X[predictor_cols], embedding_df], axis=1)
        y_train = X[feature_to_impute]
        
        return X_train, y_train
    
    def transform(self, X):
        """
        Transform new data using the fitted imputer.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with missing values.
            
        Returns:
        --------
        pandas.DataFrame
            Imputed data.
        """
        # For new data, we need to compute embeddings
        # This will be done in _prepare_data_for_imputation
        # Reset embeddings to force recomputation
        if hasattr(self, 'embeddings_'):
            del self.embeddings_
        
        # Continue with standard transform process
        return super().transform(X)
