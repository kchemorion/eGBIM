"""
Implementation of transformer-based embedding approaches for missingness modeling.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class BaseEmbeddingGBImputer(BaseEstimator, TransformerMixin):
    """
    Base class for Gradient Boosting Imputation with embedding-based missingness modeling.
    """
    
    def __init__(self, boosting_model='lightgbm', max_iter=5, tol=1e-3, random_state=42,
                 model_params=None, verbose=True):
        """
        Initialize the imputer.
        
        Parameters:
        -----------
        boosting_model : str, default='lightgbm'
            Type of boosting model to use ('lightgbm', 'xgboost', 'catboost', or 'histgbr').
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
            elif self.boosting_model == 'catboost' or self.boosting_model == 'histgbr':
                self.model_params = {
                    'learning_rate': 0.05,
                    'max_iter': 500,
                    'max_depth': 6,
                    'max_bins': 255,
                    'random_state': self.random_state,
                    'verbose': 0
                }
    
    def _get_model(self):
        """Get the appropriate boosting model."""
        if self.boosting_model == 'lightgbm':
            return lgb.LGBMRegressor(**self.model_params)
        elif self.boosting_model == 'xgboost':
            return xgb.XGBRegressor(**self.model_params)
        elif self.boosting_model == 'catboost' or self.boosting_model == 'histgbr':
            return HistGradientBoostingRegressor(**self.model_params)
        else:
            raise ValueError(f"Unsupported boosting model: {self.boosting_model}")
    
    def _create_embeddings(self, X):
        """
        Create embeddings of missingness patterns.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
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
        
        # Get embeddings for current data
        if not hasattr(self, 'embeddings_') or X.shape[0] != self.embeddings_.shape[0]:
            # For new data in transform, compute embeddings
            embeddings = self._create_embeddings(X)
        else:
            # Use pre-computed embeddings from fit_transform
            embeddings = self.embeddings_
        
        # Create embedding feature DataFrame
        embedding_df = pd.DataFrame(
            embeddings,
            columns=[f'embedding_{i}' for i in range(embeddings.shape[1])],
            index=X.index
        )
        
        # Combine original features with embedding features
        X_train = pd.concat([X[predictor_cols], embedding_df], axis=1)
        y_train = X[feature_to_impute]
        
        return X_train, y_train
    
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
        
        # Create embeddings
        self.embeddings_ = self._create_embeddings(X_imputed)
        
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
                
                if hasattr(model, 'feature_importances_'):
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
            
            # Update embeddings after each iteration
            self.embeddings_ = self._create_embeddings(X_imputed)
        
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
        
        # Create embeddings for the new data
        embeddings = self._create_embeddings(X_imputed)
        self.embeddings_ = embeddings
        
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


class PCAEmbeddingGBImputer(BaseEmbeddingGBImputer):
    """
    Gradient Boosting Imputation with PCA-based embedding of missingness patterns.
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
            Additional parameters for BaseEmbeddingGBImputer.
        """
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.include_values = include_values
    
    def _create_embeddings(self, X):
        """
        Create PCA-based embeddings of missingness patterns.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data.
            
        Returns:
        --------
        numpy.ndarray
            Embeddings of missingness patterns.
        """
        # Create binary missingness mask
        mask_matrix = X.isna().astype(int).values
        
        if self.include_values:
            # Standardize values
            X_std = (X - X.mean()) / X.std()
            
            # Replace NaN with 0 after standardization (in case of constant columns)
            X_std = X_std.fillna(0)
            
            # Combine mask with standardized values
            combined_matrix = np.hstack([mask_matrix, X_std.values])
        else:
            # Use only the mask
            combined_matrix = mask_matrix
        
        # Apply PCA to create embeddings
        if not hasattr(self, 'pca_'):
            self.pca_ = PCA(n_components=self.embedding_dim, random_state=self.random_state)
            embeddings = self.pca_.fit_transform(combined_matrix)
        else:
            embeddings = self.pca_.transform(combined_matrix)
        
        return embeddings


if TORCH_AVAILABLE:
    class TabularDataset(Dataset):
        """
        Torch dataset for tabular data with missing values.
        """
        def __init__(self, X, mask, feature_means):
            """
            Initialize the dataset.
            
            Parameters:
            -----------
            X : pandas.DataFrame
                Input data with missing values filled.
            mask : pandas.DataFrame
                Binary mask of missing values (1 if missing, 0 if present).
            feature_means : pandas.Series
                Mean values for each feature, used for initial imputation.
            """
            self.X = torch.tensor(X.values, dtype=torch.float32)
            self.mask = torch.tensor(mask.values, dtype=torch.float32)
            self.feature_means = torch.tensor(feature_means.values, dtype=torch.float32)
            
        def __len__(self):
            return len(self.X)
            
        def __getitem__(self, idx):
            return {
                'features': self.X[idx],
                'mask': self.mask[idx],
                'feature_means': self.feature_means
            }


    class SelfAttention(nn.Module):
        """
        Self-attention module for the transformer encoder.
        """
        def __init__(self, embed_dim, num_heads, dropout=0.1):
            """
            Initialize the self-attention module.
            
            Parameters:
            -----------
            embed_dim : int
                Dimension of the embedding vectors.
            num_heads : int
                Number of attention heads.
            dropout : float, default=0.1
                Dropout rate.
            """
            super().__init__()
            self.multihead_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            
        def forward(self, x):
            return self.multihead_attn(x, x, x)[0]


    class TransformerEncoderLayer(nn.Module):
        """
        Transformer encoder layer with self-attention and feed-forward network.
        """
        def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
            """
            Initialize the transformer encoder layer.
            
            Parameters:
            -----------
            embed_dim : int
                Dimension of the embedding vectors.
            num_heads : int
                Number of attention heads.
            ff_dim : int
                Dimension of the feed-forward network.
            dropout : float, default=0.1
                Dropout rate.
            """
            super().__init__()
            self.self_attn = SelfAttention(embed_dim, num_heads, dropout)
            self.feed_forward = nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, embed_dim)
            )
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            # Self-attention with residual connection and layer normalization
            attn_output = self.self_attn(x)
            x = self.norm1(x + self.dropout(attn_output))
            
            # Feed-forward with residual connection and layer normalization
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))
            
            return x


    class TransformerEncoder(nn.Module):
        """
        Transformer encoder for tabular data with missing values.
        """
        def __init__(self, n_features, embed_dim=32, num_heads=4, ff_dim=64, 
                    num_layers=2, dropout=0.1, pool_type='cls'):
            """
            Initialize the transformer encoder.
            
            Parameters:
            -----------
            n_features : int
                Number of features in the input data.
            embed_dim : int, default=32
                Dimension of the embedding vectors.
            num_heads : int, default=4
                Number of attention heads.
            ff_dim : int, default=64
                Dimension of the feed-forward network.
            num_layers : int, default=2
                Number of transformer encoder layers.
            dropout : float, default=0.1
                Dropout rate.
            pool_type : str, default='cls'
                Type of pooling to use for the encoder output ('cls', 'mean', or 'sum').
            """
            super().__init__()
            self.n_features = n_features
            self.embed_dim = embed_dim
            self.pool_type = pool_type
            
            # Create feature and position embeddings
            self.feature_embedding = nn.Embedding(n_features, embed_dim)
            self.value_projection = nn.Linear(1, embed_dim)
            self.mask_embedding = nn.Embedding(2, embed_dim)  # 0=present, 1=missing
            
            # Add a learnable [CLS] token embedding
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            
            # Create the transformer encoder layers
            self.layers = nn.ModuleList([
                TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ])
            
            # Output projection for the embedding
            self.output_projection = nn.Linear(embed_dim, embed_dim)
            
        def forward(self, batch):
            """
            Forward pass through the transformer encoder.
            
            Parameters:
            -----------
            batch : dict
                Batch of data containing 'features', 'mask', and 'feature_means'.
                
            Returns:
            --------
            torch.Tensor
                Embeddings for each sample in the batch.
            """
            features = batch['features']  # Shape: [batch_size, n_features]
            mask = batch['mask']  # Shape: [batch_size, n_features]
            
            batch_size = features.shape[0]
            
            # Create position IDs (feature indices)
            position_ids = torch.arange(self.n_features, device=features.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, n_features]
            
            # Create feature embeddings
            pos_embeddings = self.feature_embedding(position_ids)  # Shape: [batch_size, n_features, embed_dim]
            
            # Create value embeddings
            values = features.unsqueeze(-1)  # Shape: [batch_size, n_features, 1]
            value_embeddings = self.value_projection(values)  # Shape: [batch_size, n_features, embed_dim]
            
            # Create mask embeddings
            mask_embeddings = self.mask_embedding(mask.long())  # Shape: [batch_size, n_features, embed_dim]
            
            # Combine embeddings
            combined_embeddings = pos_embeddings + value_embeddings + mask_embeddings
            
            # Add [CLS] token at the beginning
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: [batch_size, 1, embed_dim]
            combined_embeddings = torch.cat([cls_tokens, combined_embeddings], dim=1)  # Shape: [batch_size, n_features+1, embed_dim]
            
            # Pass through transformer encoder layers
            for layer in self.layers:
                combined_embeddings = layer(combined_embeddings)
            
            # Pool the output based on pool_type
            if self.pool_type == 'cls':
                # Use the [CLS] token representation
                pooled_output = combined_embeddings[:, 0]  # Shape: [batch_size, embed_dim]
            elif self.pool_type == 'mean':
                # Use the mean of all token representations
                pooled_output = combined_embeddings.mean(dim=1)  # Shape: [batch_size, embed_dim]
            elif self.pool_type == 'sum':
                # Use the sum of all token representations
                pooled_output = combined_embeddings.sum(dim=1)  # Shape: [batch_size, embed_dim]
            else:
                raise ValueError(f"Unsupported pool_type: {self.pool_type}")
            
            # Project the output
            embeddings = self.output_projection(pooled_output)  # Shape: [batch_size, embed_dim]
            
            return embeddings


    class LinformerAttention(nn.Module):
        """
        Linformer attention module with linear complexity.
        """
        def __init__(self, embed_dim, num_heads, seq_len, k=256, dropout=0.1):
            """
            Initialize the Linformer attention module.
            
            Parameters:
            -----------
            embed_dim : int
                Dimension of the embedding vectors.
            num_heads : int
                Number of attention heads.
            seq_len : int
                Length of the input sequence.
            k : int, default=256
                Dimension to project the keys and values to.
            dropout : float, default=0.1
                Dropout rate.
            """
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            self.seq_len = seq_len
            self.k = min(k, seq_len)  # k can't be larger than sequence length
            
            # Projections for query, key, value
            self.query = nn.Linear(embed_dim, embed_dim)
            self.key = nn.Linear(embed_dim, embed_dim)
            self.value = nn.Linear(embed_dim, embed_dim)
            
            # Linformer projections for keys and values
            self.E = nn.Parameter(torch.randn(self.k, seq_len) / seq_len)
            self.F = nn.Parameter(torch.randn(self.k, seq_len) / seq_len)
            
            # Output projection
            self.out_proj = nn.Linear(embed_dim, embed_dim)
            
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            """
            Forward pass through the Linformer attention module.
            
            Parameters:
            -----------
            x : torch.Tensor
                Input tensor of shape [batch_size, seq_len, embed_dim].
                
            Returns:
            --------
            torch.Tensor
                Output tensor of shape [batch_size, seq_len, embed_dim].
            """
            batch_size, seq_len, embed_dim = x.shape
            
            # Compute query, key, value projections
            q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            q = q.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
            
            k = self.key(x)  # [batch_size, seq_len, embed_dim]
            v = self.value(x)  # [batch_size, seq_len, embed_dim]
            
            # Apply Linformer projections to keys and values
            k_proj = torch.matmul(self.E.to(x.device), k)  # [batch_size, k, embed_dim]
            v_proj = torch.matmul(self.F.to(x.device), v)  # [batch_size, k, embed_dim]
            
            # Reshape for multi-head attention
            k_proj = k_proj.view(batch_size, self.k, self.num_heads, self.head_dim)
            k_proj = k_proj.permute(0, 2, 1, 3)  # [batch_size, num_heads, k, head_dim]
            
            v_proj = v_proj.view(batch_size, self.k, self.num_heads, self.head_dim)
            v_proj = v_proj.permute(0, 2, 1, 3)  # [batch_size, num_heads, k, head_dim]
            
            # Compute attention scores
            scores = torch.matmul(q, k_proj.transpose(-1, -2)) / (self.head_dim ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            context = torch.matmul(attn_weights, v_proj)  # [batch_size, num_heads, seq_len, head_dim]
            context = context.permute(0, 2, 1, 3)  # [batch_size, seq_len, num_heads, head_dim]
            context = context.reshape(batch_size, seq_len, embed_dim)
            
            # Apply output projection
            output = self.out_proj(context)
            
            return output


    class LinformerEncoderLayer(nn.Module):
        """
        Linformer encoder layer with linear complexity.
        """
        def __init__(self, embed_dim, num_heads, ff_dim, seq_len, k=256, dropout=0.1):
            """
            Initialize the Linformer encoder layer.
            
            Parameters:
            -----------
            embed_dim : int
                Dimension of the embedding vectors.
            num_heads : int
                Number of attention heads.
            ff_dim : int
                Dimension of the feed-forward network.
            seq_len : int
                Length of the input sequence.
            k : int, default=256
                Dimension to project the keys and values to.
            dropout : float, default=0.1
                Dropout rate.
            """
            super().__init__()
            self.self_attn = LinformerAttention(embed_dim, num_heads, seq_len, k, dropout)
            self.feed_forward = nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, embed_dim)
            )
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            """
            Forward pass through the Linformer encoder layer.
            
            Parameters:
            -----------
            x : torch.Tensor
                Input tensor of shape [batch_size, seq_len, embed_dim].
                
            Returns:
            --------
            torch.Tensor
                Output tensor of shape [batch_size, seq_len, embed_dim].
            """
            # Self-attention with residual connection and layer normalization
            attn_output = self.self_attn(x)
            x = self.norm1(x + self.dropout(attn_output))
            
            # Feed-forward with residual connection and layer normalization
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))
            
            return x


    class LinformerEncoder(nn.Module):
        """
        Linformer encoder for tabular data with missing values.
        """
        def __init__(self, n_features, embed_dim=32, num_heads=4, ff_dim=64, 
                    num_layers=2, k=256, dropout=0.1, pool_type='cls'):
            """
            Initialize the Linformer encoder.
            
            Parameters:
            -----------
            n_features : int
                Number of features in the input data.
            embed_dim : int, default=32
                Dimension of the embedding vectors.
            num_heads : int, default=4
                Number of attention heads.
            ff_dim : int, default=64
                Dimension of the feed-forward network.
            num_layers : int, default=2
                Number of transformer encoder layers.
            k : int, default=256
                Dimension to project the keys and values to.
            dropout : float, default=0.1
                Dropout rate.
            pool_type : str, default='cls'
                Type of pooling to use for the encoder output ('cls', 'mean', or 'sum').
            """
            super().__init__()
            self.n_features = n_features
            self.embed_dim = embed_dim
            self.pool_type = pool_type
            
            # Create feature and position embeddings
            self.feature_embedding = nn.Embedding(n_features, embed_dim)
            self.value_projection = nn.Linear(1, embed_dim)
            self.mask_embedding = nn.Embedding(2, embed_dim)  # 0=present, 1=missing
            
            # Add a learnable [CLS] token embedding
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            
            # Create the Linformer encoder layers
            seq_len = n_features + 1  # +1 for the [CLS] token
            self.layers = nn.ModuleList([
                LinformerEncoderLayer(embed_dim, num_heads, ff_dim, seq_len, k, dropout)
                for _ in range(num_layers)
            ])
            
            # Output projection for the embedding
            self.output_projection = nn.Linear(embed_dim, embed_dim)
            
        def forward(self, batch):
            """
            Forward pass through the Linformer encoder.
            
            Parameters:
            -----------
            batch : dict
                Batch of data containing 'features', 'mask', and 'feature_means'.
                
            Returns:
            --------
            torch.Tensor
                Embeddings for each sample in the batch.
            """
            features = batch['features']  # Shape: [batch_size, n_features]
            mask = batch['mask']  # Shape: [batch_size, n_features]
            
            batch_size = features.shape[0]
            
            # Create position IDs (feature indices)
            position_ids = torch.arange(self.n_features, device=features.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, n_features]
            
            # Create feature embeddings
            pos_embeddings = self.feature_embedding(position_ids)  # Shape: [batch_size, n_features, embed_dim]
            
            # Create value embeddings
            values = features.unsqueeze(-1)  # Shape: [batch_size, n_features, 1]
            value_embeddings = self.value_projection(values)  # Shape: [batch_size, n_features, embed_dim]
            
            # Create mask embeddings
            mask_embeddings = self.mask_embedding(mask.long())  # Shape: [batch_size, n_features, embed_dim]
            
            # Combine embeddings
            combined_embeddings = pos_embeddings + value_embeddings + mask_embeddings
            
            # Add [CLS] token at the beginning
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: [batch_size, 1, embed_dim]
            combined_embeddings = torch.cat([cls_tokens, combined_embeddings], dim=1)  # Shape: [batch_size, n_features+1, embed_dim]
            
            # Pass through Linformer encoder layers
            for layer in self.layers:
                combined_embeddings = layer(combined_embeddings)
            
            # Pool the output based on pool_type
            if self.pool_type == 'cls':
                # Use the [CLS] token representation
                pooled_output = combined_embeddings[:, 0]  # Shape: [batch_size, embed_dim]
            elif self.pool_type == 'mean':
                # Use the mean of all token representations
                pooled_output = combined_embeddings.mean(dim=1)  # Shape: [batch_size, embed_dim]
            elif self.pool_type == 'sum':
                # Use the sum of all token representations
                pooled_output = combined_embeddings.sum(dim=1)  # Shape: [batch_size, embed_dim]
            else:
                raise ValueError(f"Unsupported pool_type: {self.pool_type}")
            
            # Project the output
            embeddings = self.output_projection(pooled_output)  # Shape: [batch_size, embed_dim]
            
            return embeddings


    class PerformerAttention(nn.Module):
        """
        Performer attention module with random feature approximation of the softmax kernel.
        """
        def __init__(self, embed_dim, num_heads, num_features=256, dropout=0.1):
            """
            Initialize the Performer attention module.
            
            Parameters:
            -----------
            embed_dim : int
                Dimension of the embedding vectors.
            num_heads : int
                Number of attention heads.
            num_features : int, default=256
                Number of random features for approximation.
            dropout : float, default=0.1
                Dropout rate.
            """
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            self.num_features = num_features
            
            # Projections for query, key, value
            self.query = nn.Linear(embed_dim, embed_dim)
            self.key = nn.Linear(embed_dim, embed_dim)
            self.value = nn.Linear(embed_dim, embed_dim)
            
            # Random feature maps for keys and queries
            self.feature_map_k = nn.Parameter(torch.randn(num_heads, self.head_dim, num_features) / (self.head_dim ** 0.5))
            self.feature_map_q = nn.Parameter(torch.randn(num_heads, self.head_dim, num_features) / (self.head_dim ** 0.5))
            
            # Output projection
            self.out_proj = nn.Linear(embed_dim, embed_dim)
            
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            """
            Forward pass through the Performer attention module.
            
            Parameters:
            -----------
            x : torch.Tensor
                Input tensor of shape [batch_size, seq_len, embed_dim].
                
            Returns:
            --------
            torch.Tensor
                Output tensor of shape [batch_size, seq_len, embed_dim].
            """
            batch_size, seq_len, embed_dim = x.shape
            
            # Compute query, key, value projections
            q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            q = q.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
            
            k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
            
            v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
            
            # Apply random feature maps
            q_prime = torch.exp(q / (self.head_dim ** 0.5))  # Apply softmax normalization
            q_mapped = torch.matmul(q_prime, self.feature_map_q)  # [batch_size, num_heads, seq_len, num_features]
            
            k_prime = torch.exp(k / (self.head_dim ** 0.5))  # Apply softmax normalization
            k_mapped = torch.matmul(k_prime, self.feature_map_k)  # [batch_size, num_heads, seq_len, num_features]
            
            # Approximate attention
            kv = k_mapped.transpose(2, 3) @ v  # [batch_size, num_heads, num_features, head_dim]
            qkv = q_mapped @ kv  # [batch_size, num_heads, seq_len, head_dim]
            
            # Normalize
            normalizer = 1.0 / (torch.sum(q_mapped, dim=2, keepdim=True) @ torch.sum(k_mapped, dim=2, keepdim=True).transpose(2, 3))
            attn_output = qkv * normalizer
            
            # Reshape and project
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
            output = self.out_proj(attn_output)
            
            return output


    class PerformerEncoderLayer(nn.Module):
        """
        Performer encoder layer with efficient attention.
        """
        def __init__(self, embed_dim, num_heads, ff_dim, num_features=256, dropout=0.1):
            """
            Initialize the Performer encoder layer.
            
            Parameters:
            -----------
            embed_dim : int
                Dimension of the embedding vectors.
            num_heads : int
                Number of attention heads.
            ff_dim : int
                Dimension of the feed-forward network.
            num_features : int, default=256
                Number of random features for approximation.
            dropout : float, default=0.1
                Dropout rate.
            """
            super().__init__()
            self.self_attn = PerformerAttention(embed_dim, num_heads, num_features, dropout)
            self.feed_forward = nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, embed_dim)
            )
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            """
            Forward pass through the Performer encoder layer.
            
            Parameters:
            -----------
            x : torch.Tensor
                Input tensor of shape [batch_size, seq_len, embed_dim].
                
            Returns:
            --------
            torch.Tensor
                Output tensor of shape [batch_size, seq_len, embed_dim].
            """
            # Self-attention with residual connection and layer normalization
            attn_output = self.self_attn(x)
            x = self.norm1(x + self.dropout(attn_output))
            
            # Feed-forward with residual connection and layer normalization
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))
            
            return x


    class PerformerEncoder(nn.Module):
        """
        Performer encoder for tabular data with missing values.
        """
        def __init__(self, n_features, embed_dim=32, num_heads=4, ff_dim=64, 
                    num_layers=2, num_features=256, dropout=0.1, pool_type='cls'):
            """
            Initialize the Performer encoder.
            
            Parameters:
            -----------
            n_features : int
                Number of features in the input data.
            embed_dim : int, default=32
                Dimension of the embedding vectors.
            num_heads : int, default=4
                Number of attention heads.
            ff_dim : int, default=64
                Dimension of the feed-forward network.
            num_layers : int, default=2
                Number of transformer encoder layers.
            num_features : int, default=256
                Number of random features for approximation.
            dropout : float, default=0.1
                Dropout rate.
            pool_type : str, default='cls'
                Type of pooling to use for the encoder output ('cls', 'mean', or 'sum').
            """
            super().__init__()
            self.n_features = n_features
            self.embed_dim = embed_dim
            self.pool_type = pool_type
            
            # Create feature and position embeddings
            self.feature_embedding = nn.Embedding(n_features, embed_dim)
            self.value_projection = nn.Linear(1, embed_dim)
            self.mask_embedding = nn.Embedding(2, embed_dim)  # 0=present, 1=missing
            
            # Add a learnable [CLS] token embedding
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            
            # Create the Performer encoder layers
            self.layers = nn.ModuleList([
                PerformerEncoderLayer(embed_dim, num_heads, ff_dim, num_features, dropout)
                for _ in range(num_layers)
            ])
            
            # Output projection for the embedding
            self.output_projection = nn.Linear(embed_dim, embed_dim)
            
        def forward(self, batch):
            """
            Forward pass through the Performer encoder.
            
            Parameters:
            -----------
            batch : dict
                Batch of data containing 'features', 'mask', and 'feature_means'.
                
            Returns:
            --------
            torch.Tensor
                Embeddings for each sample in the batch.
            """
            features = batch['features']  # Shape: [batch_size, n_features]
            mask = batch['mask']  # Shape: [batch_size, n_features]
            
            batch_size = features.shape[0]
            
            # Create position IDs (feature indices)
            position_ids = torch.arange(self.n_features, device=features.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, n_features]
            
            # Create feature embeddings
            pos_embeddings = self.feature_embedding(position_ids)  # Shape: [batch_size, n_features, embed_dim]
            
            # Create value embeddings
            values = features.unsqueeze(-1)  # Shape: [batch_size, n_features, 1]
            value_embeddings = self.value_projection(values)  # Shape: [batch_size, n_features, embed_dim]
            
            # Create mask embeddings
            mask_embeddings = self.mask_embedding(mask.long())  # Shape: [batch_size, n_features, embed_dim]
            
            # Combine embeddings
            combined_embeddings = pos_embeddings + value_embeddings + mask_embeddings
            
            # Add [CLS] token at the beginning
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: [batch_size, 1, embed_dim]
            combined_embeddings = torch.cat([cls_tokens, combined_embeddings], dim=1)  # Shape: [batch_size, n_features+1, embed_dim]
            
            # Pass through Performer encoder layers
            for layer in self.layers:
                combined_embeddings = layer(combined_embeddings)
            
            # Pool the output based on pool_type
            if self.pool_type == 'cls':
                # Use the [CLS] token representation
                pooled_output = combined_embeddings[:, 0]  # Shape: [batch_size, embed_dim]
            elif self.pool_type == 'mean':
                # Use the mean of all token representations
                pooled_output = combined_embeddings.mean(dim=1)  # Shape: [batch_size, embed_dim]
            elif self.pool_type == 'sum':
                # Use the sum of all token representations
                pooled_output = combined_embeddings.sum(dim=1)  # Shape: [batch_size, embed_dim]
            else:
                raise ValueError(f"Unsupported pool_type: {self.pool_type}")
            
            # Project the output
            embeddings = self.output_projection(pooled_output)  # Shape: [batch_size, embed_dim]
            
            return embeddings


    class TransformerEmbeddingGBImputer(BaseEmbeddingGBImputer):
        """
        Gradient Boosting Imputation with transformer-based embedding of missingness patterns.
        """
        
        def __init__(self, embedding_dim=32, transformer_type='bert', num_heads=4, ff_dim=64, 
                    num_layers=2, batch_size=64, num_epochs=10, learning_rate=1e-3, pool_type='cls',
                    include_values=True, device=None, **kwargs):
            """
            Initialize the imputer.
            
            Parameters:
            -----------
            embedding_dim : int, default=32
                Dimension of the embedding vector.
            transformer_type : str, default='bert'
                Type of transformer encoder to use ('bert', 'performer', 'linformer', or 'reformer').
            num_heads : int, default=4
                Number of attention heads.
            ff_dim : int, default=64
                Dimension of the feed-forward network.
            num_layers : int, default=2
                Number of transformer encoder layers.
            batch_size : int, default=64
                Batch size for training the transformer.
            num_epochs : int, default=10
                Number of epochs for training the transformer.
            learning_rate : float, default=1e-3
                Learning rate for training the transformer.
            pool_type : str, default='cls'
                Type of pooling to use for the encoder output ('cls', 'mean', or 'sum').
            include_values : bool, default=True
                Whether to include observed values in the embedding computation.
            device : str, optional
                Device to use for training ('cuda' or 'cpu'). If None, use CUDA if available.
            **kwargs : dict
                Additional parameters for BaseEmbeddingGBImputer.
            """
            super().__init__(**kwargs)
            self.embedding_dim = embedding_dim
            self.transformer_type = transformer_type
            self.num_heads = num_heads
            self.ff_dim = ff_dim
            self.num_layers = num_layers
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.learning_rate = learning_rate
            self.pool_type = pool_type
            self.include_values = include_values
            
            # Set device
            if device is None:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = device
            
        def _create_transformer_encoder(self, n_features):
            """
            Create the appropriate transformer encoder based on the specified type.
            
            Parameters:
            -----------
            n_features : int
                Number of features in the input data.
                
            Returns:
            --------
            torch.nn.Module
                Transformer encoder model.
            """
            if self.transformer_type == 'bert':
                return TransformerEncoder(
                    n_features=n_features,
                    embed_dim=self.embedding_dim,
                    num_heads=self.num_heads,
                    ff_dim=self.ff_dim,
                    num_layers=self.num_layers,
                    dropout=0.1,
                    pool_type=self.pool_type
                )
            elif self.transformer_type == 'linformer':
                return LinformerEncoder(
                    n_features=n_features,
                    embed_dim=self.embedding_dim,
                    num_heads=self.num_heads,
                    ff_dim=self.ff_dim,
                    num_layers=self.num_layers,
                    k=min(256, n_features + 1),  # k can't be larger than sequence length
                    dropout=0.1,
                    pool_type=self.pool_type
                )
            elif self.transformer_type == 'performer':
                return PerformerEncoder(
                    n_features=n_features,
                    embed_dim=self.embedding_dim,
                    num_heads=self.num_heads,
                    ff_dim=self.ff_dim,
                    num_layers=self.num_layers,
                    num_features=min(256, n_features * 2),  # Number of random features
                    dropout=0.1,
                    pool_type=self.pool_type
                )
            else:
                raise ValueError(f"Unsupported transformer_type: {self.transformer_type}")
            
        def _train_transformer(self, X, mask):
            """
            Train the transformer encoder.
            
            Parameters:
            -----------
            X : pandas.DataFrame
                Input data with missing values filled.
            mask : pandas.DataFrame
                Binary mask of missing values (1 if missing, 0 if present).
                
            Returns:
            --------
            torch.nn.Module
                Trained transformer encoder model.
            """
            # Create dataset
            dataset = TabularDataset(X, mask, X.mean())
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # Create model
            n_features = X.shape[1]
            model = self._create_transformer_encoder(n_features)
            model.to(self.device)
            
            # Create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            
            # Train model
            model.train()
            for epoch in range(self.num_epochs):
                total_loss = 0
                for batch in dataloader:
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    embeddings = model(batch)
                    
                    # Compute reconstruction loss (predict mask from embeddings)
                    # Create a linear projection layer for mask prediction
                    if not hasattr(model, 'mask_predictor'):
                        model.mask_predictor = nn.Linear(model.embed_dim, batch['mask'].size(1)).to(self.device)
                    
                    # Predict mask
                    pred_mask = model.mask_predictor(embeddings)
                    
                    # Compute loss: binary cross-entropy between predicted and actual mask
                    # (we're training the model to predict which values are missing)
                    mask_loss = F.binary_cross_entropy_with_logits(
                        pred_mask, batch['mask'], reduction='mean'
                    )
                    
                    # Backward pass
                    optimizer.zero_grad()
                    mask_loss.backward()
                    optimizer.step()
                    
                    total_loss += mask_loss.item()
                
                if self.verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss/len(dataloader):.6f}")
            
            return model
            
        def _create_embeddings(self, X):
            """
            Create transformer-based embeddings of missingness patterns.
            
            Parameters:
            -----------
            X : pandas.DataFrame
                Input data.
                
            Returns:
            --------
            numpy.ndarray
                Embeddings of missingness patterns.
            """
            # Create binary missingness mask
            mask = X.isna().astype(int)
            
            # Train transformer if not already trained
            if not hasattr(self, 'transformer_'):
                self.transformer_ = self._train_transformer(X, mask)
            
            # Create dataset
            dataset = TabularDataset(X, mask, X.mean())
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            
            # Generate embeddings
            self.transformer_.eval()
            embeddings = []
            
            with torch.no_grad():
                for batch in dataloader:
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Generate embeddings
                    batch_embeddings = self.transformer_(batch)
                    embeddings.append(batch_embeddings.cpu().numpy())
            
            # Concatenate embeddings
            embeddings = np.vstack(embeddings)
            
            return embeddings
else:
    # Simplified versions for when PyTorch is not available
    class TransformerEmbeddingGBImputer(PCAEmbeddingGBImputer):
        """
        Fallback to PCA-based embeddings when PyTorch is not available.
        """
        def __init__(self, embedding_dim=32, transformer_type='bert', **kwargs):
            super().__init__(embedding_dim=embedding_dim, **kwargs)
            self.transformer_type = transformer_type
            print("WARNING: PyTorch not available. Using PCA-based embeddings instead of transformer-based embeddings.")


class SimplifiedTransformerGBImputer(BaseEmbeddingGBImputer):
    """
    Gradient Boosting Imputation with a simplified transformer-based embedding.
    
    This is a simplified version that captures some aspects of self-attention
    without requiring PyTorch.
    """
    
    def __init__(self, embedding_dim=16, n_components=3, **kwargs):
        """
        Initialize the imputer.
        
        Parameters:
        -----------
        embedding_dim : int, default=16
            Dimension of the embedding vector.
        n_components : int, default=3
            Number of components to use for the simplified attention mechanism.
        **kwargs : dict
            Additional parameters for BaseEmbeddingGBImputer.
        """
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.n_components = n_components
    
    def _create_embeddings(self, X):
        """
        Create simplified transformer-based embeddings of missingness patterns.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data.
            
        Returns:
        --------
        numpy.ndarray
            Embeddings of missingness patterns.
        """
        # Create binary missingness mask
        mask_matrix = X.isna().astype(int).values
        
        # Compute "attention" using correlation matrix (simplified self-attention)
        # This captures relationships between features' missingness patterns
        corr_matrix = np.corrcoef(mask_matrix.T)
        
        # Replace NaN with 0 (in case of constant columns)
        corr_matrix = np.nan_to_num(corr_matrix, 0)
        
        # Apply PCA to the correlation matrix to get "attention components"
        if not hasattr(self, 'attention_pca_'):
            self.attention_pca_ = PCA(n_components=self.n_components, random_state=self.random_state)
            attention_components = self.attention_pca_.fit_transform(corr_matrix)
        else:
            attention_components = self.attention_pca_.transform(corr_matrix)
        
        # Apply the "attention" to the mask matrix
        # This is a simplified version of the attention mechanism in transformers
        attended_mask = mask_matrix @ attention_components
        
        # Combine original mask with attended mask
        combined_matrix = np.hstack([mask_matrix, attended_mask])
        
        # Apply PCA to create final embeddings
        if not hasattr(self, 'pca_'):
            self.pca_ = PCA(n_components=self.embedding_dim, random_state=self.random_state)
            embeddings = self.pca_.fit_transform(combined_matrix)
        else:
            embeddings = self.pca_.transform(combined_matrix)
        
        return embeddings


class HybridEmbeddingGBImputer(BaseEmbeddingGBImputer):
    """
    Gradient Boosting Imputation with a hybrid embedding approach that combines
    multiple embedding techniques.
    """
    
    def __init__(self, embedding_dim=16, use_clustering=True, use_pca=True, 
                 use_attention=True, **kwargs):
        """
        Initialize the imputer.
        
        Parameters:
        -----------
        embedding_dim : int, default=16
            Dimension of the embedding vector.
        use_clustering : bool, default=True
            Whether to include clustering-based embeddings.
        use_pca : bool, default=True
            Whether to include PCA-based embeddings.
        use_attention : bool, default=True
            Whether to include attention-based embeddings.
        **kwargs : dict
            Additional parameters for BaseEmbeddingGBImputer.
        """
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.use_clustering = use_clustering
        self.use_pca = use_pca
        self.use_attention = use_attention
        
        # Determine dimensions for each component
        n_components = sum([use_clustering, use_pca, use_attention])
        self.component_dim = embedding_dim // n_components if n_components > 0 else embedding_dim
    
    def _create_embeddings(self, X):
        """
        Create hybrid embeddings of missingness patterns.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data.
            
        Returns:
        --------
        numpy.ndarray
            Embeddings of missingness patterns.
        """
        # Create binary missingness mask
        mask_matrix = X.isna().astype(int).values
        
        embeddings_list = []
        
        # 1. Clustering-based embeddings
        if self.use_clustering:
            if not hasattr(self, 'kmeans_'):
                self.kmeans_ = KMeans(n_clusters=self.component_dim, 
                                     random_state=self.random_state,
                                     n_init=10)
                self.kmeans_.fit(mask_matrix)
            
            # Get distances to cluster centers as embeddings
            cluster_distances = self.kmeans_.transform(mask_matrix)
            embeddings_list.append(cluster_distances)
        
        # 2. PCA-based embeddings
        if self.use_pca:
            if not hasattr(self, 'pca_'):
                self.pca_ = PCA(n_components=self.component_dim, 
                               random_state=self.random_state)
                pca_embeddings = self.pca_.fit_transform(mask_matrix)
            else:
                pca_embeddings = self.pca_.transform(mask_matrix)
            
            embeddings_list.append(pca_embeddings)
        
        # 3. Attention-based embeddings (simplified)
        if self.use_attention:
            # Compute correlation matrix as a simple form of "attention"
            corr_matrix = np.corrcoef(mask_matrix.T)
            corr_matrix = np.nan_to_num(corr_matrix, 0)
            
            # Apply PCA to the correlation matrix
            if not hasattr(self, 'attention_pca_'):
                self.attention_pca_ = PCA(n_components=self.component_dim, 
                                         random_state=self.random_state)
                attention_components = self.attention_pca_.fit_transform(corr_matrix)
            else:
                attention_components = self.attention_pca_.transform(corr_matrix)
            
            # Apply attention components to the mask
            attended_mask = mask_matrix @ attention_components
            
            embeddings_list.append(attended_mask)
        
        # Combine all embeddings
        if embeddings_list:
            combined_embeddings = np.hstack(embeddings_list)
        else:
            # Fallback to PCA if no embedding method is selected
            if not hasattr(self, 'fallback_pca_'):
                self.fallback_pca_ = PCA(n_components=self.embedding_dim, 
                                        random_state=self.random_state)
                combined_embeddings = self.fallback_pca_.fit_transform(mask_matrix)
            else:
                combined_embeddings = self.fallback_pca_.transform(mask_matrix)
        
        return combined_embeddings