# Enhancing Gradient Boosting Imputation with Missingness Modeling

This repository contains an implementation of the paper "Enhancing Gradient Boosting Imputation with Missingness Modeling," which explores methods to improve gradient boosting imputation by explicitly modeling missingness patterns.

## Overview

Missing data is a critical challenge in machine learning, especially for tabular datasets. This implementation explores three strategies to enhance gradient boosting imputation:

1. **Missingness Indicator Features**: Adding binary indicators for each feature to denote missingness
2. **Missingness Pattern Clustering**: Clustering similar missingness patterns and using cluster identity as a feature
3. **Learned Embedding Approaches**: Using transformer-based models to create a dense representation of missingness patterns

## Project Structure

```
gbim_project/
├── src/
│   ├── utils.py                  # Utility functions for data processing and evaluation
│   ├── data_generator.py         # Functions to generate synthetic data with controlled missingness
│   ├── imputation_models.py      # Implementation of basic imputation models
│   ├── embedding_models.py       # Implementation of transformer-based embedding approaches
│   ├── boosting_models.py        # Implementation of different boosting algorithms
│   └── visualization.py          # Functions for creating publication-quality plots
├── data/                         # Generated synthetic datasets
├── results/                      # Experiment results
├── figures/                      # Generated plots and visualizations
├── generate_synthetic_data.py    # Script to generate synthetic datasets
├── run_experiments.py            # Script to run experiments
├── compare_boosting_algorithms.py # Script to compare different boosting algorithms
├── compare_embedding_approaches.py # Script to compare different embedding approaches
└── create_plots.py               # Script to create publication-quality plots
```

## Implemented Models

### Basic Imputation Models:
1. **StandardGBImputer**: Baseline gradient boosting imputation without special missingness modeling
2. **IndicatorGBImputer**: Gradient boosting imputation with missingness indicator features
3. **PatternClusterGBImputer**: Gradient boosting imputation with missingness pattern clustering

### Advanced Embedding Models:
1. **PCAEmbeddingGBImputer**: Using PCA to create a lower-dimensional representation of missingness patterns
2. **SimplifiedTransformerGBImputer**: A simplified implementation of transformer-based attention for missingness modeling
3. **TransformerEmbeddingGBImputer**: Full transformer-based approach with multiple variants:
   - BERT-style: Standard transformer with full self-attention
   - Linformer: Efficient transformer with linear complexity
   - Performer: Efficient transformer using random feature approximation of the softmax kernel
4. **HybridEmbeddingGBImputer**: Combining multiple embedding approaches (clustering, PCA, attention)

### Boosting Algorithms:
1. **LightGBM**: Fast gradient boosting framework with leaf-wise tree growth
2. **XGBoost**: Popular gradient boosting library with depth-wise tree growth
3. **CatBoost**: Gradient boosting optimized for categorical features (simulated)
4. **HistGradientBoostingRegressor**: Scikit-learn's implementation similar to LightGBM

## Datasets

The implementation generates synthetic datasets with different missingness mechanisms:

- **MCAR** (Missing Completely At Random) at 10%, 30%, and 50% rates
- **MAR** (Missing At Random) at 30% rate
- **MNAR** (Missing Not At Random) at 30% rate
- **Mixed** missingness at 30% rate

Each dataset includes both the data with missing values and a mask indicating which values are missing.

## Usage

### Generate Synthetic Data

```bash
python generate_synthetic_data.py
```

### Run Experiments

```bash
python run_experiments.py
```

### Compare Boosting Algorithms

```bash
python compare_boosting_algorithms.py
```

### Compare Embedding Approaches

```bash
python compare_embedding_approaches.py
```

### Create Plots

```bash
python create_plots.py
```

## Implementation Details

### Transformer-Based Embedding

The implementation includes several variants of transformer architectures for missingness modeling:

1. **BERT-style Transformer**: Standard transformer with full self-attention, providing maximum flexibility in capturing relationships between features.

2. **Linformer**: An efficient transformer variant that projects keys and values to a lower dimension, reducing complexity from O(n²) to O(n) while maintaining similar performance.

3. **Performer**: Uses random feature approximation of the softmax kernel to achieve linear complexity, ideal for higher-dimensional data.

Each transformer model processes features and their missingness status as a sequence, with a learnable [CLS] token that aggregates information. The transformer output is a dense embedding vector that captures complex dependencies in missingness patterns.

### Gradient Boosting Models

The implementation uses various gradient boosting frameworks. Each feature with missing values is treated as a regression task, with a GBM trained to predict that feature from all other features and the learned missingness embeddings.

### Missingness Modeling Strategies

1. **Indicator Features**: Binary indicators (0/1) for each feature to denote missingness
2. **Pattern Clustering**: K-means clustering on binary missingness masks to group similar patterns
3. **Transformer Embeddings**: Learned representations of missingness patterns that capture complex dependencies

### Evaluation Metrics

- **RMSE** (Root Mean Squared Error) for imputation accuracy
- **MAE** (Mean Absolute Error) for imputation accuracy
- **R²** for downstream task performance
- **Runtime** for computational efficiency

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- LightGBM
- XGBoost
- Matplotlib
- Seaborn
- PyTorch (optional, for transformer models)
- tqdm

## Paper Implementation Notes

This implementation closely follows the methodology described in the paper, with a full implementation of the transformer-based approach for missingness modeling. The main components include:

1. A comprehensive set of boosting algorithms (LightGBM, XGBoost, HistGBR)
2. Multiple transformer encoder variants (BERT-style, Linformer, Performer)
3. Three strategies for missingness modeling (indicators, clustering, embeddings)
4. Evaluation across different missingness mechanisms (MCAR, MAR, MNAR, mixed)

The implementation is designed to be modular, allowing easy comparison of different approaches and components.