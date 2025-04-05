# Enhancing Gradient Boosting Imputation with Missingness Modeling

This repository contains an implementation of the paper "Enhancing Gradient Boosting Imputation with Missingness Modeling," which explores methods to improve gradient boosting imputation by explicitly modeling missingness patterns.

## Overview

Missing data is a critical challenge in machine learning, especially for tabular datasets. This implementation explores three strategies to enhance gradient boosting imputation:

1. **Missingness Indicator Features**: Adding binary indicators for each feature to denote missingness
2. **Missingness Pattern Clustering**: Clustering similar missingness patterns and using cluster identity as a feature
3. **Simplified Embedding Approach**: Using PCA to create a lower-dimensional representation of missingness patterns (adapted from the paper's transformer-based approach)

## Project Structure

```
gbim_project/
├── src/
│   ├── utils.py                  # Utility functions for data processing and evaluation
│   ├── data_generator.py         # Functions to generate synthetic data with controlled missingness
│   ├── imputation_models.py      # Implementation of imputation models
│   └── visualization.py          # Functions for creating publication-quality plots
├── data/                         # Generated synthetic datasets
├── results/                      # Experiment results
├── figures/                      # Generated plots and visualizations
├── generate_synthetic_data.py    # Script to generate synthetic datasets
├── run_experiments.py            # Script to run experiments
└── create_plots.py               # Script to create publication-quality plots
```

## Implemented Models

1. **StandardGBImputer**: Baseline gradient boosting imputation without special missingness modeling
2. **IndicatorGBImputer**: Gradient boosting imputation with missingness indicator features
3. **PatternClusterGBImputer**: Gradient boosting imputation with missingness pattern clustering
4. **SimplifiedEmbeddingGBImputer**: Gradient boosting imputation with a simplified embedding approach using PCA

## Datasets

The implementation generates synthetic datasets with different missingness mechanisms:

- **MCAR** (Missing Completely At Random) at 10%, 30%, and 50% rates
- **MAR** (Missing At Random) at 30% rate
- **MNAR** (Missing Not At Random) at 30% rate
- **Mixed** missingness at 30% rate

## Usage

### Generate Synthetic Data

```bash
python generate_synthetic_data.py
```

### Run Experiments

```bash
python run_experiments.py
```

### Create Plots

```bash
python create_plots.py
```

## Implementation Details

### Gradient Boosting Models

The implementation uses LightGBM and XGBoost as the gradient boosting frameworks. Each feature with missing values is treated as a regression task, with a GBM trained to predict that feature from all other features.

### Missingness Modeling Strategies

1. **Indicator Features**: Binary indicators (0/1) for each feature to denote missingness
2. **Pattern Clustering**: K-means clustering on binary missingness masks to group similar patterns
3. **Simplified Embedding**: PCA applied to missingness patterns to create a dense representation

### Evaluation Metrics

- **RMSE** (Root Mean Squared Error) for imputation accuracy
- **MAE** (Mean Absolute Error) for imputation accuracy
- **R²** for downstream task performance
- **Runtime** for computational efficiency

## Adaptation Notes

This implementation adapts the paper's methodology to work with available computational resources. The main adaptation is using PCA instead of Transformer models for the embedding approach, while maintaining the core concept of creating a learned representation of missingness patterns.

## Requirements

- Python 3.10+
- NumPy
- Pandas
- Scikit-learn
- LightGBM
- XGBoost
- Matplotlib
- Seaborn
- tqdm
