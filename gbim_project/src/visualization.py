"""
Visualization utilities for the Gradient Boosting Imputation with Missingness Modeling project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

def set_plotting_style():
    """Set publication-quality plotting style."""
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

def plot_imputation_error_comparison(results_df, metric='Test_RMSE', 
                                     missingness_types=None, 
                                     save_path=None):
    """
    Plot imputation error comparison across different models and missingness types.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results dataframe with experiment results.
    metric : str, default='Test_RMSE'
        Metric to plot ('Test_RMSE', 'Test_MAE', etc.).
    missingness_types : list, optional
        List of missingness types to include. If None, use all.
    save_path : str, optional
        Path to save the figure. If None, display the figure.
    """
    set_plotting_style()
    
    # Filter missingness types if specified
    if missingness_types:
        df = results_df[results_df['Experiment'].isin(missingness_types)].copy()
    else:
        df = results_df.copy()
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create grouped bar plot
    ax = sns.barplot(x='Description', y=metric, hue='Model', data=df)
    
    # Customize plot
    plt.title(f'Imputation Error Comparison ({metric.replace("_", " ")})')
    plt.xlabel('Missingness Type')
    plt.ylabel(metric.replace('_', ' '))
    plt.xticks(rotation=45)
    plt.legend(title='Imputation Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format y-axis to show 4 decimal places
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', fontsize=10)
    
    plt.tight_layout()
    
    # Save or display figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

def plot_downstream_performance(results_df, metric='Downstream_R2', 
                               missingness_types=None, 
                               save_path=None):
    """
    Plot downstream task performance comparison.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results dataframe with experiment results.
    metric : str, default='Downstream_R2'
        Metric to plot ('Downstream_R2', 'Downstream_RMSE', etc.).
    missingness_types : list, optional
        List of missingness types to include. If None, use all.
    save_path : str, optional
        Path to save the figure. If None, display the figure.
    """
    set_plotting_style()
    
    # Filter missingness types if specified
    if missingness_types:
        df = results_df[results_df['Experiment'].isin(missingness_types)].copy()
    else:
        df = results_df.copy()
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create grouped bar plot
    ax = sns.barplot(x='Description', y=metric, hue='Model', data=df)
    
    # Customize plot
    plt.title(f'Downstream Task Performance ({metric.replace("_", " ")})')
    plt.xlabel('Missingness Type')
    plt.ylabel(metric.replace('_', ' '))
    plt.xticks(rotation=45)
    plt.legend(title='Imputation Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format y-axis to show 4 decimal places
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', fontsize=10)
    
    plt.tight_layout()
    
    # Save or display figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

def plot_runtime_comparison(results_df, missingness_types=None, save_path=None):
    """
    Plot runtime comparison across different models and missingness types.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results dataframe with experiment results.
    missingness_types : list, optional
        List of missingness types to include. If None, use all.
    save_path : str, optional
        Path to save the figure. If None, display the figure.
    """
    set_plotting_style()
    
    # Filter missingness types if specified
    if missingness_types:
        df = results_df[results_df['Experiment'].isin(missingness_types)].copy()
    else:
        df = results_df.copy()
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create grouped bar plot
    ax = sns.barplot(x='Description', y='Runtime', hue='Model', data=df)
    
    # Customize plot
    plt.title('Runtime Comparison')
    plt.xlabel('Missingness Type')
    plt.ylabel('Runtime (seconds)')
    plt.xticks(rotation=45)
    plt.legend(title='Imputation Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=10)
    
    plt.tight_layout()
    
    # Save or display figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

def plot_missingness_rate_comparison(results_df, models=None, metric='Test_RMSE', save_path=None):
    """
    Plot imputation error comparison across different missingness rates (MCAR).
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results dataframe with experiment results.
    models : list, optional
        List of models to include. If None, use all.
    metric : str, default='Test_RMSE'
        Metric to plot ('Test_RMSE', 'Test_MAE', etc.).
    save_path : str, optional
        Path to save the figure. If None, display the figure.
    """
    set_plotting_style()
    
    # Filter for MCAR experiments with different rates
    mcar_exps = ['mcar_10', 'mcar_30', 'mcar_50']
    df = results_df[results_df['Experiment'].isin(mcar_exps)].copy()
    
    # Extract missingness rate from description
    df['Missingness_Rate'] = df['Description'].str.extract(r'MCAR (\d+)%').astype(int)
    
    # Filter models if specified
    if models:
        df = df[df['Model'].isin(models)]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create line plot
    ax = sns.lineplot(x='Missingness_Rate', y=metric, hue='Model', 
                     style='Model', markers=True, dashes=False, data=df)
    
    # Customize plot
    plt.title(f'Effect of Missingness Rate on {metric.replace("_", " ")}')
    plt.xlabel('Missingness Rate (%)')
    plt.ylabel(metric.replace('_', ' '))
    plt.legend(title='Imputation Model')
    
    # Format y-axis to show 4 decimal places
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    
    plt.tight_layout()
    
    # Save or display figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

def plot_missingness_mechanism_comparison(results_df, models=None, metric='Test_RMSE', save_path=None):
    """
    Plot imputation error comparison across different missingness mechanisms (MCAR, MAR, MNAR).
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results dataframe with experiment results.
    models : list, optional
        List of models to include. If None, use all.
    metric : str, default='Test_RMSE'
        Metric to plot ('Test_RMSE', 'Test_MAE', etc.).
    save_path : str, optional
        Path to save the figure. If None, display the figure.
    """
    set_plotting_style()
    
    # Filter for 30% missingness experiments with different mechanisms
    exps = ['mcar_30', 'mar_30', 'mnar_30']
    df = results_df[results_df['Experiment'].isin(exps)].copy()
    
    # Extract missingness mechanism from description
    df['Mechanism'] = df['Description'].str.extract(r'(MCAR|MAR|MNAR)')
    
    # Filter models if specified
    if models:
        df = df[df['Model'].isin(models)]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create grouped bar plot
    ax = sns.barplot(x='Mechanism', y=metric, hue='Model', data=df)
    
    # Customize plot
    plt.title(f'Effect of Missingness Mechanism on {metric.replace("_", " ")}')
    plt.xlabel('Missingness Mechanism')
    plt.ylabel(metric.replace('_', ' '))
    plt.legend(title='Imputation Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format y-axis to show 4 decimal places
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', fontsize=10)
    
    plt.tight_layout()
    
    # Save or display figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

def create_summary_table(results_df, metrics=None):
    """
    Create a summary table of results.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results dataframe with experiment results.
    metrics : list, optional
        List of metrics to include. If None, use default metrics.
        
    Returns:
    --------
    pandas.DataFrame
        Summary table with mean metrics across experiments.
    """
    if metrics is None:
        metrics = ['Test_RMSE', 'Test_MAE', 'Downstream_R2', 'Runtime']
    
    # Group by model and calculate mean metrics
    summary = results_df.groupby('Model')[metrics].mean().reset_index()
    
    # Round metrics to 4 decimal places
    for metric in metrics:
        if metric == 'Runtime':
            summary[metric] = summary[metric].round(1)
        else:
            summary[metric] = summary[metric].round(4)
    
    return summary
