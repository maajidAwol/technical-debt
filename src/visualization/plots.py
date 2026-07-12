"""
Visualization utilities for Technical Debt Prediction Research
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import RESULTS_DIR, FIGURES_DIR


def setup_plotting_style():
    """Set up consistent plotting style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_class_distribution(y: pd.Series, save_path: Path = None):
    """Plot class distribution (High-Risk vs Low-Risk)."""
    setup_plotting_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart
    counts = y.value_counts()
    labels = ['Low-Risk (0)', 'High-Risk (1)']
    colors = ['#2ecc71', '#e74c3c']
    
    axes[0].bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=colors)
    axes[0].set_ylabel('Number of Files')
    axes[0].set_title('Class Distribution')
    
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    # Pie chart
    axes[1].pie([counts.get(0, 0), counts.get(1, 0)], labels=labels, 
                colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Class Proportion')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = FIGURES_DIR / "class_distribution.png"
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_severity_distribution(severity_counts: pd.DataFrame, save_path: Path = None):
    """Plot distribution of issue severities."""
    setup_plotting_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        'BLOCKER': '#e74c3c',
        'CRITICAL': '#e67e22',
        'MAJOR': '#f1c40f',
        'MINOR': '#3498db',
        'INFO': '#95a5a6'
    }
    
    bar_colors = [colors.get(sev, '#95a5a6') for sev in severity_counts['SEVERITY']]
    
    bars = ax.bar(severity_counts['SEVERITY'], severity_counts['count'], color=bar_colors)
    ax.set_xlabel('Severity Level')
    ax.set_ylabel('Number of Issues')
    ax.set_title('Distribution of SonarQube Issue Severities')
    
    # Add percentage labels
    for bar, pct in zip(bars, severity_counts['percentage']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{pct}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = FIGURES_DIR / "severity_distribution.png"
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_model_comparison(results_df: pd.DataFrame, save_path: Path = None):
    """Plot model comparison bar chart."""
    setup_plotting_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    x = np.arange(len(results_df))
    width = 0.2
    
    # Extract mean values
    for i, metric in enumerate(metrics):
        if metric == 'AUC-ROC':
            col = 'AUC_mean'
        else:
            col = f"{metric.replace('-', '_').replace('Score', '')}_mean".lower()
            col = col.replace('__', '_')
            if col == 'f1__mean':
                col = 'F1_mean'
        
        if col in results_df.columns:
            values = results_df[col]
        elif f"{metric}_mean".replace('-', '_') in results_df.columns:
            values = results_df[f"{metric}_mean".replace('-', '_')]
        else:
            continue
            
        ax.bar(x + i*width, values, width, label=metric)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = FIGURES_DIR / "model_comparison.png"
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15, 
                           save_path: Path = None):
    """Plot feature importance bar chart."""
    setup_plotting_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = importance_df.head(top_n)
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_features)))
    
    bars = ax.barh(range(len(top_features)), top_features['importance'], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Most Important Features for High-Risk TD Prediction')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = FIGURES_DIR / "feature_importance.png"
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_cross_project_results(results_df: pd.DataFrame, save_path: Path = None):
    """Plot cross-project validation results."""
    setup_plotting_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # F1 scores by project
    ax1 = axes[0, 0]
    projects = results_df['test_project'].str[:20]  # Truncate names
    ax1.barh(projects, results_df['f1_score'], color='steelblue')
    ax1.set_xlabel('F1-Score')
    ax1.set_title('F1-Score by Test Project')
    ax1.axvline(results_df['f1_score'].mean(), color='red', linestyle='--', 
                label=f'Mean: {results_df["f1_score"].mean():.3f}')
    ax1.legend()
    
    # Metrics distribution
    ax2 = axes[0, 1]
    metrics_to_plot = ['precision', 'recall', 'f1_score', 'auc_roc']
    box_data = [results_df[m] for m in metrics_to_plot]
    bp = ax2.boxplot(box_data, labels=['Precision', 'Recall', 'F1', 'AUC'])
    ax2.set_ylabel('Score')
    ax2.set_title('Metrics Distribution Across Projects')
    
    # Scatter: samples vs F1
    ax3 = axes[1, 0]
    ax3.scatter(results_df['test_samples'], results_df['f1_score'], 
                c=results_df['test_high_risk_pct'], cmap='RdYlGn_r', 
                s=100, alpha=0.7)
    ax3.set_xlabel('Number of Test Samples')
    ax3.set_ylabel('F1-Score')
    ax3.set_title('F1-Score vs Test Set Size')
    cbar = plt.colorbar(ax3.collections[0], ax=ax3)
    cbar.set_label('High-Risk %')
    
    # High-risk percentage vs F1
    ax4 = axes[1, 1]
    ax4.scatter(results_df['test_high_risk_pct'], results_df['f1_score'], 
                s=100, alpha=0.7, color='coral')
    ax4.set_xlabel('High-Risk Percentage in Test Set')
    ax4.set_ylabel('F1-Score')
    ax4.set_title('F1-Score vs High-Risk Ratio')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = FIGURES_DIR / "cross_project_results.png"
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    print("Visualization utilities loaded.")
    print("Available functions:")
    print("  - plot_class_distribution(y)")
    print("  - plot_severity_distribution(severity_df)")
    print("  - plot_model_comparison(results_df)")
    print("  - plot_feature_importance(importance_df)")
    print("  - plot_cross_project_results(results_df)")
