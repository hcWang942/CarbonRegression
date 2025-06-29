import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Set up project directories
PROJECT_ROOT = Path.cwd()
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Create DataFrame from the performance metrics data
data = {
    'Metric': ['test_r2', 'test_mse', 'rmse', 'mae', 'adj_r2'],
    'Linear Regression': [0.3685, 0.7260, 0.8521, 0.6547, 0.2854],
    'LASSO Regression': [0.3902, 0.7011, 0.8373, 0.6413, 0.2913],
    'Ridge Regression': [0.3894, 0.7020, 0.8378, 0.6424, 0.2707]
}

df = pd.DataFrame(data)

# Convert values to percentages for plotting
df_percentage = df.copy()
for col in ['Linear Regression', 'LASSO Regression', 'Ridge Regression']:
    df_percentage[col] = df_percentage[col] * 100  # Convert to percentage

# Define model names and their alpha values for the plot
model_names = [
    'Linear Regression', 
    'LASSO Regression (α=0.0498)', 
    'Ridge Regression (α=15.1991)'
]

# Define colors for each metric
metric_colors = {
    'test_r2': '#8da0cb',  # Pastel blue
    'test_mse': '#66c2a5',  # Pastel green
    'rmse': '#fc8d62',      # Pastel orange
    'mae': '#e78ac3',       # Pastel purple
    'adj_r2': '#a6d854'     # Pastel lime
}

# Create vertical version with metrics on x-axis and models as groups
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('white')  # Set figure background to white
ax.set_facecolor('white')  # Set axis background to white

# Set up positions for grouped bars
x_positions = np.arange(len(df_percentage['Metric']))
model_width = 0.25
model_offsets = [-model_width, 0, model_width]
model_colors = ['#8da0cb', '#66c2a5', '#fc8d62']  # Colors for models

# Create the vertical grouped bars
for i, model in enumerate(['Linear Regression', 'LASSO Regression', 'Ridge Regression']):
    values = df_percentage[model].values
    
    ax.bar(
        x_positions + model_offsets[i], 
        values, 
        width=model_width, 
        color=model_colors[i], 
        label=model_names[i],
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )

# Add value labels above each bar
for i, model in enumerate(['Linear Regression', 'LASSO Regression', 'Ridge Regression']):
    values = df_percentage[model].values
    for j, value in enumerate(values):
        ax.text(
            x_positions[j] + model_offsets[i], 
            value + 1.5,  # Position slightly above the bar
            f'{value:.1f}%', 
            ha='center', 
            va='bottom', 
            fontsize=9, 
            fontweight='bold'
        )

# Customize the chart
ax.set_xlabel('Metric', fontsize=14, fontweight='bold')
ax.set_ylabel('Prediction Error (%)', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(x_positions)
ax.set_xticklabels(df_percentage['Metric'], fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(fontsize=12)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save the chart
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'model_metrics_comparison_percentage_clean.png'), dpi=300, bbox_inches='tight', facecolor='white')

# Create a horizontal version with percentages
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor('white')  # Set figure background to white
ax.set_facecolor('white')  # Set axis background to white

# Set up positions for bars
bar_height = 0.15
y_positions = np.arange(len(model_names))
metric_offsets = np.linspace(-2*bar_height, 2*bar_height, len(df_percentage['Metric']))

# Create the horizontal bars for each metric
for i, metric in enumerate(df_percentage['Metric']):
    values = [df_percentage.loc[df_percentage['Metric'] == metric, model].values[0] 
              for model in ['Linear Regression', 'LASSO Regression', 'Ridge Regression']]
    
    # Create horizontal bars
    bars = ax.barh(
        y_positions + metric_offsets[i], 
        values, 
        height=bar_height, 
        color=metric_colors[metric], 
        label=metric,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add value labels to the right of each bar
    for j, bar in enumerate(bars):
        ax.text(
            bar.get_width() + 0.5, 
            bar.get_y() + bar.get_height()/2, 
            f'{values[j]:.1f}%', 
            va='center', 
            fontsize=9, 
            fontweight='bold'
        )

# Customize the chart
ax.set_xlabel('Prediction Error (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('Model', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')
ax.set_yticks(y_positions)
ax.set_yticklabels(model_names, fontsize=12)
ax.grid(axis='x', linestyle='--', alpha=0.7)
ax.legend(title='Metrics', fontsize=12, title_fontsize=14)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save the chart
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'model_metrics_comparison_horizontal_percentage_clean.png'), dpi=300, bbox_inches='tight', facecolor='white')

print(f"Charts with clean white background saved to {RESULTS_DIR}")