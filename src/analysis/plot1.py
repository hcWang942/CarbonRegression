import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import matplotlib.cm as cm

# Set up project directories
PROJECT_ROOT = Path.cwd()
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Data containing coefficients from different regression methods
data = {
    'Feature': ['PPE', 'ROE', 'TL', 'TotalWaterUse', 'SOxEmission', 'TA'],
    'Linear Regression': [0.2810, -0.1259, 0.1365, -0.0202, 0.4597, 0],  
    'LASSO Regression': [0.0620, -0.0659, 0.0660, 0.0000, 0.3786, 0.2654],
    'Ridge Regression': [0.1177, -0.0980, 0.1115, 0.0116, 0.3466, 0.1629]
}

# Create three separate dataframes for each regression method
df = pd.DataFrame(data)
df_linear = df[['Feature', 'Linear Regression']].copy()
df_linear.columns = ['Feature', 'Coefficient']
df_linear['Method'] = 'Linear Regression'

df_lasso = df[['Feature', 'LASSO Regression']].copy()
df_lasso.columns = ['Feature', 'Coefficient']
df_lasso['Method'] = 'LASSO Regression'

df_ridge = df[['Feature', 'Ridge Regression']].copy()
df_ridge.columns = ['Feature', 'Coefficient']
df_ridge['Method'] = 'Ridge Regression'

# Function to create a horizontal bar chart for a specific regression method
def create_bar_chart(df, filename):
    # Take absolute values for sorting and plotting
    df['AbsCoefficient'] = df['Coefficient'].abs()
    
    # Sort by absolute coefficient values in descending order
    df = df.sort_values('AbsCoefficient', ascending=True)  # Changed to True for vertical ordering
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a new color palette with lower saturation
    # Using a colormap with lower saturation
    cmap = plt.cm.Blues_r  # Blues_r gives lighter blues for higher values
    norm = plt.Normalize(0, df['AbsCoefficient'].max())
    
    # Create horizontal bar chart with improved colors
    bars = ax.barh(df['Feature'], df['AbsCoefficient'], color=cmap(norm(df['AbsCoefficient'])))
    
    # Add value labels to the right of the bars
    for i, bar in enumerate(bars):
        value = df['Coefficient'].iloc[i]
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.4f}', va='center', fontsize=14)
    
    # Set chart labels
    ax.set_xlabel('Absolute Coefficient Value', fontsize=14)
    ax.set_ylabel('Socioeconomic Indicators', fontsize=14)
    
    # Customize grid
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set x-axis limit to match the data range with a bit of padding
    max_coef = df['AbsCoefficient'].max() * 1.1
    ax.set_xlim(0, max_coef)
    
    # Add tick marks
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Create individual charts for each regression method
create_bar_chart(df_linear, 'linear_regression.png')
create_bar_chart(df_lasso, 'lasso_regression.png')
create_bar_chart(df_ridge, 'ridge_regression.png')

# Create a facet-like figure with all three charts for comparison
fig, axes = plt.subplots(3, 1, figsize=(12, 18), constrained_layout=True)
methods = ['Linear Regression', 'LASSO Regression', 'Ridge Regression']
dataframes = [df_linear, df_lasso, df_ridge]
colormaps = [plt.cm.PuBu, plt.cm.BuGn, plt.cm.YlGnBu]  # Different color schemes for each method

for i, (method, df_method, ax, cmap) in enumerate(zip(methods, dataframes, axes, colormaps)):
    # Sort by absolute coefficient values
    df_method['AbsCoefficient'] = df_method['Coefficient'].abs()
    df_method = df_method.sort_values('AbsCoefficient', ascending=True)  # Changed to True for vertical ordering
    
    # Create color normalization
    norm = plt.Normalize(0, df_method['AbsCoefficient'].max())
    
    # Create horizontal bar chart with improved colors
    bars = ax.barh(df_method['Feature'], df_method['AbsCoefficient'], 
                   color=cmap(norm(df_method['AbsCoefficient'] * 0.7 + 0.3)))  # Reduce saturation
    
    # Add value labels
    for j, bar in enumerate(bars):
        value = df_method['Coefficient'].iloc[j]
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.4f}', va='center', fontsize=14)
    
    # Only add x-label to the bottom subplot
    if i == 2:
        ax.set_xlabel('Absolute Coefficient Value', fontsize=14)
    
    ax.set_ylabel('Socioeconomic Indicators', fontsize=14)
    
    # Customize grid
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set consistent x-axis limit across all subplots
    max_coef_overall = max([df_linear['AbsCoefficient'].max(), 
                           df_lasso['AbsCoefficient'].max(), 
                           df_ridge['AbsCoefficient'].max()]) * 1.1
    ax.set_xlim(0, max_coef_overall)
    
    # Add tick marks
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Add method name as text in the plot
    ax.text(0.98, 0.95, method, transform=ax.transAxes, 
            ha='right', va='top', fontsize=14, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))

# Save the combined figure
plt.savefig(os.path.join(RESULTS_DIR, 'combined_regression_comparison.png'), dpi=300, bbox_inches='tight')

# Create a single combined chart with three models side by side
plt.figure(figsize=(14, 10))

# Consistent feature ordering across models
# Sort features based on the average absolute coefficient across all models
feature_avg_coef = pd.DataFrame({
    'Feature': df['Feature'],
    'AvgAbsCoef': (df['Linear Regression'].abs() + 
                    df['LASSO Regression'].abs() + 
                    df['Ridge Regression'].abs()) / 3
})
feature_order = feature_avg_coef.sort_values('AvgAbsCoef', ascending=True)['Feature'].tolist()

# Create single chart with side-by-side bars
bar_width = 0.25
index = np.arange(len(feature_order))

# Replace None with 0 for plotting
df = df.fillna(0)

# Convert to absolute values for plotting
abs_coefs = pd.DataFrame({
    'Feature': df['Feature'],
    'Linear': df['Linear Regression'].abs(),
    'LASSO': df['LASSO Regression'].abs(),
    'Ridge': df['Ridge Regression'].abs()
})

# Reorder based on consistent feature ordering
abs_coefs = abs_coefs.set_index('Feature').loc[feature_order].reset_index()

# Select pastel colors with lower saturation for better aesthetics
colors = ['#8da0cb', '#66c2a5', '#fc8d62']  # Pastel blue, green, orange

# Create the bars
plt.barh([p - bar_width for p in index], abs_coefs['Linear'], bar_width, 
        alpha=0.8, color=colors[0], label='Linear Regression')
plt.barh([p for p in index], abs_coefs['LASSO'], bar_width, 
        alpha=0.8, color=colors[1], label='LASSO Regression')
plt.barh([p + bar_width for p in index], abs_coefs['Ridge'], bar_width, 
        alpha=0.8, color=colors[2], label='Ridge Regression')

# Add coefficient values as text
for i, model in enumerate(['Linear', 'LASSO', 'Ridge']):
    offset = (i - 1) * bar_width
    for j, (feature, value) in enumerate(zip(abs_coefs['Feature'], abs_coefs[model])):
        # Get original coefficient with sign
        orig_value = df.loc[df['Feature'] == feature, 
                          ['Linear Regression', 'LASSO Regression', 'Ridge Regression'][i]].values[0]
        plt.text(value + 0.01, index[j] + offset, f'{orig_value:.4f}', 
                va='center', fontsize=12)

# Customize chart
plt.xlabel('Absolute Coefficient Value', fontsize=14)
plt.ylabel('Socioeconomic Indicators', fontsize=14)
plt.yticks(index, feature_order, fontsize=14)
plt.legend(loc='lower right', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Save the side-by-side comparison
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'side_by_side_comparison.png'), dpi=300, bbox_inches='tight')

print(f"All charts saved to {RESULTS_DIR}")