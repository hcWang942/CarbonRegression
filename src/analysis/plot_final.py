import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path.cwd()
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

coef_data = {
    'Feature': ['PPE', 'ROE', 'TL', 'TotalWaterUse', 'SOxEmission', 'TA'],
    'Linear Regression': [0.2810, -0.1259, 0.1365, -0.0202, 0.4597, 0],  
    'LASSO Regression': [0.0620, -0.0659, 0.0660, 0.0000, 0.3786, 0.2654],
    'Ridge Regression': [0.1177, -0.0980, 0.1115, 0.0116, 0.3466, 0.1629]
}
coef_df = pd.DataFrame(coef_data)

plt.figure(figsize=(14, 10))

feature_avg_coef = pd.DataFrame({
    'Feature': coef_df['Feature'],
    'AvgAbsCoef': (coef_df['Linear Regression'].abs() + 
                   coef_df['LASSO Regression'].abs() + 
                   coef_df['Ridge Regression'].abs()) / 3
})
feature_order = feature_avg_coef.sort_values('AvgAbsCoef', ascending=True)['Feature'].tolist()

bar_width = 0.25
index = np.arange(len(feature_order))

coef_df = coef_df.fillna(0)
abs_coefs = pd.DataFrame({
    'Feature': coef_df['Feature'],
    'Linear': coef_df['Linear Regression'].abs(),
    'LASSO': coef_df['LASSO Regression'].abs(),
    'Ridge': coef_df['Ridge Regression'].abs()
})

abs_coefs = abs_coefs.set_index('Feature').loc[feature_order].reset_index()

colors = ['#8da0cb', '#66c2a5', '#fc8d62']

plt.barh([p - bar_width for p in index], abs_coefs['Linear'], bar_width, 
         alpha=0.8, color=colors[0], label='Linear Regression')
plt.barh([p for p in index], abs_coefs['LASSO'], bar_width, 
         alpha=0.8, color=colors[1], label='LASSO Regression')
plt.barh([p + bar_width for p in index], abs_coefs['Ridge'], bar_width, 
         alpha=0.8, color=colors[2], label='Ridge Regression')

for i, model in enumerate(['Linear', 'LASSO', 'Ridge']):
    offset = (i - 1) * bar_width
    for j, (feature, value) in enumerate(zip(abs_coefs['Feature'], abs_coefs[model])):
        orig_value = coef_df.loc[coef_df['Feature'] == feature, 
                           ['Linear Regression', 'LASSO Regression', 'Ridge Regression'][i]].values[0]
        plt.text(value + 0.01, index[j] + offset, f'{orig_value:.4f}', 
                 va='center', fontsize=12)

plt.xlabel('Absolute Coefficient Value', fontsize=14)
plt.ylabel('Socioeconomic Indicators', fontsize=14)
plt.yticks(index, feature_order, fontsize=14)
plt.legend(loc='lower right', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'side_by_side_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

metrics_data = {
    'Metric': ['test_r2', 'test_mse', 'rmse', 'mae', 'adj_r2'],
    'Linear Regression': [0.3685, 0.7260, 0.8521, 0.6547, 0.2854],
    'LASSO Regression': [0.3902, 0.7011, 0.8373, 0.6413, 0.2913],
    'Ridge Regression': [0.3894, 0.7020, 0.8378, 0.6424, 0.2707]
}
metrics_df = pd.DataFrame(metrics_data)

df_percentage = metrics_df.copy()
for col in ['Linear Regression', 'LASSO Regression', 'Ridge Regression']:
    df_percentage[col] = df_percentage[col] * 100

model_names = [
    'Linear Regression', 
    'LASSO Regression (α=0.0498)', 
    'Ridge Regression (α=15.1991)'
]

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

x_positions = np.arange(len(df_percentage['Metric']))
model_width = 0.25
model_offsets = [-model_width, 0, model_width]
model_colors = ['#8da0cb', '#66c2a5', '#fc8d62']

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

for i, model in enumerate(['Linear Regression', 'LASSO Regression', 'Ridge Regression']):
    values = df_percentage[model].values
    for j, value in enumerate(values):
        ax.text(
            x_positions[j] + model_offsets[i], 
            value + 1.5,
            f'{value:.1f}%', 
            ha='center', 
            va='bottom', 
            fontsize=9, 
            fontweight='bold'
        )

ax.set_xlabel('Metric', fontsize=14, fontweight='bold')
ax.set_ylabel('Prediction Error (%)', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(x_positions)
ax.set_xticklabels(df_percentage['Metric'], fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'model_metrics_comparison_percentage_clean.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Results ha s: {RESULTS_DIR}")
print(f"生成的文件:")
print(f"- side_by_side_comparison.png (特征系数对比)")
print(f"- model_metrics_comparison_percentage_clean.png (模型性能指标对比)")