import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# 设置项目目录
PROJECT_ROOT = Path.cwd()
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ------------------------------
# 1. 生成系数对比图表（side_by_side_comparison.png）
# ------------------------------

# 回归系数数据
coef_data = {
    'Feature': ['PPE', 'ROE', 'TL', 'TotalWaterUse', 'SOxEmission', 'TA'],
    'Linear Regression': [0.2810, -0.1259, 0.1365, -0.0202, 0.4597, 0],  
    'LASSO Regression': [0.0620, -0.0659, 0.0660, 0.0000, 0.3786, 0.2654],
    'Ridge Regression': [0.1177, -0.0980, 0.1115, 0.0116, 0.3466, 0.1629]
}
coef_df = pd.DataFrame(coef_data)

# 创建单个组合图表（三种模型并排对比）
plt.figure(figsize=(14, 10))

# 基于所有模型的平均绝对系数排序特征，确保排序一致性
feature_avg_coef = pd.DataFrame({
    'Feature': coef_df['Feature'],
    'AvgAbsCoef': (coef_df['Linear Regression'].abs() + 
                   coef_df['LASSO Regression'].abs() + 
                   coef_df['Ridge Regression'].abs()) / 3
})
feature_order = feature_avg_coef.sort_values('AvgAbsCoef', ascending=True)['Feature'].tolist()

# 并排条形图的参数设置
bar_width = 0.25
index = np.arange(len(feature_order))

# 填充缺失值（若有）并转换为绝对值用于绘图
coef_df = coef_df.fillna(0)
abs_coefs = pd.DataFrame({
    'Feature': coef_df['Feature'],
    'Linear': coef_df['Linear Regression'].abs(),
    'LASSO': coef_df['LASSO Regression'].abs(),
    'Ridge': coef_df['Ridge Regression'].abs()
})

# 按统一的特征顺序重新排序
abs_coefs = abs_coefs.set_index('Feature').loc[feature_order].reset_index()

# 选择低饱和度的柔和颜色
colors = ['#8da0cb', '#66c2a5', '#fc8d62']  # 柔和蓝、柔和绿、柔和橙

# 绘制水平条形图（三种模型并排）
plt.barh([p - bar_width for p in index], abs_coefs['Linear'], bar_width, 
         alpha=0.8, color=colors[0], label='Linear Regression')
plt.barh([p for p in index], abs_coefs['LASSO'], bar_width, 
         alpha=0.8, color=colors[1], label='LASSO Regression')
plt.barh([p + bar_width for p in index], abs_coefs['Ridge'], bar_width, 
         alpha=0.8, color=colors[2], label='Ridge Regression')

# 为每个条形添加原始系数值（带符号）
for i, model in enumerate(['Linear', 'LASSO', 'Ridge']):
    offset = (i - 1) * bar_width  # 调整每个模型的水平偏移
    for j, (feature, value) in enumerate(zip(abs_coefs['Feature'], abs_coefs[model])):
        # 获取带符号的原始系数值
        orig_value = coef_df.loc[coef_df['Feature'] == feature, 
                           ['Linear Regression', 'LASSO Regression', 'Ridge Regression'][i]].values[0]
        plt.text(value + 0.01, index[j] + offset, f'{orig_value:.4f}', 
                 va='center', fontsize=12)

# 图表美化设置
plt.xlabel('Absolute Coefficient Value', fontsize=14)
plt.ylabel('Socioeconomic Indicators', fontsize=14)
plt.yticks(index, feature_order, fontsize=14)  # 设置y轴标签为特征名称
plt.legend(loc='lower right', fontsize=14)  # 添加图例
plt.grid(axis='x', linestyle='--', alpha=0.7)  # 仅显示x轴网格线

# 移除顶部和右侧边框
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 保存图表
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'side_by_side_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()  # 关闭当前图表，避免图形重叠


# ------------------------------
# 2. 生成模型性能指标图表（model_metrics_comparison_percentage_clean.png）
# ------------------------------

# 模型性能指标数据
metrics_data = {
    'Metric': ['test_r2', 'test_mse', 'rmse', 'mae', 'adj_r2'],
    'Linear Regression': [0.3685, 0.7260, 0.8521, 0.6547, 0.2854],
    'LASSO Regression': [0.3902, 0.7011, 0.8373, 0.6413, 0.2913],
    'Ridge Regression': [0.3894, 0.7020, 0.8378, 0.6424, 0.2707]
}
metrics_df = pd.DataFrame(metrics_data)

# 将数值转换为百分比用于绘图
df_percentage = metrics_df.copy()
for col in ['Linear Regression', 'LASSO Regression', 'Ridge Regression']:
    df_percentage[col] = df_percentage[col] * 100  # 转换为百分比

# 定义模型名称及其alpha值（用于图表标签）
model_names = [
    'Linear Regression', 
    'LASSO Regression (α=0.0498)', 
    'Ridge Regression (α=15.1991)'
]

# 创建垂直条形图（指标在x轴，模型为分组）
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('white')  # 设置图表背景为白色
ax.set_facecolor('white')  # 设置坐标轴背景为白色

# 设置分组条形图的位置
x_positions = np.arange(len(df_percentage['Metric']))
model_width = 0.25
model_offsets = [-model_width, 0, model_width]
model_colors = ['#8da0cb', '#66c2a5', '#fc8d62']  # 每个模型的颜色

# 绘制垂直分组条形图
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

# 在每个条形上方添加数值标签
for i, model in enumerate(['Linear Regression', 'LASSO Regression', 'Ridge Regression']):
    values = df_percentage[model].values
    for j, value in enumerate(values):
        ax.text(
            x_positions[j] + model_offsets[i], 
            value + 1.5,  # 位置略高于条形
            f'{value:.1f}%', 
            ha='center', 
            va='bottom', 
            fontsize=9, 
            fontweight='bold'
        )

# 图表美化
ax.set_xlabel('Metric', fontsize=14, fontweight='bold')
ax.set_ylabel('Prediction Error (%)', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(x_positions)
ax.set_xticklabels(df_percentage['Metric'], fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(fontsize=12)

# 移除顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 保存图表
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'model_metrics_comparison_percentage_clean.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()  # 关闭当前图表

print(f"图表已保存至: {RESULTS_DIR}")
print(f"生成的文件:")
print(f"- side_by_side_comparison.png (特征系数对比)")
print(f"- model_metrics_comparison_percentage_clean.png (模型性能指标对比)")