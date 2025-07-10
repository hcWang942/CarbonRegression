import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis, shapiro, jarque_bera
import warnings
import os
from pathlib import Path
from data_transform_functions import transform_data
warnings.filterwarnings('ignore')

# 统一设置结果目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'src', 'preprocessing', 'preprocessing_results')


# 相关性分析类
class CorrelationAnalyzer:
    def __init__(self, data, threshold=0.3):
        self.data = data
        self.threshold = threshold
        self.corr_matrix = None
        self.significant_vars = None
        
    def compute_correlations(self):
        self.corr_matrix = self.data.corr()
        scope1_corr = self.corr_matrix.iloc[0]
        self.significant_vars = scope1_corr[abs(scope1_corr) > self.threshold].index.tolist()
        return self.significant_vars
    
    def compute_correlations_all(self):
        self.corr_matrix = self.data.corr()
        scope1_corr = self.corr_matrix.iloc[0]
        all_vars = scope1_corr.index.tolist()
        return all_vars
    
    def plot_heatmap(self, save_path=None, use_all_vars=False):
        plt.figure(figsize=(15, 12) if use_all_vars else (12, 10))
        
        if use_all_vars:
            plot_corr = self.corr_matrix
            title_suffix = "All Variables"
        else:
            plot_corr = self.corr_matrix.loc[self.significant_vars, self.significant_vars]
            title_suffix = f"Variables with |correlation| > {self.threshold}"
        
        sns.heatmap(plot_corr, 
                   annot=True, 
                   cmap="YlGnBu", 
                   vmin=-1, 
                   vmax=1, 
                   center=0,
                   fmt='.2f',
                   annot_kws={'size': 8 if use_all_vars else 10})
        
        plt.title(f'Correlation Heatmap ({title_suffix})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    def save_results(self, output_dir, include_all=False):
        os.makedirs(output_dir, exist_ok=True)
        
        scope1_corr = self.corr_matrix.iloc[0][self.significant_vars]
        results_df = pd.DataFrame({
            'Variable': scope1_corr.index,
            'Correlation_with_Scope1': scope1_corr.values
        }).sort_values('Correlation_with_Scope1', key=abs, ascending=False)
        
        with open(os.path.join(output_dir, 'correlation_analysis.txt'), 'w') as f:
            f.write("Correlation Analysis Results\n")
            f.write("===========================\n\n")
            f.write(f"Correlation threshold: {self.threshold}\n")
            f.write(f"Number of significant variables: {len(self.significant_vars)-1}\n\n")
            f.write("Significant correlations with Scope 1:\n")
            f.write(results_df.to_string())
        
        if include_all:
            all_scope1_corr = self.corr_matrix.iloc[0]
            all_results_df = pd.DataFrame({
                'Variable': all_scope1_corr.index,
                'Correlation_with_Scope1': all_scope1_corr.values
            }).sort_values('Correlation_with_Scope1', key=abs, ascending=False)
            
            with open(os.path.join(output_dir, 'correlation_analysis_all.txt'), 'w') as f:
                f.write("Complete Correlation Analysis Results (All Variables)\n")
                f.write("====================================================\n\n")
                f.write(f"Total number of variables: {len(all_scope1_corr)-1}\n")
                f.write(f"Variables with |correlation| > {self.threshold}: {len(self.significant_vars)-1}\n")
                f.write(f"Variables with |correlation| <= {self.threshold}: {len(all_scope1_corr) - len(self.significant_vars)}\n\n")
                f.write("All correlations with Scope 1 (sorted by absolute value):\n")
                f.write(all_results_df.to_string())
            
            corr_values = all_scope1_corr.drop('Scope1').abs()
            summary_stats = {
                'Total_Variables': len(corr_values),
                'Mean_Abs_Correlation': corr_values.mean(),
                'Median_Abs_Correlation': corr_values.median(),
                'Max_Abs_Correlation': corr_values.max(),
                'Min_Abs_Correlation': corr_values.min(),
                'Std_Abs_Correlation': corr_values.std(),
                f'Variables_Above_{self.threshold}': sum(corr_values > self.threshold),
                f'Percentage_Above_{self.threshold}': (sum(corr_values > self.threshold) / len(corr_values)) * 100
            }
            
            print("\nCorrelation Summary Statistics:")
            print("=" * 40)
            for key, value in summary_stats.items():
                if 'Percentage' in key:
                    print(f"{key}: {value:.2f}%")
                else:
                    print(f"{key}: {value:.4f}")


# 分布统计计算函数
def calculate_distribution_stats(data, var_name):
    clean_data = data.dropna()
    
    if len(clean_data) == 0:
        return None
    
    skewness = skew(clean_data)
    kurt = kurtosis(clean_data)
    
    if len(clean_data) >= 3:
        shapiro_stat, shapiro_p = shapiro(clean_data) if len(clean_data) <= 5000 else (np.nan, np.nan)
        jb_stat, jb_p = jarque_bera(clean_data)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan
        jb_stat, jb_p = np.nan, np.nan
    
    return {
        'Variable': var_name,
        'N': len(clean_data),
        'Mean': np.mean(clean_data),
        'Std': np.std(clean_data),
        'Skewness': skewness,
        'Kurtosis': kurt,
        'Shapiro_Stat': shapiro_stat,
        'Shapiro_p': shapiro_p,
        'JB_Stat': jb_stat,
        'JB_p': jb_p
    }


# 主函数
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    file_path = os.path.join(DATA_DIR, 'scope1_cleaned.xlsx')
    original_df = pd.read_excel(file_path)
    df_transformed = transform_data(original_df)
    print("Dataset shape after transformation:", df_transformed.shape)
    print("\nColumn names after transformation:")
    print(df_transformed.columns.tolist())
    print("\nFirst few rows after transformation:")
    print(df_transformed.head())
    
    # 相关性分析
    print("\n" + "="*80)
    print("STARTING CORRELATION ANALYSIS")
    print("="*80)
    corr_analyzer = CorrelationAnalyzer(df_transformed)
    significant_vars = corr_analyzer.compute_correlations()
    all_vars = corr_analyzer.compute_correlations_all()
    
    corr_analyzer.plot_heatmap(save_path=os.path.join(RESULTS_DIR, 'correlation_heatmap.png'))
    corr_analyzer.plot_heatmap(save_path=os.path.join(RESULTS_DIR, 'correlation_heatmap_all.png'), use_all_vars=True)
    
    corr_analyzer.save_results(RESULTS_DIR, include_all=True)
    print(f"\nCorrelation analysis complete. Heatmaps saved to: {RESULTS_DIR}")
    print(f"Number of variables with significant correlation (|r| > {corr_analyzer.threshold}): {len(significant_vars)-1}")
    print(f"Total number of variables analyzed: {len(all_vars)-1}")
    print("\nVariables with significant correlation to Scope 1:")
    for var in significant_vars[1:]: 
        correlation = corr_analyzer.corr_matrix.iloc[0][var]
        print(f"- {var}: {correlation:.3f}")
    
    # 分布分析
    print("\n" + "="*80)
    print("STARTING DISTRIBUTION ANALYSIS")
    print("="*80)
    key_variables = [
        'TA', 'Inflation', 'TL', 'PPE', 'SOxEmission', 'ROE'
    ]
    
    available_vars = [var for var in key_variables if var in df_transformed.columns]
    print(f"\nAvailable variables for distribution analysis: {available_vars}")
    
    possible_roe_names = ['ROE', 'roe', 'ROE_ratio', 'ReturnOnEquity', 'Return_on_Equity']
    roe_found = None
    for roe_name in possible_roe_names:
        if roe_name in df_transformed.columns:
            roe_found = roe_name
            break
    
    if roe_found and roe_found not in available_vars:
        available_vars.append(roe_found)
        print(f"Found ROE as: {roe_found}")
    
    print(f"\nFinal variables for distribution analysis: {available_vars}")
    print(f"Number of variables: {len(available_vars)}")
    
    if len(available_vars) < len(key_variables):
        print("\nAll available columns in dataset:")
        for i, col in enumerate(df_transformed.columns):
            print(f"{i+1}. {col}")
    
    stats_results = []
    for var in available_vars:
        if var in df_transformed.columns:
            result = calculate_distribution_stats(df_transformed[var], var)
            if result:
                stats_results.append(result)
    
    stats_df = pd.DataFrame(stats_results)
    print("\n" + "="*80)
    print("DISTRIBUTION STATISTICS TABLE")
    print("="*80)
    print(stats_df.round(4))
    
    n_vars = len(available_vars)
    if n_vars <= 6:
        n_cols = 3
        n_rows = 2
    elif n_vars <= 9:
        n_cols = 3
        n_rows = 3
    else:
        n_cols = 4
        n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle('Distribution Analysis of Independent Variables with High Correlation Showing Deviations from Normality', 
                 fontsize=14, fontweight='bold')
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, var in enumerate(available_vars):
        if i >= len(axes):
            break
            
        ax = axes[i]
        data = df_transformed[var].dropna()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(var)
            continue
        
        n_bins = min(30, int(np.sqrt(len(data))))
        counts, bins, patches = ax.hist(data, bins=n_bins, density=True, alpha=0.7, 
                                      color='skyblue', edgecolor='black', linewidth=0.5)
        
        mu, sigma = np.mean(data), np.std(data)
        x = np.linspace(data.min(), data.max(), 100)
        normal_curve = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, normal_curve, 'r--', linewidth=2, label='Normal Distribution')
        
        skewness = skew(data)
        kurt = kurtosis(data)
        
        if var in corr_analyzer.corr_matrix.index:
            corr_value = corr_analyzer.corr_matrix.loc['Scope1', var] if 'Scope1' in corr_analyzer.corr_matrix.index else None
            r_text = f"r = {corr_value:.3f}" if corr_value is not None else ""
            stats_text = f'Skew: {skewness:.2f}\nKurt: {kurt:.2f}\n{r_text}'
        else:
            stats_text = f'Skew: {skewness:.2f}\nKurt: {kurt:.2f}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=9)
        
        ax.set_title(f'{var}', fontweight='bold', fontsize=11)
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        if i == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    
    for j in range(len(available_vars), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'distribution_analysis.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nDistribution figure saved to: {fig_path}")
    
    print("\n" + "="*100)
    print("SUPPLEMENTARY TABLE FOR PAPER")
    print("="*100)
    print("Table X. Descriptive Statistics and Normality Tests for Selected High-Correlation Variables")
    print("-"*100)
    
    paper_table = stats_df[['Variable', 'N', 'Mean', 'Std', 'Skewness', 'Kurtosis', 
                           'Shapiro_p', 'JB_p']].copy()
    paper_table['Shapiro_p'] = paper_table['Shapiro_p'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
    paper_table['JB_p'] = paper_table['JB_p'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
    
    for col in ['Mean', 'Std', 'Skewness', 'Kurtosis']:
        paper_table[col] = paper_table[col].round(4)
    
    print(paper_table.to_string(index=False))
    
    print("\nNote: Analysis focuses on independent variables with |correlation| ≥ 0.4 with Scope1 emissions.")
    print("Shapiro_p and JB_p are p-values from Shapiro-Wilk and Jarque-Bera normality tests respectively.")
    print("P-values < 0.05 indicate significant deviation from normal distribution.")
    
    if 'ROE' not in available_vars and roe_found is None:
        print("\nNote: ROE variable not found in dataset. Please check variable name in the dataset.")
    
    print("\n" + "="*80)
    print("DISTRIBUTION INTERPRETATION SUMMARY")
    print("="*80)
    
    for _, row in stats_df.iterrows():
        var_name = row['Variable']
        skewness = row['Skewness']
        kurt = row['Kurtosis']
        shapiro_p = row['Shapiro_p']
        
        print(f"\n{var_name}:")
        
        if abs(skewness) < 0.5:
            skew_interp = "approximately symmetric"
        elif skewness > 0.5:
            skew_interp = "positively skewed (right tail)"
        else:
            skew_interp = "negatively skewed (left tail)"
        
        if abs(kurt) < 0.5:
            kurt_interp = "normal peakedness"
        elif kurt > 0.5:
            kurt_interp = "heavy-tailed (leptokurtic)"
        else:
            kurt_interp = "light-tailed (platykurtic)"
        
        if pd.notna(shapiro_p):
            if shapiro_p < 0.05:
                norm_interp = "significantly deviates from normal distribution"
            else:
                norm_interp = "does not significantly deviate from normal distribution"
        else:
            norm_interp = "normality test unavailable"
        
        print(f"  - Distribution: {skew_interp}, {kurt_interp}")
        print(f"  - Normality: {norm_interp}")
        if pd.notna(shapiro_p):
            print(f"  - Shapiro-Wilk p-value: {shapiro_p:.4f}")
    
    print(f"\nAll results saved to: {RESULTS_DIR}")
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()