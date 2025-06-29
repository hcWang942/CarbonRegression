import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
from data_transform import transform_data

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'src', 'preprocessing', 'preprocessing_results')

class CorrelationAnalyzer:
    def __init__(self, data, threshold=0.3):
        """
        Parameters:
        -----------
        data : 
            The input dataframe containing Scope 1 and other variables
        threshold : 
            The correlation threshold to filter variables (default: 0.3)
        """
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
        """Compute correlations for all variables (no threshold filtering)"""
        self.corr_matrix = self.data.corr()
        scope1_corr = self.corr_matrix.iloc[0]
        # Return all variables except Scope1 itself
        all_vars = scope1_corr.index.tolist()
        return all_vars
    
    def plot_heatmap(self, save_path=None, use_all_vars=False):
        plt.figure(figsize=(15, 12) if use_all_vars else (12, 10))
        
        if use_all_vars:
            plot_corr = self.corr_matrix
            title_suffix = "All Variables"
        else:
            # Create heatmap using significant variables only
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
        
        # Save significant variables results
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
        
        # Save significant variables CSV
        results_df.to_csv(os.path.join(output_dir, 'significant_correlations.csv'), index=False)
        
        if include_all:
            # Save all variables results
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
            
            all_results_df.to_csv(os.path.join(output_dir, 'all_correlations.csv'), index=False)
            
            corr_values = all_scope1_corr.drop('Scope1').abs()  # Remove Scope1 itself and take absolute values
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
            
            summary_df = pd.DataFrame([summary_stats])
            summary_df.to_csv(os.path.join(output_dir, 'correlation_summary_statistics.csv'), index=False)
            
            print("\nCorrelation Summary Statistics:")
            print("=" * 40)
            for key, value in summary_stats.items():
                if 'Percentage' in key:
                    print(f"{key}: {value:.2f}%")
                else:
                    print(f"{key}: {value:.4f}")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, 'scope1_cleaned.xlsx')
    original_df = pd.read_excel(file_path)
    df_transformed = transform_data(original_df)
    analyzer = CorrelationAnalyzer(df_transformed)
    
    significant_vars = analyzer.compute_correlations()
    
    all_vars = analyzer.compute_correlations_all()
    
    # Plot heatmap for significant variables
    analyzer.plot_heatmap(save_path=os.path.join(RESULTS_DIR, 'correlation_heatmap.png'))
    
    # Plot heatmap for all variables
    analyzer.plot_heatmap(save_path=os.path.join(RESULTS_DIR, 'correlation_heatmap_all.png'), use_all_vars=True)
    
    # Save results (including all variables)
    analyzer.save_results(RESULTS_DIR, include_all=True)
    
    print(f"\nAnalysis complete. Results saved to: {RESULTS_DIR}")
    print(f"Number of variables with significant correlation (|r| > {analyzer.threshold}): {len(significant_vars)-1}")
    print(f"Total number of variables analyzed: {len(all_vars)-1}")
    
    print("\nVariables with significant correlation to Scope 1:")
    for var in significant_vars[1:]: 
        correlation = analyzer.corr_matrix.iloc[0][var]
        print(f"- {var}: {correlation:.3f}")
    
    print(f"\nVariables with |correlation| <= {analyzer.threshold}:")
    non_significant_vars = [var for var in all_vars if var not in significant_vars and var != 'Scope1']
    for var in non_significant_vars:
        correlation = analyzer.corr_matrix.iloc[0][var]
        print(f"- {var}: {correlation:.3f}")

if __name__ == "__main__":
    main()