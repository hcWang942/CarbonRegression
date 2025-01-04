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
    
    def plot_heatmap(self, save_path=None):
        plt.figure(figsize=(12, 10))
        
        # Create heatmap using significant variables only
        significant_corr = self.corr_matrix.loc[self.significant_vars, self.significant_vars]
        sns.heatmap(significant_corr, 
                   annot=True, 
                   cmap='coolwarm', 
                   vmin=-1, 
                   vmax=1, 
                   center=0,
                   fmt='.2f')
        
        plt.title('Correlation Heatmap (Variables with |correlation| > {})'.format(self.threshold))
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    def save_results(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        scope1_corr = self.corr_matrix.iloc[0][self.significant_vars]
        results_df = pd.DataFrame({
            'Variable': scope1_corr.index,
            'Correlation_with_Scope1': scope1_corr.values
        }).sort_values('Correlation_with_Scope1', ascending=False)
        with open(os.path.join(output_dir, 'correlation_analysis.txt'), 'w') as f:
            f.write("Correlation Analysis Results\n")
            f.write("===========================\n\n")
            f.write(f"Correlation threshold: {self.threshold}\n")
            f.write(f"Number of significant variables: {len(self.significant_vars)-1}\n\n")
            f.write("Significant correlations with Scope 1:\n")
            f.write(results_df.to_string())

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, 'scope1_cleaned.xlsx')
    original_df = pd.read_excel(file_path)
    df_transformed = transform_data(original_df)
    analyzer = CorrelationAnalyzer(df_transformed)
    significant_vars = analyzer.compute_correlations()
    analyzer.plot_heatmap(save_path=os.path.join(RESULTS_DIR, 'correlation_heatmap.png'))
    
    analyzer.save_results(RESULTS_DIR)
    
    print(f"\nAnalysis complete. Results saved to: {RESULTS_DIR}")
    print(f"Number of variables with significant correlation: {len(significant_vars)-1}")
    print("Variables with significant correlation to Scope 1:")
    for var in significant_vars[1:]: 
        correlation = analyzer.corr_matrix.iloc[0][var]
        print(f"- {var}: {correlation:.3f}")

if __name__ == "__main__":
    main()