import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import shapiro
from itertools import combinations
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import plot_partregress_grid
from data_transform import transform_data
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
Conditional_Checking_DIR = os.path.join(PROJECT_ROOT, 'src', 'preprocessing')

class RegressionDiagnostics:
    def __init__(self, data, output_dir='.'):
        self.data = data
        self.output_dir = output_dir
        self.dependent_var = self.data.columns[0]
        self.independent_vars = self.data.columns[1:].tolist()
        
        X = self.data[self.independent_vars]
        X = sm.add_constant(X)
        y = self.data[self.dependent_var]
        self.model = sm.OLS(y, X).fit()
        self.residuals = self.model.resid

    def durbin_watson_test(self):
        residuals = np.array(self.residuals)
        diff_resid = np.diff(residuals)
        dw_stat = np.sum(diff_resid**2) / np.sum(residuals**2)
        
        interpretation = ""
        if dw_stat < 1.5:
            interpretation = "Positive autocorrelation may be present"
        elif dw_stat > 2.5:
            interpretation = "Negative autocorrelation may be present"
        else:
            interpretation = "No significant autocorrelation detected"
            
        return {
            'statistic': dw_stat,
            'interpretation': interpretation
        }

    def plot_outliers(self, save_path=None):
        n_columns = len(self.data.columns)
        n_rows = (n_columns + 2) // 3
        n_cols = min(3, n_columns)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        fig.suptitle('Outlier Analysis - Box Plots', fontsize=16, y=1.02)
        axes = axes.flatten()
        
        for idx, column in enumerate(self.data.columns):
            sns.boxplot(y=self.data[column], ax=axes[idx],
                    fliersize=10,
                    flierprops={'marker': '*',
                               'markerfacecolor': 'red',
                               'markeredgecolor': 'black',
                               'markersize': 12})
            
            axes[idx].set_title(f'{column}', fontsize=12, pad=10)
            axes[idx].set_ylabel('Values')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].yaxis.grid(True, linestyle='--', alpha=0.7)
            
            for spine in axes[idx].spines.values():
                spine.set_linewidth(2)
        
        for idx in range(len(self.data.columns), len(axes)):
            fig.delaxes(axes[idx])
            
        plt.tight_layout()
        
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def check_normality(self):
        shapiro_results = {}
        for column in self.data.columns:
            stat, p_value = stats.shapiro(self.data[column])
            shapiro_results[column] = {
                'Statistic': stat,
                'P-value': p_value,
                'Is_Normal': p_value > 0.05
            }
        return pd.DataFrame(shapiro_results).T

    def check_multicollinearity(self):
        def calculate_vif(X):
            vif_data = []
            for var in X.columns:
                y = X[var]
                X_i = X.drop(columns=[var])
                X_i = sm.add_constant(X_i)
                model = sm.OLS(y, X_i).fit()
                r2 = model.rsquared
                vif = 1 / (1 - r2)
                vif_data.append({'Variable': var, 'VIF': vif})
            return pd.DataFrame(vif_data)

        X = self.data[self.independent_vars].copy()
        current_vars = self.independent_vars.copy()
        iteration = 1
        vif_history = []

        while True:
            vif_df = calculate_vif(X)
            max_vif = vif_df['VIF'].max()
            
            results = []
            for _, row in vif_df.iterrows():
                results.append({
                    'Variable': row['Variable'],
                    'VIF': row['VIF'],
                    'High_Multicollinearity': row['VIF'] > 5,
                    'Status': 'Current'
                })
            
            results_df = pd.DataFrame(results)
            vif_history.append(results_df)
            
            if iteration == 1:  # Only open in write mode for first iteration
                mode = 'w'
            else:
                mode = 'a'
                
            with open(os.path.join(output_dir, 'vif_analysis.txt'), mode) as f:
                f.write(f"\nVIF Analysis - Iteration {iteration}:\n")
                f.write(results_df.to_string())
                f.write("\n")
            
            print(f"\nVIF Analysis - Iteration {iteration}:")
            print(results_df)
            
            if max_vif <= 5:
                # Save final results
                with open(os.path.join(self.output_dir, 'vif_analysis.txt'), 'a') as f:
                    f.write("\nMulticollinearity Analysis:\n")
                    f.write(results_df.to_string())
                    f.write("\n")
                break
                
            var_to_remove = vif_df.loc[vif_df['VIF'].idxmax(), 'Variable']
            X = X.drop(columns=[var_to_remove])
            current_vars.remove(var_to_remove)
            iteration += 1
        
        return vif_history[-1]

    def plot_partial_regression(self, save_path=None):
        fig = plt.figure(figsize=(20, 15))
        plot_partregress_grid(self.model, fig=fig)
        
        fig.suptitle('Partial Regression Plots', fontsize=16)
        
        axes = fig.get_axes()
        for ax, col in zip(axes, self.independent_vars):
            coef = self.model.params[col]
            se = self.model.bse[col]
            t = self.model.tvalues[col]
            p = self.model.pvalues[col]
            ax.text(0.05, 0.05, 
                   f'{col}\ncoef = {coef:.8f}\nse = {se:.8f}\nt = {t:.2f}\np = {p:.4f}',
                   transform=ax.transAxes, fontsize=8)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return self.model.summary()
    
    def levene_homogeneity_test(self):
        levene_results = {}
        for col1, col2 in combinations(self.data.columns, 2):
            stat, p_value = stats.levene(self.data[col1], self.data[col2])
            levene_results[f"{col1} vs {col2}"] = {
                'Statistic': stat,
                'P-value': p_value,
                'Equal_Variance': p_value > 0.05
            }
        return pd.DataFrame(levene_results).T

    def independent_ttest(self):
        ttest_results = {}
        for col1, col2 in combinations(self.data.columns, 2):
            stat, p_value = stats.ttest_ind(self.data[col1], self.data[col2])
            ttest_results[f"{col1} vs {col2}"] = {
                'Statistic': stat,
                'P-value': p_value,
                'Significantly_Different': p_value < 0.05
            }
        return pd.DataFrame(ttest_results).T

    def find_outliers(self, column):
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.data[column][(self.data[column] < lower_bound) | 
                                   (self.data[column] > upper_bound)]
        return outliers

    def save_test_results(self, output_dir='.'):
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Shapiro-Wilk test results
        with open(os.path.join(output_dir, 'shapiro_results.txt'), 'w') as f:
            f.write("Normality Test Results:\n")
            f.write(self.check_normality().to_string())
            
        # Save Levene test results
        with open(os.path.join(output_dir, 'levene_results.txt'), 'w') as f:
            f.write("Levene's Test Results:\n")
            f.write(self.levene_homogeneity_test().to_string())
            
        # Save t-test results
        with open(os.path.join(output_dir, 'ttest_results.txt'), 'w') as f:
            f.write("T-Test Results:\n")
            f.write(self.independent_ttest().to_string())
            
        # Save Durbin-Watson results
        with open(os.path.join(output_dir, 'durbin_watson_results.txt'), 'w') as f:
            dw_results = self.durbin_watson_test()
            f.write("Durbin-Watson Test Results:\n")
            f.write(f"Statistic: {dw_results['statistic']:.4f}\n")
            f.write(f"Interpretation: {dw_results['interpretation']}")
            
        # Save outliers results
        with open(os.path.join(output_dir, 'outliers_results.txt'), 'w') as f:
            f.write("Outlier Analysis Results:\n")
            for column in self.data.columns:
                outliers = self.find_outliers(column)
                f.write(f"\n{column}:\n")
                f.write(f"Number of outliers: {len(outliers)}\n")
                if len(outliers) > 0:
                    f.write("Outlier values:\n")
                    f.write(str(outliers.values))

    def run_all_diagnostics(self, save_plots=False, save_results=False, output_dir='.'):
        if save_plots or save_results:
            os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'outliers': {col: self.find_outliers(col) for col in self.data.columns},
            'normality': self.check_normality(),
            'multicollinearity': self.check_multicollinearity(),
            'durbin_watson': self.durbin_watson_test(),
            'levene_test': self.levene_homogeneity_test(),
            'ttest': self.independent_ttest()
        }
        
        if save_plots:
            self.plot_outliers(save_path=os.path.join(output_dir, 'boxplots.png'))
            self.plot_partial_regression(save_path=os.path.join(output_dir, 'partial_regression.png'))
        else:
            self.plot_outliers()
            self.plot_partial_regression()
        
        if save_results:
            self.save_test_results(output_dir)
        
        return results

if __name__ == "__main__":
    file_path = 'data/delete1_selected.xlsx'
    original_df = pd.read_excel(file_path)
    df_transformed = transform_data(original_df)
    
    diagnostics = RegressionDiagnostics(df_transformed)
    output_dir = os.path.join(Conditional_Checking_DIR, 'results')
    
    results = diagnostics.run_all_diagnostics(
        save_plots=True,
        save_results=True,
        output_dir=output_dir
    )
    
    print("\nDiagnostic Results Summary:")
    print("\nDurbin-Watson Test:")
    print(f"Statistic: {results['durbin_watson']['statistic']:.4f}")
    print(f"Interpretation: {results['durbin_watson']['interpretation']}")
    
    print("\nNormality Tests:")
    print(results['normality'])
    
    print("\nLevene's Tests:")
    print(results['levene_test'])
    
    print("\nT-Tests:")
    print(results['ttest'])
    
    print("\nMulticollinearity Analysis:")
    print(results['multicollinearity'])
    
    print("\nOutlier Analysis:")
    for var, outliers in results['outliers'].items():
        print(f"\n{var}:")
        print(f"Number of outliers: {len(outliers)}")
        if len(outliers) > 0:
            print("Outlier values:")
            print(outliers.values)