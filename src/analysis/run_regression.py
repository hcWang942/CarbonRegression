import pandas as pd
import warnings
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tabulate import tabulate
import os

sys.path.append(str(Path(__file__).parent.parent))
from preprocessing.data_transform import transform_data
from regression_models import RegressionModels

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'src', 'analysis', 'regression_result_test')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

def print_results(model_name, metrics, coefficients, best_alpha=None):
    result_str = f"\n=== {model_name} Regression Results ===\n"
    if best_alpha is not None:
        result_str += f"Best alpha: {best_alpha:.4f}\n"
    
    params_table = [["Intercept", f"{coefficients['intercept']:.4f}"]]
    for feature, coef in coefficients['coefficients'].items():
        params_table.append([feature, f"{coef:.4f}"])
    
    result_str += "\n=== Model Parameters ===\n"
    result_str += tabulate(params_table, headers=["Feature", "Coefficient"], tablefmt="pretty")
    
    # Add accuracy metrics to the output
    metrics_table = []
    for metric, value in metrics.items():
        # Format accuracy metrics differently for better readability
        if metric.startswith('accuracy_within_'):
            metrics_table.append([metric, f"{value:.2f}%"])
        else:
            metrics_table.append([metric, f"{value:.4f}"])
    
    result_str += "\n\n=== Model Metrics ===\n"
    result_str += tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="pretty")
    
    print(result_str)
    return result_str

def load_and_transform_data(filename):
    file_path = os.path.join(DATA_DIR, filename)
    df = pd.read_excel(file_path)
    return transform_data(df)

def save_results(results_str, filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, filename), 'w') as f:
        f.write(results_str)

def main():
    # Create directories for results and plots
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Linear Regression on first dataset
    print("\n===== Running Linear Regression =====")
    df_mlr = load_and_transform_data('scope1_regression1.xlsx')
    target_col = df_mlr.columns[0]
    models_mlr = RegressionModels(df_mlr, target_col)
    
    # Run with visualization enabled, saving to the plots directory
    model, y_pred, metrics, coefficients = models_mlr.fit_linear(visualize=True, save_dir=PLOTS_DIR)
    mlr_results = print_results("Linear", metrics, coefficients)
    save_results(mlr_results, 'mlr_results.txt')
    
    # Penalized Regressions on second dataset
    print("\n===== Running Ridge and Lasso Regressions =====")
    df_penalized = load_and_transform_data('scope1_regression2.xlsx')
    target_col = df_penalized.columns[0]
    models_penalized = RegressionModels(df_penalized, target_col)
    
    # Run Ridge regression with visualization
    best_alpha, model, y_pred, metrics, coefficients = models_penalized.fit_ridge(visualize=True, save_dir=PLOTS_DIR)
    ridge_results = print_results("Ridge", metrics, coefficients, best_alpha)
    save_results(ridge_results, 'ridge_results.txt')
    
    # Run Lasso regression with visualization
    best_alpha, model, y_pred, metrics, coefficients = models_penalized.fit_lasso(visualize=True, save_dir=PLOTS_DIR)
    lasso_results = print_results("Lasso", metrics, coefficients, best_alpha)
    save_results(lasso_results, 'lasso_results.txt')
    
    # Generate comparative visualizations
    print("\n===== Generating Model Comparisons =====")
    models_compared = models_penalized.compare_models(save_dir=PLOTS_DIR)
    print(f"Comparison plots saved to: {PLOTS_DIR}")
    
    print(f"\nAll analysis complete! Results and visualizations saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()