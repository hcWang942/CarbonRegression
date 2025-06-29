import pandas as pd
import warnings
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tabulate import tabulate
import os
import datetime

sys.path.append(str(Path(__file__).parent.parent))
from preprocessing.data_transform import transform_data
from regression_models_cv import RegressionModels

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'src', 'analysis', 'regression_result_test_cv')
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
    
    # Create a single comprehensive results file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    comprehensive_results_file = os.path.join(RESULTS_DIR, f'comprehensive_results_{timestamp}.txt')
    
    with open(comprehensive_results_file, 'w') as comprehensive_file:
        # Add header with timestamp
        comprehensive_file.write(f"=====================================================\n")
        comprehensive_file.write(f"  COMPREHENSIVE REGRESSION ANALYSIS RESULTS\n")
        comprehensive_file.write(f"  Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        comprehensive_file.write(f"=====================================================\n\n")
        
        # Linear Regression on first dataset
        section_header = "\n" + "="*50 + "\n" + "LINEAR REGRESSION RESULTS" + "\n" + "="*50 + "\n"
        print(section_header)
        comprehensive_file.write(section_header)
        
        df_mlr = load_and_transform_data('scope1_regression1.xlsx')
        target_col = df_mlr.columns[0]
        comprehensive_file.write(f"Dataset: scope1_regression1.xlsx\n")
        comprehensive_file.write(f"Target variable: {target_col}\n")
        comprehensive_file.write(f"Number of features: {len(df_mlr.columns)-1}\n")
        comprehensive_file.write(f"Number of samples: {len(df_mlr)}\n\n")
        
        models_mlr = RegressionModels(df_mlr, target_col, test_size=0.4)
        
        # Run with visualization enabled, saving to the plots directory
        model, y_pred, metrics, coefficients = models_mlr.fit_linear(visualize=True, save_dir=PLOTS_DIR)
        mlr_results = print_results("Linear", metrics, coefficients)
        save_results(mlr_results, 'mlr_results.txt')
        comprehensive_file.write(mlr_results + "\n")
        
        # Penalized Regressions on second dataset
        section_header = "\n" + "="*50 + "\n" + "RIDGE REGRESSION RESULTS" + "\n" + "="*50 + "\n"
        print(section_header)
        comprehensive_file.write(section_header)
        
        df_penalized = load_and_transform_data('scope1_regression2.xlsx')
        target_col = df_penalized.columns[0]
        comprehensive_file.write(f"Dataset: scope1_regression2.xlsx\n")
        comprehensive_file.write(f"Target variable: {target_col}\n")
        comprehensive_file.write(f"Number of features: {len(df_penalized.columns)-1}\n")
        comprehensive_file.write(f"Number of samples: {len(df_penalized)}\n\n")
        
        models_penalized = RegressionModels(df_penalized, target_col, test_size=0.4)
        
        # Run Ridge regression with visualization
        best_alpha, model, y_pred, metrics, coefficients = models_penalized.fit_ridge(visualize=True, save_dir=PLOTS_DIR)
        ridge_results = print_results("Ridge", metrics, coefficients, best_alpha)
        save_results(ridge_results, 'ridge_results.txt')
        comprehensive_file.write(ridge_results + "\n")
        
        # Run Lasso regression with visualization
        section_header = "\n" + "="*50 + "\n" + "LASSO REGRESSION RESULTS" + "\n" + "="*50 + "\n"
        print(section_header)
        comprehensive_file.write(section_header)
        
        best_alpha, model, y_pred, metrics, coefficients = models_penalized.fit_lasso(visualize=True, save_dir=PLOTS_DIR)
        lasso_results = print_results("Lasso", metrics, coefficients, best_alpha)
        save_results(lasso_results, 'lasso_results.txt')
        comprehensive_file.write(lasso_results + "\n")
        
        # Run ElasticNet regression with visualization
        section_header = "\n" + "="*50 + "\n" + "ELASTIC NET REGRESSION RESULTS" + "\n" + "="*50 + "\n"
        print(section_header)
        comprehensive_file.write(section_header)

        best_alpha, best_l1_ratio, model, y_pred, metrics, coefficients = models_penalized.fit_elastic_net(visualize=True, save_dir=PLOTS_DIR)
        elastic_results = print_results("ElasticNet", metrics, coefficients, best_alpha)
        elastic_results = elastic_results.replace("Best alpha:", f"Best alpha: {best_alpha:.4f}, Best l1_ratio: {best_l1_ratio:.2f}")
        save_results(elastic_results, 'elastic_net_results.txt')
        comprehensive_file.write(elastic_results + "\n")
        
        # Add cross-validation section
        section_header = "\n" + "="*50 + "\n" + "CROSS-VALIDATION RESULTS" + "\n" + "="*50 + "\n"
        print(section_header)
        comprehensive_file.write(section_header)
        comprehensive_file.write("Running 5-fold cross-validation on all models...\n\n")

        cv_results = models_penalized.cross_validate_models(cv=5, visualize=True, save_dir=PLOTS_DIR)
        comprehensive_file.write("Cross-validation completed. Results summary:\n\n")

        # Create CV results table
        cv_table = []
        cv_table.append(["Model", "CV R² (mean±std)", "CV RMSE (mean±std)", "CV Accuracy ±20% (mean±std)", "CV Accuracy ±40% (mean±std)"])

        for model_name, metrics in cv_results.items():
            alpha_info = f", α={metrics['best_alpha']:.4f}" if 'best_alpha' in metrics else ""
            l1_info = f", l1_ratio={metrics['best_l1_ratio']:.2f}" if 'best_l1_ratio' in metrics else ""
            
            cv_table.append([
                f"{model_name}{alpha_info}{l1_info}",
                f"{metrics['cv_r2_mean']:.4f}±{metrics['cv_r2_std']:.4f}",
                f"{metrics['cv_rmse_mean']:.4f}±{metrics['cv_rmse_std']:.4f}",
                f"{metrics['cv_accuracy_20_mean']:.2f}%±{metrics['cv_accuracy_20_std']:.2f}%",
                f"{metrics['cv_accuracy_40_mean']:.2f}%±{metrics['cv_accuracy_40_std']:.2f}%"
            ])

        comprehensive_file.write(tabulate(cv_table, headers="firstrow", tablefmt="pretty"))
        comprehensive_file.write("\n\n")
        
        # Generate comparative visualizations
        section_header = "\n" + "="*50 + "\n" + "MODEL COMPARISON" + "\n" + "="*50 + "\n"
        print(section_header)
        comprehensive_file.write(section_header)
        
        # Add note about comparison plots to the comprehensive file
        comprehensive_file.write("Comparison plots have been generated in the 'plots' directory:\n")
        comprehensive_file.write("- Model accuracy comparisons\n")
        comprehensive_file.write("- R² and adjusted R² comparisons\n")
        comprehensive_file.write("- Error metrics (RMSE and MAE) comparisons\n")
        comprehensive_file.write("- Combined prediction plots\n")
        comprehensive_file.write("- Cross-validation performance comparisons\n")
        
        models_compared = models_penalized.compare_models(save_dir=PLOTS_DIR, include_elastic_net=True)
        print(f"Comparison plots saved to: {PLOTS_DIR}")
        
        # Add summary of all models
        summary_header = "\n" + "="*50 + "\n" + "SUMMARY OF ALL MODELS" + "\n" + "="*50 + "\n"
        comprehensive_file.write(summary_header)
        
        # Create a comparison table of key metrics for all models
        summary_table = []
        
        # Headers for the summary table
        summary_table.append(["Model", "Best Alpha", "R²", "Adj. R²", "RMSE", "Accuracy (±20%)", "Accuracy (±40%)"])
        
        # Linear regression row
        linear_model, linear_pred, linear_metrics, linear_coef = models_mlr.fit_linear(visualize=True, save_dir=PLOTS_DIR)
        mlr_results = print_results("Linear", linear_metrics, linear_coef)
        summary_table.append([
            "Linear", 
            "N/A", 
            f"{linear_metrics['test_r2']:.4f}", 
            f"{linear_metrics['adj_r2']:.4f}", 
            f"{linear_metrics['rmse']:.4f}", 
            f"{linear_metrics['accuracy_within_20%']:.2f}%", 
            f"{linear_metrics['accuracy_within_40%']:.2f}%"
        ])
        
        # Ridge regression row
        ridge_metrics = models_compared["Ridge"]["results"][2]  # metrics is at index 2
        summary_table.append([
            "Ridge", 
            f"{models_compared['Ridge']['alpha']:.4f}", 
            f"{ridge_metrics['test_r2']:.4f}", 
            f"{ridge_metrics['adj_r2']:.4f}", 
            f"{ridge_metrics['rmse']:.4f}", 
            f"{ridge_metrics['accuracy_within_20%']:.2f}%", 
            f"{ridge_metrics['accuracy_within_40%']:.2f}%"
        ])
        
        # Lasso regression row
        lasso_metrics = models_compared["Lasso"]["results"][2]  # metrics is at index 2
        summary_table.append([
            "Lasso", 
            f"{models_compared['Lasso']['alpha']:.4f}", 
            f"{lasso_metrics['test_r2']:.4f}", 
            f"{lasso_metrics['adj_r2']:.4f}", 
            f"{lasso_metrics['rmse']:.4f}", 
            f"{lasso_metrics['accuracy_within_20%']:.2f}%", 
            f"{lasso_metrics['accuracy_within_40%']:.2f}%"
        ])
        
        # ElasticNet regression row
        elastic_metrics = models_compared["ElasticNet"]["results"][2] if "ElasticNet" in models_compared else metrics
        summary_table.append([
            "ElasticNet", 
            f"{models_compared.get('ElasticNet', {}).get('alpha', best_alpha):.4f} (l1={best_l1_ratio:.2f})", 
            f"{elastic_metrics['test_r2']:.4f}", 
            f"{elastic_metrics['adj_r2']:.4f}", 
            f"{elastic_metrics['rmse']:.4f}", 
            f"{elastic_metrics['accuracy_within_20%']:.2f}%", 
            f"{elastic_metrics['accuracy_within_40%']:.2f}%"
        ])
        
        comprehensive_file.write(tabulate(summary_table, headers="firstrow", tablefmt="pretty"))
        comprehensive_file.write("\n\n")
        
        # Add conclusion
        comprehensive_file.write("\nCONCLUSION:\n")
        
        # Find best model based on adjusted R²
        best_model_adj_r2 = max(["Linear", "Ridge", "Lasso", "ElasticNet"], 
                               key=lambda m: models_compared.get(m, {}).get("results", [None, None, {}])[2].get("adj_r2", 0) 
                               if m != "Linear" else linear_metrics.get("adj_r2", 0))
        
        # Find best model based on RMSE
        best_model_rmse = min(["Linear", "Ridge", "Lasso", "ElasticNet"], 
                             key=lambda m: models_compared.get(m, {}).get("results", [None, None, {}])[2].get("rmse", float('inf')) 
                             if m != "Linear" else linear_metrics.get("rmse", float('inf')))
        
        # Find best model based on accuracy within 20%
        best_model_acc20 = max(["Linear", "Ridge", "Lasso", "ElasticNet"], 
                              key=lambda m: models_compared.get(m, {}).get("results", [None, None, {}])[2].get("accuracy_within_20%", 0) 
                              if m != "Linear" else linear_metrics.get("accuracy_within_20%", 0))

        # Find best model based on CV R²
        best_model_cv_r2 = max(cv_results.keys(), key=lambda m: cv_results[m]['cv_r2_mean'])
        
        comprehensive_file.write(f"- Best model based on adjusted R²: {best_model_adj_r2}\n")
        comprehensive_file.write(f"- Best model based on RMSE: {best_model_rmse}\n")
        comprehensive_file.write(f"- Best model based on accuracy within ±20%: {best_model_acc20}\n")
        comprehensive_file.write(f"- Best model based on cross-validation R²: {best_model_cv_r2}\n\n")
        
        comprehensive_file.write("Overall recommendation: ")
        # Determine overall best model (simple majority vote)
        models_count = {"Linear": 0, "Ridge": 0, "Lasso": 0, "ElasticNet": 0}
        models_count[best_model_adj_r2] += 1
        models_count[best_model_rmse] += 1
        models_count[best_model_acc20] += 1
        models_count[best_model_cv_r2] += 1
        overall_best = max(models_count.items(), key=lambda x: x[1])[0]
        
        comprehensive_file.write(f"{overall_best} regression appears to be the most suitable model for this dataset.\n")
        
    print(f"\nAll analysis complete! Comprehensive results saved to: {comprehensive_results_file}")
    print(f"Individual visualizations saved to: {PLOTS_DIR}")

if __name__ == "__main__":
    main()