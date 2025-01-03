import pandas as pd
import warnings
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from preprocessing.data_transform import transform_data
from regression_models import RegressionModels

warnings.filterwarnings('ignore')

def print_results(model_name, metrics, coefficients, best_alpha=None):
    print(f"\n=== {model_name} Regression Results ===")
    if best_alpha is not None:
        print(f"Best alpha: {best_alpha:.4f}")
        
    print("\n=== Model Parameters ===")
    print(f"Intercept: {coefficients['intercept']:.4f}")
    print("\nFeature coefficients:")
    for feature, coef in coefficients['coefficients'].items():
        print(f"{feature}: {coef:.4f}")
    
    print("\n=== Model Metrics ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

def main():
    file_path = 'data/delete1_selected.xlsx'
    df = pd.read_excel(file_path)
    df_transformed = transform_data(df)
    
    target_col = df_transformed.columns[0]
    models = RegressionModels(df_transformed, target_col)
    
    # Linear Regression
    model, y_pred, metrics, coefficients = models.fit_linear()
    print_results("Linear", metrics, coefficients)
    
    # Ridge Regression
    best_alpha, model, y_pred, metrics, coefficients = models.fit_ridge()
    print_results("Ridge", metrics, coefficients, best_alpha)
    
    # Lasso Regression
    best_alpha, model, y_pred, metrics, coefficients = models.fit_lasso()
    print_results("Lasso", metrics, coefficients, best_alpha)

if __name__ == "__main__":
    main()