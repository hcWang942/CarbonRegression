import pandas as pd
import numpy as np
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'src', 'analysis', 'results')

def calculate_percentage_difference(original, predicted):
    return np.abs((original - predicted) / original) * 100

def categorize_percentage_difference(differences):
    categories = {
        '20%': 0,
        '40%': 0,
        '60%': 0,
        '80%': 0,
        '>80%': 0
    }
    for diff in differences:
        if diff <= 20:
            categories['20%'] += 1
        elif diff <= 40:
            categories['40%'] += 1
        elif diff <= 60:
            categories['60%'] += 1
        elif diff <= 80:
            categories['80%'] += 1
        else:
            categories['>80%'] += 1
    return categories


def calculate_predictions(data, coefficients):
    intercept = coefficients.get('Intercept', 0)
    features = [col for col in data.columns if col in coefficients and col != 'Intercept']
    X = data[features].values
    coef = np.array([coefficients[col] for col in features])
    predictions = intercept + np.dot(X, coef)
    print("intercept", intercept)
    return predictions


def test_model(model_name, data_path, coefficients):
    data = pd.read_excel(data_path)
    original_values = data.iloc[:, 0].values
    predicted_values = calculate_predictions(data, coefficients)
    differences = calculate_percentage_difference(original_values, predicted_values)
    category_stats = categorize_percentage_difference(differences)
    print(f"Model: {model_name}")
    for category, count in category_stats.items():
        print(f"{category}: {count}")

    result_df = pd.DataFrame({
        'Original Scope1': original_values,
        'Predicted Scope1': predicted_values
    })

    output_path = os.path.join(RESULTS_DIR, f'scope1_{model_name}_results.xlsx')
    result_df.to_excel(output_path, index=False)
    print(f"Results for {model_name} saved to {output_path}")

    return category_stats

mlr_coefficients = {
    'Intercept': 0.0213,
    'PPE': 0.2810,
    'ROE': -0.1259,
    'TL': 0.1365,
    'TotalWater Use': -0.0202,
    'SOxEmission': 0.4597
}

lasso_coefficients = {
    'Intercept': 0.0213,
    'TA': 0.2654,
    'PPE': 0.0620,
    'ROE': -0.0659,
    'TL': 0.0660,
    'TotalWater Use': -0.0000,
    'SOxEmission': 0.3786
}

ridge_coefficients = {
    'Intercept': 0.0213,
    'TA': 0.1629,
    'PPE': 0.1177,
    'ROE': -0.0980,
    'TL': 0.1115,
    'TotalWater Use': 0.0116,
    'SOxEmission': 0.3466
}

mlr_data_path = os.path.join(DATA_DIR, 'scope1_regression_mlr.xlsx')
mlr_categories = test_model('MLR', mlr_data_path, mlr_coefficients)

lasso_data_path = os.path.join(DATA_DIR, 'scope1_regression_ridge.xlsx')
lasso_categories = test_model('Lasso', lasso_data_path, lasso_coefficients)

ridge_data_path = os.path.join(DATA_DIR, 'scope1_regression_lasso.xlsx')
ridge_categories = test_model('Ridge', ridge_data_path, ridge_coefficients)