# Carbon Emission Regression Analysis

This repository contains a comprehensive analysis framework for studying the relationship between Scope 1 carbon emissions and various socioeconomic indicators in the American Energy sector. The analysis focuses on publicly listed companies in the oil, gas, and energy sectors, examining their environmental impact through statistical modeling.(MLR, Ridge, and LASSO)

## Project Structure

```
CARBON-EMISSION-REGRESSION/
│
├── data/
│   ├── scope1_original.xlsx      # Raw collected data
│   ├── scope1_cleaned.xlsx       # Cleaned data with handling of missing values
│   ├── scope1_selected.xlsx      # Features with correlation ≥ 0.3 with Scope 1
│   ├── scope1_regression1.xlsx   # Dataset for MLR (passed conditional checks)
│   └── scope1_regression2.xlsx   # Dataset for LASSO & Ridge regression
│
└── src/
    ├── preprocessing/
    │   ├── data_transform.py           # Data cleaning and transformation
    │   ├── conditional_checking.py      # Regression assumptions testing
    │   ├── preprocessing.py            # Correlation analysis
    │   └── run_diagnostics.py          # Main script for diagnostics
    │
    └── analysis/
        ├── regression_models.py        # Implementation of regression models
        └── run_regression.py           # Main script for regression analysis
```

## Data Processing Pipeline

1. **Data Cleaning** (`scope1_cleaned.xlsx`):
   - Removes rows with missing data
   - Special handling for NumberOfBoardMeetings:
     * Rows with missing values only in NumberOfBoardMeetings are retained
     * This decision was made as this variable's correlation with Scope 1 is below 0.3

2. **Feature Selection** (`scope1_selected.xlsx`):
   - Includes features with correlation coefficient ≥ 0.3 with Scope 1 emissions
   - Based on correlation analysis from preprocessing.py

3. **Regression Datasets**:
   - `scope1_regression1.xlsx`: For Multiple Linear Regression
     * Passed all conditional checks and assumptions for classical linear regression
     * Carefully selected features based on comprehensive diagnostic tests
   - `scope1_regression2.xlsx`: For Penalized Regression (LASSO and Ridge)
     * Includes wider range of potential predictors
     * Suitable for regularization techniques that handle multicollinearity
     * Used when dealing with potential overfitting

## Key Features

- **Data Transformation**:
  - Rank-based Inverse Normal Transformation (RINT)
  - Missing value handling

- **Statistical Diagnostics**:
  - Comprehensive Conditional Checking:
    * Durbin-Watson test for autocorrelation
    * Shapiro-Wilk test for normality
    * VIF analysis for multicollinearity
    * Levene's test for homoscedasticity
    * Independent t-tests for variable relationships
    * Box plots and statistical tests for outlier detection
    * Partial regression plots for linearity analysis
    
  - Regression Analysis:
    * Correlation analysis with threshold filtering
    * Cross-validation for model validation
    * Model comparison metrics:
      - Adjusted R-squared
      - Mean Squared Error (MSE)
      - Root Mean Squared Error (RMSE)
      - Mean Absolute Error (MAE)
  - Advanced Model Selection:
    * Stepwise variable selection
    * Cross-validated alpha selection for Ridge and LASSO
    * Feature importance analysis through coefficient paths
    * Model performance comparison across different regression techniques

- **Regression Models**:
  - Multiple Linear Regression (MLR)
  - Ridge Regression with cross-validation
  - LASSO Regression with cross-validation

## Usage

1. **Data Preprocessing**:
```python
python src/preprocessing/run_diagnostics.py
```

2. **Running Regression Analysis**:
```python
python src/analysis/run_regression.py
```

## Dependencies

- pandas
- numpy
- scikit-learn
- statsmodels
- scipy
- matplotlib
- seaborn

## Installation

```bash
git clone https://github.com/hcWang942/CARBON-EMISSION-REGRESSION.git
cd CARBON-EMISSION-REGRESSION
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{article,
  title={Article Title},
  author={Wang Haocheng, Ma Qiyan, Phuang Zhen Xin, Ng Wai Lam, Woon Kok Sin},
  journal={Energy Economics},
  year={2025}
}
```
