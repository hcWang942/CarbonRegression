# Carbon Emission Regression Analysis

![1961752170909_ pic_hd](https://github.com/user-attachments/assets/c98dd9c4-491d-452a-a8bd-8f54b88ded59)

## Overview
This repository presents a comprehensive analysis framework aimed at exploring the relationship between Scope 1 carbon emissions and various socioeconomic indicators within the American Energy sector. Focusing on publicly-listed companies in the oil, gas, and energy sectors, the project utilizes statistical modeling techniques such as Multiple Linear Regression (MLR), Ridge Regression, and LASSO Regression to assess their environmental impact.

## Project Structure
```
CARBON-EMISSION-REGRESSION/
│
├── data/
│   ├── scope1_original.xlsx      # Raw collected data
│   ├── scope1_cleaned.xlsx       # Cleaned data with handling of missing values
│   ├── scope1_selected.xlsx      # Features with correlation ≥ 0.3 with Scope 1
│   ├── scope1_regression_mlr.xlsx   # Dataset for MLR (passed assumption tests)
│   └── scope1_regression_LASSO_Ridge.xlsx   # Dataset for LASSO & Ridge regression
│
└── src/
    ├── analysis/
    │   ├── regression_models.py        # Implementation of regression models
    │   ├── run_other_plots.py          # Script to generate additional comparison plots
    │   └── run_regression_all_result.py # Main script for comprehensive regression analysis
    └── preprocessing/
        ├── assumption_testing_functions.py # Functions for regression assumption testing
        ├── data_transform_functions.py # Functions for data transformation
        ├── run_assumption_testing.py          # Main script for assumption testing
        └── run_preprocess_analysis.py  # Script for preprocessing and correlation analysis
```

### General Pipeline for Carbon Emission Regression Analysis

#### I. Data Processing
This stage focuses on cleaning the raw data, selecting relevant features, and preparing datasets suitable for regression analysis.

1. **Data Cleaning**
    - **Input**: `scope1_original.xlsx` (raw collected data)
    - **Steps**:
        - Remove rows with missing data from the dataset.
        - Implement special handling for the `NumberOfBoardMeetings` column. Retain rows with missing values only in this column since its correlation with Scope 1 emissions is below 0.3.
        - The cleaning process is carried out using functions in `src/preprocessing/data_transform.py`. For example, the `clean_data` function can remove the first row, replace '-' with `NaN`, convert data to numeric, and fill missing values with the mean.

2. **Feature Selection**
    - **Input**: `scope1_cleaned.xlsx`
    - **Steps**:
        - Conduct a correlation analysis between each feature and Scope 1 emissions. This analysis is performed in `src/preprocessing/preprocessing.py`.
        - Select features with a correlation coefficient of at least 0.3 with Scope 1 emissions. These selected features are more likely to have a significant impact on the target variable and will be used in the subsequent regression analysis.

#### II. Regression Analysis
This stage involves preparing datasets for different regression models, running the models, and evaluating their performance.

1. **Dataset Preparation for Regression**
    - **Multiple Linear Regression (MLR)**
        - **Input**: `scope1_selected.xlsx`
        - **Steps**:
            - Apply conditional checks and assumption tests for classical linear regression on the `scope1_selected.xlsx` dataset. These tests include checking for autocorrelation (e.g., Durbin - Watson test), normality (e.g., Shapiro - Wilk test), multicollinearity (e.g., VIF analysis), homoscedasticity (e.g., Levene's test), and linearity (e.g., partial regression plots). The tests are implemented in `src/preprocessing/conditional_checking.py`.
            - Based on the results of the diagnostic tests, carefully select features that meet the requirements of classical linear regression. The resulting dataset `scope1_regression_mlr.xlsx` is then used for MLR.

    - **Penalized Regression (LASSO and Ridge)**
        - **Input**: `scope1_selected.xlsx`
        - **Steps**:
            - Include a wider range of potential predictors from the `scope1_selected.xlsx` dataset to form the `scope1_regression_LASSO_Ridge.xlsx` dataset.
            - This dataset is suitable for regularization techniques like LASSO and Ridge regression, which can handle multicollinearity and prevent overfitting.

2. **Running Regression Models**
    - **Multiple Linear Regression (MLR)**
        - **Input**: `scope1_regression_mlr.xlsx`
        - **Steps**:
            - Use the `RegressionModels` class in `src/analysis/regression_models.py` to fit a linear regression model on the `scope1_regression_mlr.xlsx` dataset.
            - Evaluate the model using metrics such as adjusted R - squared, mean squared error (MSE), root mean squared error (RMSE), mean absolute error (MAE), and accuracy within certain error thresholds.
            - Optionally, visualize the model results, such as actual vs predicted values and feature importance.

    - **Penalized Regression (LASSO and Ridge)**
        - **Input**: `scope1_regression_LASSO_Ridge.xlsx`
        - **Steps**:
            - Fit Ridge and LASSO regression models using cross - validation to select the best alpha value. This is also implemented in the `RegressionModels` class in `src/analysis/regression_models.py`.
            - Evaluate the models using the same set of metrics as in MLR.
            - Compare the performance of different regression models (MLR, Ridge, and LASSO) and select the best model based on criteria such as adjusted R - squared, RMSE, and accuracy within a certain error range.

## Key Features
### Data Transformation  
- **Rank-based Inverse Normal Transformation (RINT)**:  Applied to make the data more suitable for regression analysis.  
- **Missing Value Handling**:  Ensures data integrity by dealing with missing values appropriately.  

### Assumption Testing  
- **Comprehensive Assumption Testing**:  
  - **Durbin-Watson test**: Checks for autocorrelation in the residuals.  
  - **Shapiro-Wilk test**: Tests the normality of the data.  
  - **VIF analysis**: Assesses multicollinearity among the independent variables.  
  - **Levene's test**: Verifies the homoscedasticity of the data.  
  - **Independent t-tests**: Analyzes the relationships between variables.  
  - **Box plots and statistical tests**: Detect outliers in the data.  
  - **Partial regression plots**: Examine the linearity between variables.  

### Regression Analysis  
- **Correlation analysis**: Filters features based on a correlation threshold.  
- **Cross-validation**: Validates the performance of the regression models.  
- **Model comparison metrics**:  
  - **Adjusted R-squared**: Accounts for the number of predictors in the model.  
  - **Mean Squared Error (MSE)**: Measures the average of the squares of the errors.  
  - **Root Mean Squared Error (RMSE)**: The square root of the MSE, providing a measure of the average magnitude of the error.  
  - **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors in a set of predictions.  

### Advanced Model Selection  
- **Stepwise variable selection**: Automatically selects the most relevant variables for the model.  
- **Cross-validated alpha selection for Ridge and LASSO**: Chooses the optimal regularization parameter.  
- **Feature importance analysis through coefficient paths**: Identifies the relative importance of each feature in the model.  

### Regression Models  
- **Multiple Linear Regression (MLR)**:  A basic linear model that predicts the relationship between a dependent variable and multiple independent variables.  
- **Ridge Regression with cross-validation**:   A regularized regression method that helps to reduce the impact of multicollinearity.  
- **LASSO Regression with cross-validation**:  Another regularized regression method that can perform feature selection by shrinking some coefficients to zero.  

## Usage

### 1. Data Preprocessing
```python
python src/preprocessing/run_assumption_testing.py
python src/preprocessing/run_preprocess_analysis.py
```

### 2. Running Regression Analysis
```python
python src/analysis/run_regression_all_result.py
python src/analysis/run_other_plots.py
```

## Dependencies
- pandas
- numpy
- scikit-learn
- statsmodels
- scipy
- matplotlib
- seaborn
- tabulate
- openpyxl
- pathlib

## Installation
```bash
git clone https://github.com/hcWang942/CARBON-EMISSION-REGRESSION.git
cd CARBON-EMISSION-REGRESSION
pip install -r requirements.txt
```

## Contributing & Contributors ![Contributors](https://img.shields.io/github/contributors/hcWang942/CarbonRegression)

**hcWang942** - *Project Maintainer* - [GitHub](https://github.com/hcWang942)



Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License.

## Authors
**Qiyan Ma<sup>a,1</sup>, Haocheng Wang<sup>b,1</sup>, Zhen Xin Phuang<sup>c,d</sup>, Wai Lam Ng<sup>c,d</sup>, Xiongfeng Pan<sup>e</sup>, Hua Shang<sup>e</sup>, Kok Sin Woon<sup>c,d,*</sup>**  
<sup>a</sup> School of Economics and Management, Xiamen University Malaysia, Jalan Sunsuria, Bandar Sunsuria, 43900 Sepang, Selangor, Malaysia  
<sup>b</sup> School of Mathematics and Physics, Xiamen University Malaysia, Jalan Sunsuria, Bandar Sunsuria, 43900 Sepang, Selangor, Malaysia  
<sup>c</sup> School of Energy and Chemical Engineering, Department of New Energy Science and Engineering, Xiamen University Malaysia, Jalan Sunsuria, Bandar Sunsuria, 43900 Sepang, Selangor, Malaysia  
<sup>d</sup> Thrust of Carbon Neutrality and Climate Change, The Hong Kong University of Science and Technology (Guangzhou), Guangdong Province, 511455, China  
<sup>e</sup> School of Economics and Management, Dalian University of Technology, Dalian, Liaoning Province, 116024, China  
<sup>*</sup> Corresponding author: koksinwoon@hkust-gz.edu.cn  
<sup>1</sup> Qiyan Ma and Haocheng Wang contributed equally to this research

## Citation

If you use this code in your research, please cite:

```bibtex
@article{article,
  title={Predicting Corporate Emissions in the Oil and Gas Sector: A Comparative Regression Model Analysis using Environmental, Financial, and Governance Indicators},
  author={Qiyan Ma*, Haocheng Wang*, Zhen Xin Phuang, Wai Lam Ng, Xiongfeng Pan, Hua Shang, Kok Sin Woon},
  journal={Energy Economics},
  year={2025}
}
```
