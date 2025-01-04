import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge, LassoCV, Lasso
from sklearn.metrics import mean_squared_error, r2_score

def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    y_train_array = y_train.values if isinstance(y_train, pd.Series) else y_train
    y_test_array = y_test.values if isinstance(y_test, pd.Series) else y_test
    
    y_train_pred = model.predict(X_train_array)
    y_test_pred = model.predict(X_test_array)
    
    metrics = {
        'train_r2': r2_score(y_train_array, y_train_pred),
        'test_r2': r2_score(y_test_array, y_test_pred),
        'train_mse': mean_squared_error(y_train_array, y_train_pred),
        'test_mse': mean_squared_error(y_test_array, y_test_pred),
        'rmse': np.sqrt(mean_squared_error(y_test_array, y_test_pred)),
        'mae': np.mean(np.abs(y_test_array - y_test_pred))
    }
    
    n_params = len(model.coef_) if model_name == "Linear" else np.sum(model.coef_ != 0) + 1
    metrics['adj_r2'] = adjusted_r2(metrics['test_r2'], len(y_test), n_params)
    
    coefficients = {
        'intercept': model.intercept_,
        'coefficients': dict(zip(X_train.columns, model.coef_))
    }
    
    return model, y_test_pred, metrics, coefficients

class RegressionModels:
    def __init__(self, data, target_col, test_size=0.4, random_state=42):
        self.Y = data[target_col]
        self.X = data.drop(columns=[target_col])
        self.scaler = StandardScaler()
        self.prepare_data(test_size, random_state)
        
    def prepare_data(self, test_size, random_state):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=random_state
        )
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns
        )
    
    def fit_linear(self):
        model = LinearRegression()
        model.fit(self.X_train_scaled.values, self.y_train.values)
        return evaluate_model(
            model, self.X_train_scaled, self.X_test_scaled,
            self.y_train, self.y_test, "Linear"
        )
    
    def fit_ridge(self, alphas=np.logspace(-3, 3, 100)):
        ridge_cv = RidgeCV(alphas=alphas, cv=5)
        ridge_cv.fit(self.X_train_scaled.values, self.y_train.values)
        model = Ridge(alpha=ridge_cv.alpha_)
        model.fit(self.X_train_scaled.values, self.y_train.values)
        results = evaluate_model(
            model, self.X_train_scaled, self.X_test_scaled,
            self.y_train, self.y_test, "Ridge"
        )
        return ridge_cv.alpha_, *results
    
    def fit_lasso(self, alphas=np.logspace(-3, 3, 100)):
        lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)
        lasso_cv.fit(self.X_train_scaled.values, self.y_train.values)
        model = Lasso(alpha=lasso_cv.alpha_)
        model.fit(self.X_train_scaled.values, self.y_train.values)
        results = evaluate_model(
            model, self.X_train_scaled, self.X_test_scaled,
            self.y_train, self.y_test, "Lasso"
        )
        return lasso_cv.alpha_, *results