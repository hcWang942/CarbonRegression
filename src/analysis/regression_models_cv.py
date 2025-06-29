import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge, LassoCV, Lasso, ElasticNetCV, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

def adjusted_r2(r2, n, p):
    """Calculate adjusted R-squared metric"""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def calculate_accuracy_thresholds(y_true, y_pred, thresholds=[0.2, 0.4]):
    """Calculate percentage of predictions within given error thresholds
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        thresholds: List of error thresholds (e.g., [0.2, 0.4] for 20% and 40%)
        
    Returns:
        Dictionary with accuracy percentages for each threshold
    """
    accuracy_dict = {}
    for threshold in thresholds:
        # Calculate relative error
        rel_error = np.abs(y_true - y_pred) / np.abs(y_true)
        # Count predictions within threshold
        within_threshold = np.mean(rel_error <= threshold) * 100
        accuracy_dict[f'accuracy_within_{int(threshold*100)}%'] = within_threshold
    
    return accuracy_dict

def cv_evaluate(model, X, y, cv=5):
    """
    Evaluate model using k-fold cross-validation
    
    Args:
        model: Sklearn model object
        X: Features array
        y: Target array
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary with cross-validated R² and RMSE
    """
    # Setup cross-validation strategy
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Calculate basic CV metrics
    cv_r2 = cross_val_score(model, X, y, cv=kf, scoring='r2')
    cv_neg_rmse = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
    cv_neg_mae = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
    
    # Calculate accuracy within thresholds using manual CV
    accuracies_20 = []
    accuracies_40 = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate accuracies
        rel_error = np.abs(y_test - y_pred) / np.abs(y_test)
        accuracy_20 = np.mean(rel_error <= 0.2) * 100
        accuracy_40 = np.mean(rel_error <= 0.4) * 100
        
        accuracies_20.append(accuracy_20)
        accuracies_40.append(accuracy_40)
    
    return {
        'cv_r2_mean': cv_r2.mean(),
        'cv_r2_std': cv_r2.std(),
        'cv_rmse_mean': -cv_neg_rmse.mean(),
        'cv_rmse_std': cv_neg_rmse.std(),
        'cv_mae_mean': -cv_neg_mae.mean(),
        'cv_mae_std': cv_neg_mae.std(),
        'cv_accuracy_20_mean': np.mean(accuracies_20),
        'cv_accuracy_20_std': np.std(accuracies_20),
        'cv_accuracy_40_mean': np.mean(accuracies_40),
        'cv_accuracy_40_std': np.std(accuracies_40)
    }

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate regression model with comprehensive metrics"""
    # Convert to arrays if DataFrames/Series
    X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    y_train_array = y_train.values if isinstance(y_train, pd.Series) else y_train
    y_test_array = y_test.values if isinstance(y_test, pd.Series) else y_test
    
    # Make predictions
    y_train_pred = model.predict(X_train_array)
    y_test_pred = model.predict(X_test_array)
    
    # Calculate standard metrics
    metrics = {
        'train_r2': r2_score(y_train_array, y_train_pred),
        'test_r2': r2_score(y_test_array, y_test_pred),
        'train_mse': mean_squared_error(y_train_array, y_train_pred),
        'test_mse': mean_squared_error(y_test_array, y_test_pred),
        'rmse': np.sqrt(mean_squared_error(y_test_array, y_test_pred)),
        'mae': np.mean(np.abs(y_test_array - y_test_pred))
    }
    
    # Add adjusted R2
    n_params = len(model.coef_) if model_name == "Linear" else np.sum(model.coef_ != 0) + 1
    metrics['adj_r2'] = adjusted_r2(metrics['test_r2'], len(y_test), n_params)
    
    # Add accuracy within thresholds
    accuracy_metrics = calculate_accuracy_thresholds(y_test_array, y_test_pred)
    metrics.update(accuracy_metrics)
    
    # Get model coefficients
    coefficients = {
        'intercept': model.intercept_,
        'coefficients': dict(zip(X_train.columns, model.coef_))
    }
    
    return model, y_test_pred, metrics, coefficients

def plot_predictions(y_true, y_pred, model_name, save_path=None):
    """Plot actual vs predicted values with error bounds
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        model_name: Name of the model for the plot title
        save_path: Path to save the figure, if None, the plot is displayed
    """
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
    
    # Add perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # Add ±20% error bounds
    plt.plot([min_val, max_val], [min_val*0.8, max_val*0.8], 'g:', label='-20% Error Bound')
    plt.plot([min_val, max_val], [min_val*1.2, max_val*1.2], 'g:', label='+20% Error Bound')
    
    # Add ±40% error bounds
    plt.plot([min_val, max_val], [min_val*0.6, max_val*0.6], 'y:', label='-40% Error Bound')
    plt.plot([min_val, max_val], [min_val*1.4, max_val*1.4], 'y:', label='+40% Error Bound')
    
    # Calculate accuracy metrics for display
    accuracy_20 = np.mean(np.abs(y_true - y_pred) / np.abs(y_true) <= 0.2) * 100
    accuracy_40 = np.mean(np.abs(y_true - y_pred) / np.abs(y_true) <= 0.4) * 100
    
    # Customize the plot
    plt.title(f"{model_name} Regression: Actual vs Predicted Values", fontsize=14)
    plt.xlabel("Actual Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add accuracy text
    text_info = (
        f"Accuracy (within ±20%): {accuracy_20:.1f}%\n"
        f"Accuracy (within ±40%): {accuracy_40:.1f}%"
    )
    
    plt.annotate(text_info, xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                 fontsize=10, ha='left', va='top')
    
    # Handle duplicated labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_feature_importance(coefficients, model_name, top_n=None, save_path=None):
    """Visualize feature importance based on absolute coefficient values
    
    Args:
        coefficients: Dictionary of model coefficients
        model_name: Name of the model for the plot title
        top_n: Number of top features to display (None for all)
        save_path: Path to save the figure, if None, the plot is displayed
    """
    # Extract coefficients
    coefs = coefficients['coefficients']
    feature_names = list(coefs.keys())
    importance = np.abs(list(coefs.values()))
    
    # Create DataFrame for sorting
    df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    df_importance = df_importance.sort_values('Importance', ascending=False)
    
    # Select top N features if specified
    if top_n is not None and top_n < len(df_importance):
        df_importance = df_importance.iloc[:top_n]
    
    # Plot
    plt.figure(figsize=(10, max(6, len(df_importance) * 0.3)))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(x='Importance', y='Feature', data=df_importance, palette='viridis')
    
    # Customize plot
    plt.title(f"Feature Importance - {model_name} Regression", fontsize=14)
    plt.xlabel('Absolute Coefficient Value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    
    # Add values to the bars
    for i, importance in enumerate(df_importance['Importance']):
        ax.text(importance + 0.01, i, f"{importance:.4f}", va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_cv_results(cv_results, save_path=None):
    """Plot cross-validation results for all models
    
    Args:
        cv_results: Dictionary of CV results for each model
        save_path: Path to save the figure, if None, the plot is displayed
    """
    # Extract data for plotting
    models = list(cv_results.keys())
    metrics = ['cv_r2_mean', 'cv_rmse_mean', 'cv_accuracy_20_mean', 'cv_accuracy_40_mean']
    metric_labels = ['R²', 'RMSE', 'Accuracy ±20%', 'Accuracy ±40%']
    
    plt.figure(figsize=(15, 10))
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        plt.subplot(2, 2, i+1)
        
        values = [cv_results[model][metric] for model in models]
        errors = [cv_results[model][metric.replace('mean', 'std')] for model in models]
        
        bars = plt.bar(models, values, yerr=errors, capsize=10, alpha=0.7)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, value + (errors[bars.index(bar)] if errors else 0) + 0.01,
                     f'{value:.2f}', ha='center', va='bottom')
        
        plt.title(f'Cross-Validation {label}', fontsize=14)
        plt.ylabel(label)
        plt.grid(axis='y', alpha=0.3)
        
        # For accuracy metrics, set y-axis to start from 0 and end at 100
        if 'Accuracy' in label:
            plt.ylim(0, 100)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

class RegressionModels:
    def __init__(self, data, target_col, test_size=0.4, random_state=42):
        """Initialize regression models with data preprocessing
        
        Args:
            data: DataFrame containing features and target
            target_col: Name of the target column
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.Y = data[target_col]
        self.X = data.drop(columns=[target_col])
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.prepare_data(test_size, random_state)
        
    def prepare_data(self, test_size, random_state):
        """Split and scale the data"""
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
    
    def fit_linear(self, visualize=True, save_dir=None, cv=None):
        """Fit linear regression model and evaluate
        
        Args:
            visualize: Whether to create visualizations
            save_dir: Directory to save visualizations, if None, displays plots
            cv: If specified, number of cross-validation folds to use
            
        Returns:
            Tuple of (model, predictions, metrics, coefficients, cv_results)
        """
        model = LinearRegression()
        model.fit(self.X_train_scaled.values, self.y_train.values)
        
        results = evaluate_model(
            model, self.X_train_scaled, self.X_test_scaled,
            self.y_train, self.y_test, "Linear"
        )
        
        if visualize:
            save_pred_path = f"{save_dir}/linear_predictions.png" if save_dir else None
            save_feat_path = f"{save_dir}/linear_feature_importance.png" if save_dir else None
            
            plot_predictions(self.y_test, results[1], "Linear", save_pred_path)
            plot_feature_importance(results[3], "Linear", save_path=save_feat_path)
        
        # Add cross-validation if requested
        cv_results = None
        if cv is not None:
            # Scale all data
            X_scaled = self.scaler.fit_transform(self.X)
            # Perform cross-validation
            cv_results = cv_evaluate(model, X_scaled, self.Y.values, cv=cv)
        
        # Return results with optional CV
        if cv is not None:
            return *results, cv_results
        else:
            return results
    
    def fit_ridge(self, alphas=np.logspace(-5, 5, 200), visualize=True, save_dir=None, cv=None):
        """Fit Ridge regression model with cross-validation
        
        Args:
            alphas: Array of alpha values for cross-validation
            visualize: Whether to create visualizations
            save_dir: Directory to save visualizations, if None, displays plots
            cv: If specified, number of cross-validation folds to use
            
        Returns:
            Tuple of (best_alpha, model, predictions, metrics, coefficients, cv_results)
        """
        ridge_cv = RidgeCV(alphas=alphas, cv=5)
        ridge_cv.fit(self.X_train_scaled.values, self.y_train.values)
        
        model = Ridge(alpha=ridge_cv.alpha_)
        model.fit(self.X_train_scaled.values, self.y_train.values)
        
        results = evaluate_model(
            model, self.X_train_scaled, self.X_test_scaled,
            self.y_train, self.y_test, "Ridge"
        )
        
        if visualize:
            save_pred_path = f"{save_dir}/ridge_predictions.png" if save_dir else None
            save_feat_path = f"{save_dir}/ridge_feature_importance.png" if save_dir else None
            
            plot_predictions(self.y_test, results[1], "Ridge", save_pred_path)
            plot_feature_importance(results[3], "Ridge", save_path=save_feat_path)
        
        # Add cross-validation if requested
        cv_results = None
        if cv is not None:
            # Scale all data
            X_scaled = self.scaler.fit_transform(self.X)
            # Create model with best alpha
            best_model = Ridge(alpha=ridge_cv.alpha_)
            # Perform cross-validation
            cv_results = cv_evaluate(best_model, X_scaled, self.Y.values, cv=cv)
        
        # Return results with optional CV
        if cv is not None:
            return ridge_cv.alpha_, *results, cv_results
        else:
            return ridge_cv.alpha_, *results
    
    def fit_lasso(self, alphas=np.logspace(-5, 5, 200), visualize=True, save_dir=None, cv=None):
        """Fit Lasso regression model with cross-validation
        
        Args:
            alphas: Array of alpha values for cross-validation
            visualize: Whether to create visualizations
            save_dir: Directory to save visualizations, if None, displays plots
            cv: If specified, number of cross-validation folds to use
            
        Returns:
            Tuple of (best_alpha, model, predictions, metrics, coefficients, cv_results)
        """
        lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)
        lasso_cv.fit(self.X_train_scaled.values, self.y_train.values)
        
        model = Lasso(alpha=lasso_cv.alpha_)
        model.fit(self.X_train_scaled.values, self.y_train.values)
        
        results = evaluate_model(
            model, self.X_train_scaled, self.X_test_scaled,
            self.y_train, self.y_test, "Lasso"
        )
        
        if visualize:
            save_pred_path = f"{save_dir}/lasso_predictions.png" if save_dir else None
            save_feat_path = f"{save_dir}/lasso_feature_importance.png" if save_dir else None
            
            plot_predictions(self.y_test, results[1], "Lasso", save_pred_path)
            plot_feature_importance(results[3], "Lasso", save_path=save_feat_path)
        
        # Add cross-validation if requested
        cv_results = None
        if cv is not None:
            # Scale all data
            X_scaled = self.scaler.fit_transform(self.X)
            # Create model with best alpha
            best_model = Lasso(alpha=lasso_cv.alpha_)
            # Perform cross-validation
            cv_results = cv_evaluate(best_model, X_scaled, self.Y.values, cv=cv)
        
        # Return results with optional CV
        if cv is not None:
            return lasso_cv.alpha_, *results, cv_results
        else:
            return lasso_cv.alpha_, *results
        
    def fit_elastic_net(self, alphas=np.logspace(-5, 5, 100), l1_ratios=np.linspace(0.1, 0.9, 9), 
                         visualize=True, save_dir=None, cv=None):
        """Fit ElasticNet regression model with cross-validation
        
        Args:
            alphas: Array of alpha values for cross-validation
            l1_ratios: Array of l1_ratio values for cross-validation
            visualize: Whether to create visualizations
            save_dir: Directory to save visualizations, if None, displays plots
            cv: If specified, number of cross-validation folds to use
            
        Returns:
            Tuple of (best_alpha, best_l1_ratio, model, predictions, metrics, coefficients, cv_results)
        """
        elastic_cv = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=5, random_state=42)
        elastic_cv.fit(self.X_train_scaled.values, self.y_train.values)
        
        model = ElasticNet(alpha=elastic_cv.alpha_, l1_ratio=elastic_cv.l1_ratio_)
        model.fit(self.X_train_scaled.values, self.y_train.values)
        
        results = evaluate_model(
            model, self.X_train_scaled, self.X_test_scaled,
            self.y_train, self.y_test, "ElasticNet"
        )
        
        if visualize:
            save_pred_path = f"{save_dir}/elastic_net_predictions.png" if save_dir else None
            save_feat_path = f"{save_dir}/elastic_net_feature_importance.png" if save_dir else None
            
            plot_predictions(self.y_test, results[1], "ElasticNet", save_pred_path)
            plot_feature_importance(results[3], "ElasticNet", save_path=save_feat_path)
        
        # Add cross-validation if requested
        cv_results = None
        if cv is not None:
            # Scale all data
            X_scaled = self.scaler.fit_transform(self.X)
            # Create model with best parameters
            best_model = ElasticNet(alpha=elastic_cv.alpha_, l1_ratio=elastic_cv.l1_ratio_)
            # Perform cross-validation
            cv_results = cv_evaluate(best_model, X_scaled, self.Y.values, cv=cv)
        
        # Return results with optional CV
        if cv is not None:
            return elastic_cv.alpha_, elastic_cv.l1_ratio_, *results, cv_results
        else:
            return elastic_cv.alpha_, elastic_cv.l1_ratio_, *results
        
    def cross_validate_models(self, cv=5, visualize=True, save_dir=None):
        """
        Perform cross-validation on all regression models
        
        Args:
            cv: Number of cross-validation folds
            visualize: Whether to create visualizations
            save_dir: Directory to save visualizations
            
        Returns:
            Dictionary with cross-validation results for each model
        """
        # Scale all data for CV
        X_scaled = self.scaler.fit_transform(self.X)
        y = self.Y.values
        
        # Initialize models
        linear_model = LinearRegression()
        cv_results = {'Linear': cv_evaluate(linear_model, X_scaled, y, cv=cv)}
        
        # Ridge with CV
        ridge_cv = RidgeCV(alphas=np.logspace(-5, 5, 200), cv=cv)
        ridge_cv.fit(X_scaled, y)
        best_ridge = Ridge(alpha=ridge_cv.alpha_)
        ridge_results = cv_evaluate(best_ridge, X_scaled, y, cv=cv)
        ridge_results['best_alpha'] = ridge_cv.alpha_
        cv_results['Ridge'] = ridge_results
        
        # Lasso with CV
        lasso_cv = LassoCV(alphas=np.logspace(-5, 5, 200), cv=cv, random_state=42)
        lasso_cv.fit(X_scaled, y)
        best_lasso = Lasso(alpha=lasso_cv.alpha_)
        lasso_results = cv_evaluate(best_lasso, X_scaled, y, cv=cv)
        lasso_results['best_alpha'] = lasso_cv.alpha_
        cv_results['Lasso'] = lasso_results
        
        # ElasticNet with CV
        elastic_cv = ElasticNetCV(
            alphas=np.logspace(-5, 5, 100),
            l1_ratio=np.linspace(0.1, 0.9, 9),
            cv=cv,
            random_state=42
        )
        elastic_cv.fit(X_scaled, y)
        best_elastic = ElasticNet(alpha=elastic_cv.alpha_, l1_ratio=elastic_cv.l1_ratio_)
        elastic_results = cv_evaluate(best_elastic, X_scaled, y, cv=cv)
        elastic_results['best_alpha'] = elastic_cv.alpha_
        elastic_results['best_l1_ratio'] = elastic_cv.l1_ratio_
        cv_results['ElasticNet'] = elastic_results
        
        # Visualize CV results if requested
        if visualize:
            save_path = f"{save_dir}/cv_results_comparison.png" if save_dir else None
            plot_cv_results(cv_results, save_path)
        
        return cv_results
        
    def compare_models(self, save_dir=None, include_elastic_net=True):
        """Compare all regression models and visualize results
        
        Args:
            save_dir: Directory to save comparison visualizations
            include_elastic_net: Whether to include ElasticNet in the comparison
            
        Returns:
            Dictionary with results from all models
        """
        # Fit all models
        print("Fitting Linear Regression...")
        linear_results = self.fit_linear(visualize=False)
        
        print("Fitting Ridge Regression...")
        ridge_alpha, *ridge_results = self.fit_ridge(visualize=False)
        
        print("Fitting Lasso Regression...")
        lasso_alpha, *lasso_results = self.fit_lasso(visualize=False)
        
        # Initialize models dictionary
        models = {
            'Linear': {'results': linear_results, 'alpha': None},
            'Ridge': {'results': ridge_results, 'alpha': ridge_alpha},
            'Lasso': {'results': lasso_results, 'alpha': lasso_alpha}
        }
        
        # Add ElasticNet if requested
        if include_elastic_net:
            print("Fitting ElasticNet Regression...")
            elastic_alpha, elastic_l1, *elastic_results = self.fit_elastic_net(visualize=False)
            models['ElasticNet'] = {
                'results': elastic_results,
                'alpha': elastic_alpha,
                'l1_ratio': elastic_l1
            }
        
        # Compare metrics in a bar chart
        metrics_to_compare = ['test_r2', 'adj_r2', 'rmse', 'mae', 
                             'accuracy_within_20%', 'accuracy_within_40%']
        
        metrics_data = []
        for model_name, model_info in models.items():
            model_metrics = model_info['results'][2]  # metrics is at index 2
            for metric in metrics_to_compare:
                metrics_data.append({
                    'Model': model_name,
                    'Metric': metric,
                    'Value': model_metrics[metric]
                })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Separate plots for different metric types
        for metric_group, ylabel in [
            (['test_r2', 'adj_r2'], 'R² Value'),
            (['rmse', 'mae'], 'Error Value'),
            (['accuracy_within_20%', 'accuracy_within_40%'], 'Accuracy (%)')
        ]:
            plt.figure(figsize=(10, 6))
            
            df_subset = df_metrics[df_metrics['Metric'].isin(metric_group)]
            ax = sns.barplot(x='Model', y='Value', hue='Metric', data=df_subset)
            
            plt.title(f"Model Comparison: {', '.join(metric_group)}", fontsize=14)
            plt.xlabel('Model', fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', fontsize=9)
            
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(f"{save_dir}/comparison_{metric_group[0]}.png", dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        
        # Create a combined plot of actual vs predicted for all models
        n_models = len(models)
        fig_width = min(15, 5 * n_models)
        plt.figure(figsize=(fig_width, 6))
        
        for i, (model_name, model_info) in enumerate(models.items(), 1):
            plt.subplot(1, n_models, i)
            y_pred = model_info['results'][1]  # predictions at index 1
            
            # Calculate accuracy metrics
            accuracy_20 = np.mean(np.abs(self.y_test - y_pred) / np.abs(self.y_test) <= 0.2) * 100
            accuracy_40 = np.mean(np.abs(self.y_test - y_pred) / np.abs(self.y_test) <= 0.4) * 100
            
            # Create scatter plot
            plt.scatter(self.y_test, y_pred, alpha=0.6)
            
            # Add perfect prediction line
            min_val = min(self.y_test.min(), y_pred.min())
            max_val = max(self.y_test.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # Customize subplot
            alpha_text = f", α={model_info['alpha']:.4f}" if model_info['alpha'] else ""
            if model_name == "ElasticNet" and 'l1_ratio' in model_info:
                alpha_text += f", l1_ratio={model_info['l1_ratio']:.2f}"
                
            plt.title(f"{model_name}{alpha_text}", fontsize=12)
            plt.xlabel("Actual", fontsize=10)
            plt.ylabel("Predicted", fontsize=10)
            
            # Add accuracy text
            text_info = (
                f"±20%: {accuracy_20:.1f}%\n"
                f"±40%: {accuracy_40:.1f}%"
            )
            plt.annotate(text_info, xy=(0.05, 0.95), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                         fontsize=9, ha='left', va='top')
            
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(f"{save_dir}/all_models_predictions.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return models