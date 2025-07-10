import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge, LassoCV, Lasso
from sklearn.metrics import mean_squared_error, r2_score

def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def calculate_accuracy_thresholds(y_true, y_pred, thresholds=[0.2, 0.4]):
    accuracy_dict = {}
    for threshold in thresholds:
        rel_error = np.abs(y_true - y_pred) / np.abs(y_true)
        within_threshold = np.mean(rel_error <= threshold) * 100
        accuracy_dict[f'accuracy_within_{int(threshold*100)}%'] = within_threshold
    
    return accuracy_dict

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
    
    accuracy_metrics = calculate_accuracy_thresholds(y_test_array, y_test_pred)
    metrics.update(accuracy_metrics)
    
    coefficients = {
        'intercept': model.intercept_,
        'coefficients': dict(zip(X_train.columns, model.coef_))
    }
    
    return model, y_test_pred, metrics, coefficients

def plot_predictions(y_true, y_pred, model_name, save_path=None):
    plt.figure(figsize=(10, 8))
    
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
    
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.plot([min_val, max_val], [min_val*0.8, max_val*0.8], 'g:', label='-20% Error Bound')
    plt.plot([min_val, max_val], [min_val*1.2, max_val*1.2], 'g:', label='+20% Error Bound')
    
    plt.plot([min_val, max_val], [min_val*0.6, max_val*0.6], 'y:', label='-40% Error Bound')
    plt.plot([min_val, max_val], [min_val*1.4, max_val*1.4], 'y:', label='+40% Error Bound')
    
    accuracy_20 = np.mean(np.abs(y_true - y_pred) / np.abs(y_true) <= 0.2) * 100
    accuracy_40 = np.mean(np.abs(y_true - y_pred) / np.abs(y_true) <= 0.4) * 100
    
    plt.title(f"{model_name} Regression: Actual vs Predicted Values", fontsize=14)
    plt.xlabel("Actual Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    text_info = (
        f"Accuracy (within ±20%): {accuracy_20:.1f}%\n"
        f"Accuracy (within ±40%): {accuracy_40:.1f}%"
    )
    
    plt.annotate(text_info, xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                 fontsize=10, ha='left', va='top')
    
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
    coefs = coefficients['coefficients']
    feature_names = list(coefs.keys())
    importance = np.abs(list(coefs.values()))
    
    df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    df_importance = df_importance.sort_values('Importance', ascending=False)
    
    if top_n is not None and top_n < len(df_importance):
        df_importance = df_importance.iloc[:top_n]
    
    plt.figure(figsize=(10, max(6, len(df_importance) * 0.3)))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(x='Importance', y='Feature', data=df_importance, palette='viridis')
    
    plt.title(f"Feature Importance - {model_name} Regression", fontsize=14)
    plt.xlabel('Absolute Coefficient Value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    for i, importance in enumerate(df_importance['Importance']):
        ax.text(importance + 0.01, i, f"{importance:.4f}", va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

class RegressionModels:
    def __init__(self, data, target_col, test_size=0.4, random_state=42):
        self.Y = data[target_col]
        self.X = data.drop(columns=[target_col])
        self.target_col = target_col
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
    
    def fit_linear(self, visualize=True, save_dir=None):
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
        
        return results
    
    def fit_ridge(self, alphas=np.logspace(-3, 3, 100), visualize=True, save_dir=None):
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
        
        return ridge_cv.alpha_, *results
    
    def fit_lasso(self, alphas=np.logspace(-3, 3, 100), visualize=True, save_dir=None):
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
        
        return lasso_cv.alpha_, *results
        
    def compare_models(self, save_dir=None):
        linear_results = self.fit_linear(visualize=False)
        ridge_alpha, *ridge_results = self.fit_ridge(visualize=False)
        lasso_alpha, *lasso_results = self.fit_lasso(visualize=False)
        
        models = {
            'Linear': {'results': linear_results, 'alpha': None},
            'Ridge': {'results': ridge_results, 'alpha': ridge_alpha},
            'Lasso': {'results': lasso_results, 'alpha': lasso_alpha}
        }
        
        metrics_to_compare = ['test_r2', 'adj_r2', 'rmse', 'mae', 
                             'accuracy_within_20%', 'accuracy_within_40%']
        
        metrics_data = []
        for model_name, model_info in models.items():
            model_metrics = model_info['results'][2]
            for metric in metrics_to_compare:
                metrics_data.append({
                    'Model': model_name,
                    'Metric': metric,
                    'Value': model_metrics[metric]
                })
        
        df_metrics = pd.DataFrame(metrics_data)
        
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
            
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', fontsize=9)
            
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(f"{save_dir}/comparison_{metric_group[0]}.png", dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        
        plt.figure(figsize=(15, 6))
        
        for i, (model_name, model_info) in enumerate(models.items(), 1):
            plt.subplot(1, 3, i)
            y_pred = model_info['results'][1]
            
            accuracy_20 = np.mean(np.abs(self.y_test - y_pred) / np.abs(self.y_test) <= 0.2) * 100
            accuracy_40 = np.mean(np.abs(self.y_test - y_pred) / np.abs(self.y_test) <= 0.4) * 100
            
            plt.scatter(self.y_test, y_pred, alpha=0.6)
            
            min_val = min(self.y_test.min(), y_pred.min())
            max_val = max(self.y_test.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            alpha_text = f", α={model_info['alpha']:.4f}" if model_info['alpha'] else ""
            plt.title(f"{model_name}{alpha_text}", fontsize=12)
            plt.xlabel("Actual", fontsize=10)
            plt.ylabel("Predicted", fontsize=10)
            
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