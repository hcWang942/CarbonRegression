# run_diagnostics.py
import pandas as pd
from regression_diagnostics import RegressionDiagnostics
from data_transform import transform_data
import os
from pathlib import Path

# 设置路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'src', 'preprocessing', 'results')

def main():
    # 加载数据
    file_path = os.path.join(DATA_DIR, 'delete1_selected.xlsx')
    original_df = pd.read_excel(file_path)
    
    # 数据预处理
    df_transformed = transform_data(original_df)
    
    # 运行回归诊断
    diagnostics = RegressionDiagnostics(df_transformed, output_dir=RESULTS_DIR)
    results = diagnostics.run_all_diagnostics(
        save_plots=True,
        save_results=True,
        output_dir=RESULTS_DIR
    )
    
    # 打印结果摘要
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

if __name__ == "__main__":
    main()