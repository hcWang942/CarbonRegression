import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis, shapiro, jarque_bera
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Set up paths using parent and root approach
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'src', 'preprocessing', 'distribution_analysis_results')

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load the data
file_path = os.path.join(DATA_DIR, 'scope1_cleaned.xlsx')
df = pd.read_excel(file_path)

# Display basic info about the dataset
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Variables with correlation >= 0.3 with Scope1 (specific 6 variables for analysis)
# Note: Excluding Scope1 itself as it's the dependent variable
key_variables = [
    'TA',          # 0.4698
    'Inflation',   # -0.4618
    'TL',          # 0.4495
    'PPE',         # 0.4087
    'SOxEmission', # 0.4073
    'ROE'          # (correlation value to be determined)
]

# Check which variables actually exist in the dataset
available_vars = [var for var in key_variables if var in df.columns]
print(f"\nAvailable variables for analysis: {available_vars}")

# Check if ROE exists with different possible names
possible_roe_names = ['ROE', 'roe', 'ROE_ratio', 'ReturnOnEquity', 'Return_on_Equity']
roe_found = None
for roe_name in possible_roe_names:
    if roe_name in df.columns:
        roe_found = roe_name
        break

if roe_found and roe_found not in available_vars:
    available_vars.append(roe_found)
    print(f"Found ROE as: {roe_found}")

print(f"\nFinal variables for analysis: {available_vars}")
print(f"Number of variables: {len(available_vars)}")

# If some variables don't exist, let's see what's available
if len(available_vars) < len(key_variables):
    print("\nAll available columns:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")

# Function to calculate distribution statistics
def calculate_distribution_stats(data, var_name):
    """Calculate skewness, kurtosis, and normality tests for a variable"""
    # Remove any NaN values
    clean_data = data.dropna()
    
    if len(clean_data) == 0:
        return None
    
    # Calculate statistics
    skewness = skew(clean_data)
    kurt = kurtosis(clean_data)  # Excess kurtosis (normal=0)
    
    # Normality tests
    if len(clean_data) >= 3:
        shapiro_stat, shapiro_p = shapiro(clean_data) if len(clean_data) <= 5000 else (np.nan, np.nan)
        jb_stat, jb_p = jarque_bera(clean_data)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan
        jb_stat, jb_p = np.nan, np.nan
    
    return {
        'Variable': var_name,
        'N': len(clean_data),
        'Mean': np.mean(clean_data),
        'Std': np.std(clean_data),
        'Skewness': skewness,
        'Kurtosis': kurt,
        'Shapiro_Stat': shapiro_stat,
        'Shapiro_p': shapiro_p,
        'JB_Stat': jb_stat,
        'JB_p': jb_p
    }

# Create distribution statistics table
stats_results = []
for var in available_vars:
    if var in df.columns:
        result = calculate_distribution_stats(df[var], var)
        if result:
            stats_results.append(result)

# Convert to DataFrame
stats_df = pd.DataFrame(stats_results)
print("\n" + "="*80)
print("DISTRIBUTION STATISTICS TABLE")
print("="*80)
print(stats_df.round(4))

# Create the main visualization - flexible grid based on available variables
n_vars = len(available_vars)
if n_vars <= 6:
    n_cols = 3
    n_rows = 2
elif n_vars <= 9:
    n_cols = 3
    n_rows = 3
else:
    n_cols = 4
    n_rows = (n_vars + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
fig.suptitle('Distribution Analysis of Independent Variables with High Correlation Showing Deviations from Normality', 
             fontsize=14, fontweight='bold')

# Handle single row case
if n_rows == 1:
    axes = axes.reshape(1, -1)
axes = axes.flatten()

for i, var in enumerate(available_vars):
    if i >= len(axes):
        break
        
    ax = axes[i]
    data = df[var].dropna()
    
    if len(data) == 0:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(var)
        continue
    
    # Create histogram
    n_bins = min(30, int(np.sqrt(len(data))))
    counts, bins, patches = ax.hist(data, bins=n_bins, density=True, alpha=0.7, 
                                  color='skyblue', edgecolor='black', linewidth=0.5)
    
    # Overlay normal distribution
    mu, sigma = np.mean(data), np.std(data)
    x = np.linspace(data.min(), data.max(), 100)
    normal_curve = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, normal_curve, 'r--', linewidth=2, label='Normal Distribution')
    
    # Calculate and display statistics
    skewness = skew(data)
    kurt = kurtosis(data)
    
    # Add statistics text box with correlation info
    if var == 'TA':
        stats_text = f'Skew: {skewness:.2f}\nKurt: {kurt:.2f}\nr = 0.470'
    elif var == 'Inflation':
        stats_text = f'Skew: {skewness:.2f}\nKurt: {kurt:.2f}\nr = -0.462'
    elif var == 'TL':
        stats_text = f'Skew: {skewness:.2f}\nKurt: {kurt:.2f}\nr = 0.449'
    elif var == 'PPE':
        stats_text = f'Skew: {skewness:.2f}\nKurt: {kurt:.2f}\nr = 0.409'
    elif var == 'SOxEmission':
        stats_text = f'Skew: {skewness:.2f}\nKurt: {kurt:.2f}\nr = 0.407'
    elif var in ['ROE', 'roe', 'ROE_ratio', 'ReturnOnEquity', 'Return_on_Equity']:
        stats_text = f'Skew: {skewness:.2f}\nKurt: {kurt:.2f}\nr = -0.293'
    else:
        stats_text = f'Skew: {skewness:.2f}\nKurt: {kurt:.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            fontsize=9)
    
    # Full variable name as title
    ax.set_title(f'{var}', fontweight='bold', fontsize=11)
    ax.set_xlabel('Value', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    if i == 0:  # Only show legend for first subplot
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

# Hide any unused subplots
for j in range(len(available_vars), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()

# Save the figure
fig_path = os.path.join(RESULTS_DIR, 'distribution_analysis.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {fig_path}")
plt.show()

# Create supplementary table for the paper
print("\n" + "="*100)
print("SUPPLEMENTARY TABLE FOR PAPER")
print("="*100)
print("Table X. Descriptive Statistics and Normality Tests for Selected High-Correlation Variables")
print("-"*100)

# Format the table for paper
paper_table = stats_df[['Variable', 'N', 'Mean', 'Std', 'Skewness', 'Kurtosis', 
                       'Shapiro_p', 'JB_p']].copy()
paper_table['Shapiro_p'] = paper_table['Shapiro_p'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
paper_table['JB_p'] = paper_table['JB_p'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")

for col in ['Mean', 'Std', 'Skewness', 'Kurtosis']:
    paper_table[col] = paper_table[col].round(4)

print(paper_table.to_string(index=False))
table_path = os.path.join(RESULTS_DIR, 'distribution_statistics_table.csv')
paper_table.to_csv(table_path, index=False)
print(f"\nTable saved to: {table_path}")

print("\nNote: Analysis focuses on independent variables with |correlation| ≥ 0.4 with Scope1 emissions.")
print("Shapiro_p and JB_p are p-values from Shapiro-Wilk and Jarque-Bera normality tests respectively.")
print("P-values < 0.05 indicate significant deviation from normal distribution.")

# If ROE is missing, print a note
if 'ROE' not in available_vars and roe_found is None:
    print("\nNote: ROE variable not found in dataset. Please check variable name in the dataset.")

# Interpretation summary
print("\n" + "="*80)
print("INTERPRETATION SUMMARY")
print("="*80)

interpretation_results = []
for _, row in stats_df.iterrows():
    var_name = row['Variable']
    skewness = row['Skewness']
    kurt = row['Kurtosis']
    shapiro_p = row['Shapiro_p']
    
    print(f"\n{var_name}:")
    
    # Skewness interpretation
    if abs(skewness) < 0.5:
        skew_interp = "approximately symmetric"
    elif skewness > 0.5:
        skew_interp = "positively skewed (right tail)"
    else:
        skew_interp = "negatively skewed (left tail)"
    
    # Kurtosis interpretation
    if abs(kurt) < 0.5:
        kurt_interp = "normal peakedness"
    elif kurt > 0.5:
        kurt_interp = "heavy-tailed (leptokurtic)"
    else:
        kurt_interp = "light-tailed (platykurtic)"
    
    # Normality interpretation
    if pd.notna(shapiro_p):
        if shapiro_p < 0.05:
            norm_interp = "significantly deviates from normal distribution"
        else:
            norm_interp = "does not significantly deviate from normal distribution"
    else:
        norm_interp = "normality test unavailable"
    
    print(f"  - Distribution: {skew_interp}, {kurt_interp}")
    print(f"  - Normality: {norm_interp}")
    if pd.notna(shapiro_p):
        print(f"  - Shapiro-Wilk p-value: {shapiro_p:.4f}")
    
    interpretation_results.append({
        'Variable': var_name,
        'Skewness_Interpretation': skew_interp,
        'Kurtosis_Interpretation': kurt_interp,
        'Normality_Interpretation': norm_interp
    })

# Save interpretation summary
interpretation_df = pd.DataFrame(interpretation_results)
interpretation_path = os.path.join(RESULTS_DIR, 'distribution_interpretation.csv')
interpretation_df.to_csv(interpretation_path, index=False)
print(f"\nInterpretation summary saved to: {interpretation_path}")

print(f"\nAll results saved to: {RESULTS_DIR}")