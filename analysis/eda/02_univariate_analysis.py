"""
Auto-MPG Dataset: Univariate Analysis and Target Variable Profiling
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Paths
OUTPUT_DIR = Path("/workspace/analysis/eda")

# Load cleaned data
df = pd.read_csv(OUTPUT_DIR / 'auto_mpg_cleaned.csv')

print("="*60)
print("UNIVARIATE ANALYSIS")
print("="*60)

# Numeric columns (excluding car_name)
numeric_cols = ['mpg', 'cylinders', 'displacement', 'horsepower',
                'weight', 'acceleration', 'model_year', 'origin']

# Compute descriptive statistics
print("\n--- Descriptive Statistics ---")
desc_stats = df[numeric_cols].describe(percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .99])
print(desc_stats.round(2))

# Additional statistics
print("\n--- Additional Statistics ---")
additional_stats = []
for col in numeric_cols:
    data = df[col].dropna()
    row = {
        'column': col,
        'mean': data.mean(),
        'std': data.std(),
        'median': data.median(),
        'iqr': data.quantile(0.75) - data.quantile(0.25),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data),
        'cv': data.std() / data.mean() if data.mean() != 0 else np.nan,
        'min': data.min(),
        'max': data.max(),
        'range': data.max() - data.min(),
        'n_valid': len(data)
    }
    additional_stats.append(row)

additional_df = pd.DataFrame(additional_stats)
print(additional_df.round(3))

# Save univariate summary
univariate_summary = additional_df.copy()
univariate_summary['inferred_type'] = ['continuous_bounded_positive', 'discrete_categorical',
                                        'continuous_positive', 'continuous_positive',
                                        'continuous_positive', 'continuous_positive',
                                        'discrete_ordinal', 'discrete_categorical']
univariate_summary.to_csv(OUTPUT_DIR / 'univariate_summary.csv', index=False)
print(f"\nSaved univariate summary to {OUTPUT_DIR / 'univariate_summary.csv'}")

# TARGET VARIABLE ANALYSIS (MPG)
print("\n" + "="*60)
print("TARGET VARIABLE ANALYSIS: MPG")
print("="*60)

mpg = df['mpg']
print(f"\nN: {len(mpg)}")
print(f"Mean: {mpg.mean():.2f}")
print(f"Median: {mpg.median():.2f}")
print(f"Std: {mpg.std():.2f}")
print(f"Range: [{mpg.min():.1f}, {mpg.max():.1f}]")
print(f"IQR: {mpg.quantile(0.75) - mpg.quantile(0.25):.2f}")
print(f"Skewness: {stats.skew(mpg):.3f}")
print(f"Kurtosis: {stats.kurtosis(mpg):.3f}")

# Normality tests
stat_shapiro, p_shapiro = stats.shapiro(mpg)
stat_dagostino, p_dagostino = stats.normaltest(mpg)
print(f"\nNormality tests:")
print(f"  Shapiro-Wilk: W={stat_shapiro:.4f}, p={p_shapiro:.4e}")
print(f"  D'Agostino-Pearson: stat={stat_dagostino:.4f}, p={p_dagostino:.4e}")

# Check log-transformed MPG
log_mpg = np.log(mpg)
stat_shapiro_log, p_shapiro_log = stats.shapiro(log_mpg)
print(f"\nLog-transformed MPG:")
print(f"  Skewness: {stats.skew(log_mpg):.3f}")
print(f"  Shapiro-Wilk: W={stat_shapiro_log:.4f}, p={p_shapiro_log:.4e}")

# Create distribution plots
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# MPG histogram
ax = axes[0, 0]
ax.hist(mpg, bins=30, edgecolor='black', alpha=0.7, density=True)
x_range = np.linspace(mpg.min(), mpg.max(), 100)
ax.plot(x_range, stats.norm.pdf(x_range, mpg.mean(), mpg.std()), 'r-', lw=2, label='Normal fit')
ax.set_xlabel('MPG')
ax.set_ylabel('Density')
ax.set_title('MPG Distribution')
ax.legend()

# Log MPG histogram
ax = axes[0, 1]
ax.hist(log_mpg, bins=30, edgecolor='black', alpha=0.7, density=True)
x_range = np.linspace(log_mpg.min(), log_mpg.max(), 100)
ax.plot(x_range, stats.norm.pdf(x_range, log_mpg.mean(), log_mpg.std()), 'r-', lw=2, label='Normal fit')
ax.set_xlabel('log(MPG)')
ax.set_ylabel('Density')
ax.set_title('Log-MPG Distribution')
ax.legend()

# Q-Q plot for MPG
ax = axes[0, 2]
stats.probplot(mpg, dist="norm", plot=ax)
ax.set_title('Q-Q Plot: MPG')

# Q-Q plot for log MPG
ax = axes[1, 0]
stats.probplot(log_mpg, dist="norm", plot=ax)
ax.set_title('Q-Q Plot: log(MPG)')

# Box plot
ax = axes[1, 1]
ax.boxplot(mpg, vert=True)
ax.set_ylabel('MPG')
ax.set_title('MPG Boxplot')

# Empirical CDF
ax = axes[1, 2]
sorted_mpg = np.sort(mpg)
ecdf = np.arange(1, len(sorted_mpg) + 1) / len(sorted_mpg)
ax.plot(sorted_mpg, ecdf, 'b-', lw=2)
ax.set_xlabel('MPG')
ax.set_ylabel('Cumulative Probability')
ax.set_title('Empirical CDF')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'mpg_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved MPG distribution plots to {OUTPUT_DIR / 'mpg_distribution.png'}")

# Check for boundary effects
print(f"\nBoundary analysis:")
print(f"  Values at lower bound (mpg < 12): {(mpg < 12).sum()}")
print(f"  Values at upper bound (mpg > 40): {(mpg > 40).sum()}")
print(f"  These might indicate truncation or ceiling effects")

# Potential outliers (IQR method)
Q1, Q3 = mpg.quantile([0.25, 0.75])
IQR = Q3 - Q1
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR
outliers = mpg[(mpg < lower_fence) | (mpg > upper_fence)]
print(f"\nOutliers (1.5*IQR method):")
print(f"  Lower fence: {lower_fence:.2f}")
print(f"  Upper fence: {upper_fence:.2f}")
print(f"  N outliers: {len(outliers)}")
if len(outliers) > 0:
    print(f"  Outlier values: {sorted(outliers.values)}")
