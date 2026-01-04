"""
Auto-MPG Dataset: Missing Data Analysis and Outliers
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

OUTPUT_DIR = Path("/workspace/analysis/eda")
df = pd.read_csv(OUTPUT_DIR / 'auto_mpg_cleaned.csv')

print("="*60)
print("MISSING DATA ANALYSIS")
print("="*60)

# Identify rows with missing horsepower
missing_hp = df[df['horsepower'].isnull()]
print(f"\nRows with missing horsepower: {len(missing_hp)}")
print("\nMissing horsepower cases:")
print(missing_hp[['car_name', 'mpg', 'cylinders', 'displacement', 'weight', 'model_year', 'origin']])

# Check if missingness is related to other variables
print("\n--- Missingness Mechanism Analysis ---")
df['hp_missing'] = df['horsepower'].isnull().astype(int)

# Compare distributions of complete vs missing
print("\nComparing complete vs missing groups:")
for col in ['mpg', 'cylinders', 'displacement', 'weight', 'acceleration', 'model_year', 'origin']:
    complete = df[df['hp_missing'] == 0][col]
    missing = df[df['hp_missing'] == 1][col]

    # t-test or mann-whitney
    if complete.std() > 0 and missing.std() > 0:
        stat, p = stats.mannwhitneyu(complete, missing)
        sig = "*" if p < 0.05 else ""
        print(f"  {col:15s}: Complete mean={complete.mean():.2f}, Missing mean={missing.mean():.2f}, "
              f"MW p={p:.3f}{sig}")
    else:
        print(f"  {col:15s}: Complete mean={complete.mean():.2f}, Missing mean={missing.mean():.2f}")

# Year distribution of missing
print("\nMissing by year:")
missing_by_year = df[df['hp_missing'] == 1].groupby('model_year').size()
print(missing_by_year)

# Origin distribution of missing
print("\nMissing by origin:")
missing_by_origin = df[df['hp_missing'] == 1].groupby('origin').size()
origin_map = {1: 'USA', 2: 'Europe', 3: 'Japan'}
for o, count in missing_by_origin.items():
    print(f"  {origin_map[o]}: {count}")

print("\n" + "="*60)
print("OUTLIER ANALYSIS")
print("="*60)

# IQR-based outliers for each numeric column
numeric_cols = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']
print("\n--- IQR-based Outliers (1.5*IQR) ---")

for col in numeric_cols:
    data = df[col].dropna()
    Q1, Q3 = data.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data < lower) | (data > upper)]
    print(f"  {col}: {len(outliers)} outliers (lower fence={lower:.1f}, upper fence={upper:.1f})")

# Z-score based outliers
print("\n--- Z-score Outliers (|z| > 3) ---")
for col in numeric_cols:
    data = df[col].dropna()
    z_scores = np.abs(stats.zscore(data))
    n_outliers = (z_scores > 3).sum()
    if n_outliers > 0:
        outlier_vals = data[z_scores > 3].values
        print(f"  {col}: {n_outliers} outliers, values: {outlier_vals}")
    else:
        print(f"  {col}: 0 outliers")

# Multivariate outliers using Mahalanobis distance
print("\n--- Multivariate Outliers (Mahalanobis distance) ---")
cols_for_mahal = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']
df_complete = df[cols_for_mahal].dropna()

from scipy.spatial.distance import mahalanobis
mean = df_complete.mean().values
cov = df_complete.cov().values

try:
    cov_inv = np.linalg.inv(cov)
    mahal_distances = df_complete.apply(lambda row: mahalanobis(row, mean, cov_inv), axis=1)

    # Chi-squared threshold (5 variables, alpha=0.001)
    threshold = stats.chi2.ppf(0.999, df=5)
    n_outliers = (mahal_distances > threshold).sum()
    print(f"  Threshold (chi2, df=5, alpha=0.001): {threshold:.2f}")
    print(f"  Number of multivariate outliers: {n_outliers}")

    if n_outliers > 0 and n_outliers <= 10:
        outlier_idx = mahal_distances[mahal_distances > threshold].index
        print("\n  Multivariate outlier cases:")
        print(df.loc[outlier_idx, ['car_name', 'mpg', 'cylinders', 'displacement', 'horsepower', 'weight']])
except:
    print("  Could not compute Mahalanobis distances (singular covariance)")

# Unusual observations
print("\n--- Unusual Observations ---")
# High MPG for V8
high_mpg_v8 = df[(df['cylinders'] == 8) & (df['mpg'] > 20)]
print(f"\nHigh MPG (>20) for 8-cylinder cars: {len(high_mpg_v8)}")
if len(high_mpg_v8) > 0:
    print(high_mpg_v8[['car_name', 'mpg', 'horsepower', 'weight', 'model_year']])

# Low MPG for 4-cylinder
low_mpg_4cyl = df[(df['cylinders'] == 4) & (df['mpg'] < 20)]
print(f"\nLow MPG (<20) for 4-cylinder cars: {len(low_mpg_4cyl)}")

# Heavy 4-cylinder cars
heavy_4cyl = df[(df['cylinders'] == 4) & (df['weight'] > 3000)]
print(f"\nHeavy (>3000 lbs) 4-cylinder cars: {len(heavy_4cyl)}")
if len(heavy_4cyl) > 0:
    print(heavy_4cyl[['car_name', 'mpg', 'weight', 'origin']])

# Create outlier visualization
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# MPG vs Weight with cylinder coloring
ax = axes[0, 0]
for cyl, marker, color in [(4, 'o', 'blue'), (6, 's', 'green'), (8, '^', 'red')]:
    subset = df[df['cylinders'] == cyl]
    ax.scatter(subset['weight'], subset['mpg'], alpha=0.6, marker=marker,
               label=f'{cyl} cyl', color=color, s=30)
ax.set_xlabel('Weight (lbs)')
ax.set_ylabel('MPG')
ax.set_title('MPG vs Weight by Cylinders')
ax.legend()

# MPG vs Horsepower with origin coloring
ax = axes[0, 1]
origin_colors = {1: 'red', 2: 'green', 3: 'blue'}
origin_labels = {1: 'USA', 2: 'Europe', 3: 'Japan'}
for origin in [1, 2, 3]:
    subset = df[df['origin'] == origin]
    ax.scatter(subset['horsepower'], subset['mpg'], alpha=0.6,
               label=origin_labels[origin], color=origin_colors[origin], s=30)
ax.set_xlabel('Horsepower')
ax.set_ylabel('MPG')
ax.set_title('MPG vs Horsepower by Origin')
ax.legend()

# Residuals from weight model
ax = axes[1, 0]
valid = df[['mpg', 'weight']].dropna()
slope, intercept, _, _, _ = stats.linregress(valid['weight'], valid['mpg'])
residuals = valid['mpg'] - (intercept + slope * valid['weight'])
ax.scatter(range(len(residuals)), sorted(residuals), alpha=0.5, s=10)
ax.axhline(y=0, color='r', linestyle='--')
ax.axhline(y=2*residuals.std(), color='orange', linestyle='--', alpha=0.7)
ax.axhline(y=-2*residuals.std(), color='orange', linestyle='--', alpha=0.7)
ax.set_xlabel('Observation (sorted)')
ax.set_ylabel('Residual (mpg ~ weight)')
ax.set_title('Residuals from Linear Model')

# Cook's distance approximation
ax = axes[1, 1]
n = len(valid)
leverage = 1/n + (valid['weight'] - valid['weight'].mean())**2 / ((valid['weight'] - valid['weight'].mean())**2).sum()
mse = (residuals**2).sum() / (n - 2)
cooks_d = (residuals**2 / (2 * mse)) * (leverage / (1 - leverage)**2)
ax.stem(range(len(cooks_d)), cooks_d.values, markerfmt='o', basefmt='k-')
ax.axhline(y=4/n, color='r', linestyle='--', label=f'Threshold (4/n={4/n:.4f})')
ax.set_xlabel('Observation')
ax.set_ylabel("Cook's Distance")
ax.set_title("Cook's Distance (influential points)")
ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'outliers_influence.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved outlier plots to {OUTPUT_DIR / 'outliers_influence.png'}")
