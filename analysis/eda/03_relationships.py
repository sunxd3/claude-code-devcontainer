"""
Auto-MPG Dataset: Relationships and Correlation Analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

OUTPUT_DIR = Path("/workspace/analysis/eda")
df = pd.read_csv(OUTPUT_DIR / 'auto_mpg_cleaned.csv')

print("="*60)
print("CORRELATION AND RELATIONSHIP ANALYSIS")
print("="*60)

# Continuous predictors
cont_cols = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']
df_cont = df[cont_cols].dropna()

# Pearson correlation
print("\n--- Pearson Correlation Matrix ---")
corr_pearson = df_cont.corr()
print(corr_pearson.round(3))

# Spearman correlation (robust to nonlinearity)
print("\n--- Spearman Correlation Matrix ---")
corr_spearman = df_cont.corr(method='spearman')
print(corr_spearman.round(3))

# Correlations with MPG
print("\n--- Correlations with MPG ---")
for col in ['displacement', 'horsepower', 'weight', 'acceleration', 'model_year']:
    valid = df[[col, 'mpg']].dropna()
    r_pearson, p_pearson = stats.pearsonr(valid[col], valid['mpg'])
    r_spearman, p_spearman = stats.spearmanr(valid[col], valid['mpg'])
    print(f"{col:15s}: Pearson r={r_pearson:+.3f} (p={p_pearson:.2e}), "
          f"Spearman rho={r_spearman:+.3f} (p={p_spearman:.2e})")

# Check multicollinearity
print("\n--- Multicollinearity Check (Predictors Only) ---")
pred_cols = ['displacement', 'horsepower', 'weight', 'acceleration']
df_pred = df[pred_cols].dropna()
corr_pred = df_pred.corr()
print(corr_pred.round(3))

# Flag high correlations
print("\nHighly correlated pairs (|r| > 0.8):")
for i in range(len(pred_cols)):
    for j in range(i+1, len(pred_cols)):
        r = corr_pred.iloc[i, j]
        if abs(r) > 0.8:
            print(f"  {pred_cols[i]} - {pred_cols[j]}: r = {r:.3f}")

# VIF calculation
from numpy.linalg import inv
print("\n--- Variance Inflation Factors ---")
X = df_pred.values
X_std = (X - X.mean(axis=0)) / X.std(axis=0)
corr_matrix = np.corrcoef(X_std.T)
try:
    corr_inv = inv(corr_matrix)
    vif = np.diag(corr_inv)
    for col, v in zip(pred_cols, vif):
        status = "HIGH" if v > 5 else "OK"
        print(f"  {col:15s}: VIF = {v:.2f} ({status})")
except:
    print("  Could not compute VIF (singular matrix)")

# Scatterplot matrix
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
plot_cols = ['mpg', 'displacement', 'weight', 'horsepower']

for i, col_i in enumerate(plot_cols):
    for j, col_j in enumerate(plot_cols):
        ax = axes[i, j]
        if i == j:
            ax.hist(df[col_i].dropna(), bins=25, edgecolor='black', alpha=0.7)
            ax.set_xlabel(col_i)
        else:
            valid = df[[col_i, col_j]].dropna()
            ax.scatter(valid[col_j], valid[col_i], alpha=0.5, s=15)
            if j == 0:
                ax.set_ylabel(col_i)
            if i == len(plot_cols) - 1:
                ax.set_xlabel(col_j)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'scatterplot_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved scatterplot matrix to {OUTPUT_DIR / 'scatterplot_matrix.png'}")

# MPG vs each predictor with LOWESS
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
predictors = ['displacement', 'horsepower', 'weight', 'acceleration', 'model_year']

for idx, pred in enumerate(predictors):
    ax = axes.flat[idx]
    valid = df[['mpg', pred]].dropna()
    ax.scatter(valid[pred], valid['mpg'], alpha=0.5, s=20)

    # Add linear fit
    slope, intercept, r, p, se = stats.linregress(valid[pred], valid['mpg'])
    x_line = np.linspace(valid[pred].min(), valid[pred].max(), 100)
    ax.plot(x_line, intercept + slope * x_line, 'r-', lw=2,
            label=f'r={r:.2f}')

    ax.set_xlabel(pred)
    ax.set_ylabel('MPG')
    ax.legend(loc='best')
    ax.set_title(f'MPG vs {pred}')

axes.flat[5].axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'mpg_vs_predictors.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved MPG vs predictors to {OUTPUT_DIR / 'mpg_vs_predictors.png'}")

# Check nonlinearity with residual plots
print("\n--- Nonlinearity Detection ---")
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for idx, pred in enumerate(predictors):
    ax = axes.flat[idx]
    valid = df[['mpg', pred]].dropna()

    slope, intercept, r, p, se = stats.linregress(valid[pred], valid['mpg'])
    residuals = valid['mpg'] - (intercept + slope * valid[pred])

    ax.scatter(valid[pred], residuals, alpha=0.5, s=20)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel(pred)
    ax.set_ylabel('Residuals')
    ax.set_title(f'Residuals vs {pred}')

axes.flat[5].axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'residual_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved residual plots to {OUTPUT_DIR / 'residual_plots.png'}")

# Test quadratic vs linear for key predictors
print("\n--- Quadratic vs Linear Fit (key predictors) ---")
for pred in ['weight', 'horsepower', 'displacement']:
    valid = df[['mpg', pred]].dropna()
    x, y = valid[pred].values, valid['mpg'].values

    # Linear fit
    p1 = np.polyfit(x, y, 1)
    y_pred_1 = np.polyval(p1, x)
    ss_res_1 = np.sum((y - y_pred_1)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2_linear = 1 - ss_res_1/ss_tot

    # Quadratic fit
    p2 = np.polyfit(x, y, 2)
    y_pred_2 = np.polyval(p2, x)
    ss_res_2 = np.sum((y - y_pred_2)**2)
    r2_quad = 1 - ss_res_2/ss_tot

    # Improvement
    improvement = r2_quad - r2_linear
    print(f"  {pred}: Linear R2={r2_linear:.4f}, Quadratic R2={r2_quad:.4f}, "
          f"Improvement={improvement:.4f}")
