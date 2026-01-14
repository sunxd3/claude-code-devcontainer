"""Univariate analysis of key variables."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Load data
data_path = "/home/user/claude-code-devcontainer/analysis/data/student_scores.csv"
df = pd.read_csv(data_path)
output_dir = "/home/user/claude-code-devcontainer/analysis/eda/analyst_1/"

print("=" * 80)
print("UNIVARIATE ANALYSIS")
print("=" * 80)

# Treatment distribution
print("\n1. TREATMENT DISTRIBUTION")
print("-" * 40)
treatment_counts = df['treatment'].value_counts().sort_index()
treatment_pct = df['treatment'].value_counts(normalize=True).sort_index() * 100
print("Treatment counts:")
for val, count in treatment_counts.items():
    pct = treatment_pct[val]
    print(f"  {val}: {count} ({pct:.1f}%)")

# Score distribution
print("\n2. SCORE DISTRIBUTION")
print("-" * 40)
score_stats = {
    'mean': df['score'].mean(),
    'std': df['score'].std(),
    'median': df['score'].median(),
    'min': df['score'].min(),
    'max': df['score'].max(),
    'q25': df['score'].quantile(0.25),
    'q75': df['score'].quantile(0.75),
    'skewness': stats.skew(df['score']),
    'kurtosis': stats.kurtosis(df['score'])
}
print("Score statistics:")
for key, val in score_stats.items():
    print(f"  {key}: {val:.3f}")

# Test normality
shapiro_stat, shapiro_p = stats.shapiro(df['score'])
print("\nShapiro-Wilk test for normality:")
print(f"  Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}")
if shapiro_p < 0.05:
    print("  Result: Reject normality (p < 0.05)")
else:
    print("  Result: Cannot reject normality (p >= 0.05)")

# Check for zero-inflation or boundary effects
print("\nBoundary and special value checks:")
n_below_50 = (df['score'] < 50).sum()
n_above_100 = (df['score'] > 100).sum()
print(f"  Scores < 50: {n_below_50} ({n_below_50/len(df)*100:.1f}%)")
print(f"  Scores > 100: {n_above_100} ({n_above_100/len(df)*100:.1f}%)")

# School-level summary
print("\n3. SCHOOL-LEVEL DISTRIBUTION")
print("-" * 40)
school_counts = df['school_id'].value_counts().sort_index()
print("Students per school:")
print(school_counts.to_string())
print("\nSchool size statistics:")
print(f"  Mean: {school_counts.mean():.1f}")
print(f"  Std: {school_counts.std():.1f}")
print(f"  Min: {school_counts.min()}")
print(f"  Max: {school_counts.max()}")

# Create univariate summary table
univariate_summary = []
for col in ['treatment', 'score', 'school_id']:
    row = {
        'column': col,
        'n_obs': len(df[col]),
        'n_unique': df[col].nunique(),
        'mean': df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else np.nan,
        'std': df[col].std() if pd.api.types.is_numeric_dtype(df[col]) else np.nan,
        'min': df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else np.nan,
        'q25': df[col].quantile(0.25) if pd.api.types.is_numeric_dtype(df[col]) else np.nan,
        'median': df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else np.nan,
        'q75': df[col].quantile(0.75) if pd.api.types.is_numeric_dtype(df[col]) else np.nan,
        'max': df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else np.nan,
        'skewness': stats.skew(df[col]) if pd.api.types.is_numeric_dtype(df[col]) else np.nan,
    }
    univariate_summary.append(row)

univariate_df = pd.DataFrame(univariate_summary)
output_path = output_dir + "univariate_summary.csv"
univariate_df.to_csv(output_path, index=False)
print("\nUnivariate summary saved to: univariate_summary.csv")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Score histogram with density
ax1 = axes[0, 0]
ax1.hist(df['score'], bins=30, alpha=0.7, edgecolor='black', density=True)
# Overlay normal distribution
mu, sigma = df['score'].mean(), df['score'].std()
x = np.linspace(df['score'].min(), df['score'].max(), 100)
ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal fit')
ax1.set_xlabel('Score')
ax1.set_ylabel('Density')
ax1.set_title(f'Score Distribution\n(mean={mu:.1f}, sd={sigma:.1f}, skew={score_stats["skewness"]:.2f})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Score boxplot
ax2 = axes[0, 1]
bp = ax2.boxplot([df['score']], vert=True, patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
ax2.set_ylabel('Score')
ax2.set_title('Score Boxplot')
ax2.set_xticklabels(['All'])
ax2.grid(True, alpha=0.3)

# Q-Q plot
ax3 = axes[1, 0]
stats.probplot(df['score'], dist="norm", plot=ax3)
ax3.set_title('Q-Q Plot (Normal)')
ax3.grid(True, alpha=0.3)

# Treatment distribution
ax4 = axes[1, 1]
treatment_data = df['treatment'].value_counts().sort_index()
bars = ax4.bar(treatment_data.index, treatment_data.values, alpha=0.7, edgecolor='black')
ax4.set_xlabel('Treatment')
ax4.set_ylabel('Count')
ax4.set_title(f'Treatment Balance\n(0: n={treatment_counts[0]}, 1: n={treatment_counts[1]})')
ax4.set_xticks([0, 1])
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir + 'univariate_distributions.png', dpi=150, bbox_inches='tight')
print("Saved: univariate_distributions.png")
plt.close()
