"""
Auto-MPG Dataset: Grouping Structure and Temporal Analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

OUTPUT_DIR = Path("/workspace/analysis/eda")
df = pd.read_csv(OUTPUT_DIR / 'auto_mpg_cleaned.csv')

# Create readable labels
df['origin_label'] = df['origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
df['year_full'] = 1900 + df['model_year']

print("="*60)
print("GROUPING ANALYSIS: ORIGIN")
print("="*60)

# MPG by origin
print("\n--- MPG by Origin ---")
origin_stats = df.groupby('origin_label')['mpg'].agg(['count', 'mean', 'std', 'median', 'min', 'max'])
print(origin_stats.round(2))

# ANOVA test
groups = [df[df['origin'] == i]['mpg'].values for i in [1, 2, 3]]
f_stat, p_val = stats.f_oneway(*groups)
print(f"\nOne-way ANOVA: F={f_stat:.2f}, p={p_val:.2e}")

# Kruskal-Wallis (nonparametric)
h_stat, p_kw = stats.kruskal(*groups)
print(f"Kruskal-Wallis: H={h_stat:.2f}, p={p_kw:.2e}")

# Effect size (eta-squared)
ss_between = sum(len(g) * (g.mean() - df['mpg'].mean())**2 for g in groups)
ss_total = sum((df['mpg'] - df['mpg'].mean())**2)
eta_squared = ss_between / ss_total
print(f"Effect size (eta-squared): {eta_squared:.3f}")

print("\n="*60)
print("GROUPING ANALYSIS: CYLINDERS")
print("="*60)

# MPG by cylinders
print("\n--- MPG by Cylinders ---")
cyl_stats = df.groupby('cylinders')['mpg'].agg(['count', 'mean', 'std', 'median', 'min', 'max'])
print(cyl_stats.round(2))

# Focus on common cylinder counts (4, 6, 8)
df_main_cyl = df[df['cylinders'].isin([4, 6, 8])]
groups_cyl = [df_main_cyl[df_main_cyl['cylinders'] == i]['mpg'].values for i in [4, 6, 8]]
f_stat_cyl, p_val_cyl = stats.f_oneway(*groups_cyl)
print(f"\nOne-way ANOVA (4,6,8 cyl): F={f_stat_cyl:.2f}, p={p_val_cyl:.2e}")

# Cross-tabulation: Origin x Cylinders
print("\n--- Cross-tabulation: Origin x Cylinders ---")
crosstab = pd.crosstab(df['origin_label'], df['cylinders'])
print(crosstab)

print("\n--- Mean MPG by Origin x Cylinders ---")
mpg_pivot = df.groupby(['origin_label', 'cylinders'])['mpg'].mean().unstack()
print(mpg_pivot.round(2))

print("\n="*60)
print("TEMPORAL ANALYSIS: MODEL YEAR")
print("="*60)

# MPG trend over time
print("\n--- MPG by Model Year ---")
year_stats = df.groupby('model_year')['mpg'].agg(['count', 'mean', 'std', 'median'])
print(year_stats.round(2))

# Linear trend
slope, intercept, r, p, se = stats.linregress(df['model_year'], df['mpg'])
print(f"\nLinear trend: slope={slope:.3f} mpg/year, r={r:.3f}, p={p:.2e}")
print(f"Interpretation: MPG increased by ~{slope:.2f} mpg per year on average")

# Check for origin-specific trends
print("\n--- MPG Trend by Origin ---")
for origin in [1, 2, 3]:
    subset = df[df['origin'] == origin]
    slope, intercept, r, p, se = stats.linregress(subset['model_year'], subset['mpg'])
    label = {1: 'USA', 2: 'Europe', 3: 'Japan'}[origin]
    print(f"  {label}: slope={slope:.3f} mpg/year, r={r:.3f}")

# Create grouping plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# MPG by origin boxplot
ax = axes[0, 0]
origin_order = ['USA', 'Europe', 'Japan']
data_origin = [df[df['origin_label'] == o]['mpg'].values for o in origin_order]
bp = ax.boxplot(data_origin, labels=origin_order, patch_artist=True)
colors = ['#ff7f7f', '#7fbf7f', '#7f7fff']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_ylabel('MPG')
ax.set_title('MPG by Origin')

# MPG by cylinders boxplot
ax = axes[0, 1]
cyl_order = [4, 6, 8]
data_cyl = [df[df['cylinders'] == c]['mpg'].values for c in cyl_order]
ax.boxplot(data_cyl, labels=cyl_order)
ax.set_xlabel('Cylinders')
ax.set_ylabel('MPG')
ax.set_title('MPG by Cylinders')

# MPG trend over time
ax = axes[1, 0]
for origin, color, label in zip([1, 2, 3], ['red', 'green', 'blue'], ['USA', 'Europe', 'Japan']):
    subset = df[df['origin'] == origin]
    yearly = subset.groupby('model_year')['mpg'].mean()
    ax.plot(yearly.index + 1900, yearly.values, 'o-', color=color, label=label, alpha=0.7)
ax.set_xlabel('Year')
ax.set_ylabel('Mean MPG')
ax.set_title('MPG Trend by Origin')
ax.legend()
ax.grid(True, alpha=0.3)

# Overall trend with confidence band
ax = axes[1, 1]
yearly_mean = df.groupby('model_year')['mpg'].mean()
yearly_std = df.groupby('model_year')['mpg'].std()
yearly_n = df.groupby('model_year')['mpg'].count()
yearly_se = yearly_std / np.sqrt(yearly_n)

years = yearly_mean.index + 1900
ax.plot(years, yearly_mean.values, 'ko-', markersize=6)
ax.fill_between(years, yearly_mean - 1.96*yearly_se, yearly_mean + 1.96*yearly_se,
                alpha=0.3, color='blue')

# Add trend line
slope, intercept, _, _, _ = stats.linregress(df['model_year'], df['mpg'])
ax.plot(years, intercept + slope * (years - 1900), 'r--', lw=2,
        label=f'Trend: {slope:.2f} mpg/year')
ax.set_xlabel('Year')
ax.set_ylabel('Mean MPG')
ax.set_title('Overall MPG Trend (95% CI)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'grouping_temporal.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved grouping and temporal plots to {OUTPUT_DIR / 'grouping_temporal.png'}")

# Interaction: Does the year effect differ by origin?
print("\n--- Interaction Analysis: Year x Origin ---")
# Two-way patterns
for origin in [1, 2, 3]:
    subset = df[df['origin'] == origin]
    early = subset[subset['model_year'] <= 75]['mpg'].mean()
    late = subset[subset['model_year'] >= 78]['mpg'].mean()
    change = late - early
    label = {1: 'USA', 2: 'Europe', 3: 'Japan'}[origin]
    print(f"  {label}: Early (70-75) mean={early:.1f}, Late (78-82) mean={late:.1f}, Change={change:+.1f}")
