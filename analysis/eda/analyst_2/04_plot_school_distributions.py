"""Visualize school-level score distributions."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Paths
DATA_PATH = Path("/home/user/claude-code-devcontainer/analysis/data/student_scores.csv")
OUTPUT_DIR = Path("/home/user/claude-code-devcontainer/analysis/eda/analyst_2")

# Load data
df = pd.read_csv(DATA_PATH)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Plot 1: Box plots of scores by school
fig, ax = plt.subplots(figsize=(10, 6))

# Sort schools by mean score
school_order = df.groupby('school_id')['score'].mean().sort_values().index

sns.boxplot(data=df, x='school_id', y='score', order=school_order, ax=ax,
            palette='Set2', linewidth=1.5)

ax.axhline(df['score'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Grand mean ({df["score"].mean():.1f})')

ax.set_xlabel('School ID', fontsize=12)
ax.set_ylabel('Test Score', fontsize=12)
ax.set_title('Score Distributions by School', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "school_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved school_distributions.png")

# Plot 2: Violin plots showing within-school variation
fig, ax = plt.subplots(figsize=(10, 6))

sns.violinplot(data=df, x='school_id', y='score', order=school_order, ax=ax,
               palette='Set2', inner='box', linewidth=1.5)

ax.axhline(df['score'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Grand mean ({df["score"].mean():.1f})')

ax.set_xlabel('School ID', fontsize=12)
ax.set_ylabel('Test Score', fontsize=12)
ax.set_title('Score Distributions with Density (Violin Plots)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "school_violin_plots.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved school_violin_plots.png")

# Plot 3: School means with error bars
school_stats = df.groupby('school_id').agg({
    'score': ['mean', 'sem', 'count']
}).reset_index()
school_stats.columns = ['school_id', 'mean', 'sem', 'n']
school_stats = school_stats.sort_values('mean')

fig, ax = plt.subplots(figsize=(10, 6))

ax.errorbar(school_stats['school_id'], school_stats['mean'],
            yerr=1.96 * school_stats['sem'],
            fmt='o', markersize=8, capsize=5, capthick=2, linewidth=2,
            color='steelblue', ecolor='gray', label='95% CI')

ax.axhline(df['score'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Grand mean ({df["score"].mean():.1f})')

# Add sample sizes as text
for _, row in school_stats.iterrows():
    ax.text(row['school_id'], row['mean'] + 2, f"n={row['n']:.0f}",
            ha='center', va='bottom', fontsize=9)

ax.set_xlabel('School ID', fontsize=12)
ax.set_ylabel('Mean Test Score', fontsize=12)
ax.set_title('School Mean Scores with 95% Confidence Intervals', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "school_means.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved school_means.png")
