"""Visualize treatment effects by school and overall."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# Paths
DATA_PATH = Path("/home/user/claude-code-devcontainer/analysis/data/student_scores.csv")
OUTPUT_DIR = Path("/home/user/claude-code-devcontainer/analysis/eda/analyst_2")

# Load data
df = pd.read_csv(DATA_PATH)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Plot 1: Treatment vs control by school (side-by-side comparison)
fig, ax = plt.subplots(figsize=(12, 6))

# Create position for each school and treatment group
school_ids = sorted(df['school_id'].unique())
x = []
heights = []
colors = []
labels_added = {'Control': False, 'Treatment': False}

for i, school_id in enumerate(school_ids):
    school_data = df[df['school_id'] == school_id]
    control_mean = school_data[school_data['treatment'] == 0]['score'].mean()
    treated_mean = school_data[school_data['treatment'] == 1]['score'].mean()

    # Control bar
    x.append(i - 0.2)
    heights.append(control_mean)
    colors.append('lightcoral')

    # Treatment bar
    x.append(i + 0.2)
    heights.append(treated_mean)
    colors.append('lightblue')

bars = ax.bar(x, heights, width=0.35, color=colors, edgecolor='black', linewidth=1.2)

# Custom legend

legend_elements = [Patch(facecolor='lightcoral', edgecolor='black', label='Control'),
                   Patch(facecolor='lightblue', edgecolor='black', label='Treatment')]
ax.legend(handles=legend_elements, fontsize=11)

ax.set_xlabel('School ID', fontsize=12)
ax.set_ylabel('Mean Test Score', fontsize=12)
ax.set_title('Treatment vs Control Mean Scores by School', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(school_ids)))
ax.set_xticklabels(school_ids)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "treatment_by_school.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved treatment_by_school.png")

# Plot 2: Within-school treatment effects with confidence
school_effects = []
for school_id in sorted(df['school_id'].unique()):
    school_data = df[df['school_id'] == school_id]
    control_scores = school_data[school_data['treatment'] == 0]['score']
    treated_scores = school_data[school_data['treatment'] == 1]['score']

    effect = treated_scores.mean() - control_scores.mean()

    # Calculate SE for the difference
    se_control = control_scores.sem()
    se_treated = treated_scores.sem()
    se_diff = (se_control**2 + se_treated**2)**0.5

    school_effects.append({
        'school_id': school_id,
        'effect': effect,
        'se': se_diff
    })

effects_df = pd.DataFrame(school_effects).sort_values('effect')

fig, ax = plt.subplots(figsize=(10, 6))

ax.errorbar(effects_df['school_id'], effects_df['effect'],
            yerr=1.96 * effects_df['se'],
            fmt='o', markersize=10, capsize=6, capthick=2, linewidth=2,
            color='darkgreen', ecolor='gray')

# Add overall effect line
overall_effect = df[df['treatment'] == 1]['score'].mean() - df[df['treatment'] == 0]['score'].mean()
ax.axhline(overall_effect, color='red', linestyle='--', linewidth=2,
           label=f'Overall effect ({overall_effect:.2f})')
ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

ax.set_xlabel('School ID', fontsize=12)
ax.set_ylabel('Treatment Effect (Treated - Control)', fontsize=12)
ax.set_title('Within-School Treatment Effects with 95% CI', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "treatment_effects_by_school.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved treatment_effects_by_school.png")

# Plot 3: Scatter plot with school coloring
fig, ax = plt.subplots(figsize=(10, 6))

# Create a color map for schools
school_colors = plt.cm.tab10(range(len(school_ids)))
color_map = dict(zip(school_ids, school_colors))

for school_id in school_ids:
    school_data = df[df['school_id'] == school_id]
    control_data = school_data[school_data['treatment'] == 0]
    treated_data = school_data[school_data['treatment'] == 1]

    # Plot control
    ax.scatter([0] * len(control_data), control_data['score'],
               color=color_map[school_id], alpha=0.6, s=60, edgecolors='black', linewidth=0.5)

    # Plot treatment
    ax.scatter([1] * len(treated_data), treated_data['score'],
               color=color_map[school_id], alpha=0.6, s=60, edgecolors='black', linewidth=0.5,
               label=f'School {school_id}')

    # Draw connecting lines for school means
    control_mean = control_data['score'].mean()
    treated_mean = treated_data['score'].mean()
    ax.plot([0, 1], [control_mean, treated_mean],
            color=color_map[school_id], linewidth=2, alpha=0.8)

ax.set_xlim(-0.3, 1.3)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Control', 'Treatment'], fontsize=12)
ax.set_ylabel('Test Score', fontsize=12)
ax.set_title('Treatment Effects with School Structure', fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "treatment_scatter_by_school.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved treatment_scatter_by_school.png")
