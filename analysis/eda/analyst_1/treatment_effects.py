"""Analysis of treatment effects and heterogeneity."""
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
print("TREATMENT EFFECTS ANALYSIS")
print("=" * 80)

# Overall treatment effect
print("\n1. OVERALL TREATMENT EFFECT")
print("-" * 40)
control = df[df['treatment'] == 0]['score']
treated = df[df['treatment'] == 1]['score']

control_mean = control.mean()
control_std = control.std()
treated_mean = treated.mean()
treated_std = treated.std()
ate = treated_mean - control_mean

print(f"Control (n={len(control)}):")
print(f"  Mean: {control_mean:.3f}, SD: {control_std:.3f}")
print(f"Treated (n={len(treated)}):")
print(f"  Mean: {treated_mean:.3f}, SD: {treated_std:.3f}")
print(f"\nAverage Treatment Effect (ATE): {ate:.3f}")

# T-test
t_stat, p_value = stats.ttest_ind(treated, control)
print("\nTwo-sample t-test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(control)-1)*control_std**2 + (len(treated)-1)*treated_std**2) / (len(control)+len(treated)-2))
cohens_d = ate / pooled_std
print(f"\nEffect size (Cohen's d): {cohens_d:.3f}")

# Check variance equality
levene_stat, levene_p = stats.levene(treated, control)
print("\nLevene test (equality of variances):")
print(f"  Statistic: {levene_stat:.4f}, p-value: {levene_p:.4f}")
print(f"  Equal variances: {'Yes' if levene_p > 0.05 else 'No'}")

# School-level analysis
print("\n2. SCHOOL-LEVEL TREATMENT EFFECTS")
print("-" * 40)

school_effects = []
for school_id in sorted(df['school_id'].unique()):
    school_data = df[df['school_id'] == school_id]
    control_school = school_data[school_data['treatment'] == 0]['score']
    treated_school = school_data[school_data['treatment'] == 1]['score']

    n_control = len(control_school)
    n_treated = len(treated_school)

    if n_control > 0 and n_treated > 0:
        control_mean_school = control_school.mean()
        treated_mean_school = treated_school.mean()
        school_ate = treated_mean_school - control_mean_school

        school_effects.append({
            'school_id': school_id,
            'school_name': school_data['school_name'].iloc[0],
            'n_control': n_control,
            'n_treated': n_treated,
            'control_mean': control_mean_school,
            'treated_mean': treated_mean_school,
            'ate': school_ate,
            'control_sd': control_school.std() if n_control > 1 else np.nan,
            'treated_sd': treated_school.std() if n_treated > 1 else np.nan
        })

school_effects_df = pd.DataFrame(school_effects)
print(school_effects_df.to_string(index=False))

# Heterogeneity statistics
print("\nSchool-level ATE statistics:")
print(f"  Mean: {school_effects_df['ate'].mean():.3f}")
print(f"  SD: {school_effects_df['ate'].std():.3f}")
print(f"  Min: {school_effects_df['ate'].min():.3f}")
print(f"  Max: {school_effects_df['ate'].max():.3f}")
print(f"  Range: {school_effects_df['ate'].max() - school_effects_df['ate'].min():.3f}")

# Treatment assignment by school
print("\n3. TREATMENT BALANCE BY SCHOOL")
print("-" * 40)
treatment_balance = df.groupby('school_id')['treatment'].agg([
    ('n_total', 'count'),
    ('n_treated', 'sum'),
    ('pct_treated', lambda x: x.mean() * 100)
]).reset_index()
print(treatment_balance.to_string(index=False))

# Check for imbalance
pct_treated_range = treatment_balance['pct_treated'].max() - treatment_balance['pct_treated'].min()
print(f"\nTreatment proportion range across schools: {pct_treated_range:.1f}%")
if pct_treated_range > 20:
    print("  WARNING: Large variation in treatment assignment across schools")

# Save school effects to CSV
school_effects_df.to_csv(output_dir + 'school_treatment_effects.csv', index=False)
print("\nSchool treatment effects saved to: school_treatment_effects.csv")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Treatment group distributions
ax1 = axes[0, 0]
control_data = df[df['treatment'] == 0]['score']
treated_data = df[df['treatment'] == 1]['score']
ax1.hist(control_data, bins=20, alpha=0.6, label='Control', edgecolor='black', density=True)
ax1.hist(treated_data, bins=20, alpha=0.6, label='Treated', edgecolor='black', density=True)
ax1.axvline(control_mean, color='blue', linestyle='--', linewidth=2, alpha=0.7)
ax1.axvline(treated_mean, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax1.set_xlabel('Score')
ax1.set_ylabel('Density')
ax1.set_title(f'Score Distribution by Treatment\nATE = {ate:.2f} (p = {p_value:.4f})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Box plots by treatment
ax2 = axes[0, 1]
data_for_box = [control_data, treated_data]
bp = ax2.boxplot(data_for_box, labels=['Control', 'Treated'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax2.set_ylabel('Score')
ax2.set_title('Score Distribution by Treatment')
ax2.grid(True, alpha=0.3, axis='y')

# School-level treatment effects
ax3 = axes[1, 0]
school_effects_df_sorted = school_effects_df.sort_values('ate')
colors = ['red' if x < 0 else 'green' for x in school_effects_df_sorted['ate']]
ax3.barh(range(len(school_effects_df_sorted)), school_effects_df_sorted['ate'], color=colors, alpha=0.7, edgecolor='black')
ax3.set_yticks(range(len(school_effects_df_sorted)))
ax3.set_yticklabels([f"S{int(x)}" for x in school_effects_df_sorted['school_id']])
ax3.axvline(0, color='black', linestyle='--', linewidth=1)
ax3.axvline(ate, color='blue', linestyle='-', linewidth=2, alpha=0.5, label=f'Overall ATE={ate:.2f}')
ax3.set_xlabel('Treatment Effect (Score Difference)')
ax3.set_ylabel('School')
ax3.set_title('Treatment Effect by School')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='x')

# Score by school and treatment
ax4 = axes[1, 1]
school_means = df.groupby(['school_id', 'treatment'])['score'].mean().reset_index()
school_means_pivot = school_means.pivot(index='school_id', columns='treatment', values='score')

x_pos = np.arange(len(school_means_pivot))
width = 0.35
ax4.bar(x_pos - width/2, school_means_pivot[0], width, label='Control', alpha=0.7, edgecolor='black')
ax4.bar(x_pos + width/2, school_means_pivot[1], width, label='Treated', alpha=0.7, edgecolor='black')
ax4.set_xlabel('School ID')
ax4.set_ylabel('Mean Score')
ax4.set_title('Mean Score by School and Treatment')
ax4.set_xticks(x_pos)
ax4.set_xticklabels([int(x) for x in school_means_pivot.index])
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir + 'treatment_effects.png', dpi=150, bbox_inches='tight')
print("Saved: treatment_effects.png")
plt.close()

# Additional plot: School effect heterogeneity
fig, ax = plt.subplots(figsize=(10, 6))
for school_id in sorted(df['school_id'].unique()):
    school_data = df[df['school_id'] == school_id]
    control_school = school_data[school_data['treatment'] == 0]['score']
    treated_school = school_data[school_data['treatment'] == 1]['score']

    if len(control_school) > 0 and len(treated_school) > 0:
        ax.plot([0, 1], [control_school.mean(), treated_school.mean()],
                'o-', alpha=0.6, label=f'School {school_id}')

ax.set_xlabel('Treatment (0=Control, 1=Treated)')
ax.set_ylabel('Mean Score')
ax.set_title('Treatment Effect by School\n(Lines connect control and treated means within schools)')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Control', 'Treated'])
ax.legend(ncol=2, fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir + 'school_heterogeneity.png', dpi=150, bbox_inches='tight')
print("Saved: school_heterogeneity.png")
plt.close()
