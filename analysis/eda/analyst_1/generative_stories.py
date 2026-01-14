"""Test competing data-generating stories about treatment mechanism."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import linregress

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Load data
data_path = "/home/user/claude-code-devcontainer/analysis/data/student_scores.csv"
df = pd.read_csv(data_path)
output_dir = "/home/user/claude-code-devcontainer/analysis/eda/analyst_1/"

print("=" * 80)
print("TESTING COMPETING DATA-GENERATING STORIES")
print("=" * 80)

# Story 1: Fixed treatment effect (no school heterogeneity)
print("\nSTORY 1: Fixed Treatment Effect")
print("-" * 40)
print("Model: Y_ij = β0 + β1*Treatment_ij + ε_ij")
print("Assumption: Treatment effect is constant across schools")

school_effects = []
for school_id in sorted(df['school_id'].unique()):
    school_data = df[df['school_id'] == school_id]
    control = school_data[school_data['treatment'] == 0]['score']
    treated = school_data[school_data['treatment'] == 1]['score']
    if len(control) > 0 and len(treated) > 0:
        ate = treated.mean() - control.mean()
        school_effects.append(ate)

ate_variance = np.var(school_effects)
ate_mean = np.mean(school_effects)
print(f"School-level ATE variance: {ate_variance:.3f}")
print(f"School-level ATE range: [{min(school_effects):.3f}, {max(school_effects):.3f}]")
print(f"Coefficient of variation: {np.std(school_effects)/abs(ate_mean):.3f}")
print("Evidence: HIGH variance suggests school-specific effects exist")

# Story 2: Random school intercepts (varying baselines, fixed treatment)
print("\n\nSTORY 2: Random School Intercepts + Fixed Treatment Effect")
print("-" * 40)
print("Model: Y_ij = β0 + α_j + β1*Treatment_ij + ε_ij, α_j ~ N(0, τ²)")
print("Assumption: Schools differ in baseline but treatment effect is constant")

school_baselines = df.groupby('school_id')['score'].mean()
school_baseline_var = school_baselines.var()
print(f"School baseline mean variance: {school_baseline_var:.3f}")

# Compare control group means across schools
control_only = df[df['treatment'] == 0]
control_means_by_school = control_only.groupby('school_id')['score'].mean()
control_baseline_var = control_means_by_school.var()
print(f"Control group baseline variance across schools: {control_baseline_var:.3f}")

# ANOVA on control group
schools_control = [control_only[control_only['school_id'] == s]['score'].values
                   for s in sorted(control_only['school_id'].unique())]
f_stat, p_val = stats.f_oneway(*schools_control)
print(f"ANOVA on control groups: F={f_stat:.3f}, p={p_val:.4f}")
print(f"Evidence: {'Strong' if p_val < 0.05 else 'Weak'} evidence for baseline school differences")

# Story 3: Random school slopes (treatment effect varies by school)
print("\n\nSTORY 3: Random School Intercepts + Random Treatment Slopes")
print("-" * 40)
print("Model: Y_ij = β0 + α_j + (β1 + γ_j)*Treatment_ij + ε_ij")
print("Assumption: Treatment effect varies by school")

school_treatment_effects = df.groupby(['school_id', 'treatment'])['score'].mean().unstack()
if 0 in school_treatment_effects.columns and 1 in school_treatment_effects.columns:
    school_ate = school_treatment_effects[1] - school_treatment_effects[0]
    ate_sd = school_ate.std()
    print(f"SD of school-level ATEs: {ate_sd:.3f}")
    print(f"Range of school-level ATEs: {school_ate.max() - school_ate.min():.3f}")

    # Test if slope variance is significant
    # Under fixed effect, we'd expect small variance relative to sampling error
    overall_ate = df[df['treatment']==1]['score'].mean() - df[df['treatment']==0]['score'].mean()
    print(f"Overall ATE: {overall_ate:.3f}")
    print(f"Evidence: Large SD ({ate_sd:.3f}) relative to overall ATE suggests varying slopes")

# Story 4: Within-school correlation structure
print("\n\nSTORY 4: Within-School Correlation")
print("-" * 40)
print("Question: Are students within schools more similar than across schools?")

# Calculate ICC (Intraclass Correlation Coefficient)
grand_mean = df['score'].mean()
between_school_var = 0
within_school_var = 0
n_schools = df['school_id'].nunique()

for school_id in df['school_id'].unique():
    school_data = df[df['school_id'] == school_id]['score']
    n_school = len(school_data)
    school_mean = school_data.mean()
    between_school_var += n_school * (school_mean - grand_mean)**2
    within_school_var += ((school_data - school_mean)**2).sum()

between_school_var /= (n_schools - 1)
within_school_var /= (len(df) - n_schools)

icc = between_school_var / (between_school_var + within_school_var)
print(f"Between-school variance: {between_school_var:.3f}")
print(f"Within-school variance: {within_school_var:.3f}")
print(f"ICC (Intraclass Correlation): {icc:.3f}")
print(f"Interpretation: {icc*100:.1f}% of variance is between schools")
print("Evidence: ICC > 0.05 suggests hierarchical structure is important")

# Story 5: Treatment-by-baseline interaction
print("\n\nSTORY 5: Treatment Effect Moderated by Baseline School Performance")
print("-" * 40)
print("Question: Does treatment effect depend on school baseline?")

school_summary = []
for school_id in sorted(df['school_id'].unique()):
    school_data = df[df['school_id'] == school_id]
    control = school_data[school_data['treatment'] == 0]['score']
    treated = school_data[school_data['treatment'] == 1]['score']
    if len(control) > 0 and len(treated) > 0:
        baseline = control.mean()
        ate = treated.mean() - control.mean()
        school_summary.append({'school_id': school_id, 'baseline': baseline, 'ate': ate})

school_summary_df = pd.DataFrame(school_summary)
corr = school_summary_df[['baseline', 'ate']].corr().iloc[0, 1]
print(f"Correlation between baseline and ATE: {corr:.3f}")

# Linear regression
slope, intercept, r_value, p_value_reg, std_err = linregress(
    school_summary_df['baseline'], school_summary_df['ate']
)
print(f"Regression: ATE = {intercept:.3f} + {slope:.3f} * baseline")
print(f"R² = {r_value**2:.3f}, p = {p_value_reg:.4f}")
print(f"Evidence: {'Significant' if p_value_reg < 0.1 else 'No'} baseline-treatment interaction")

# Summary recommendations
print("\n" + "=" * 80)
print("SYNTHESIS: Which story fits best?")
print("=" * 80)

recommendations = []
if ate_variance > 10:
    recommendations.append("✓ Strong treatment effect heterogeneity across schools (Story 3)")
else:
    recommendations.append("✓ Modest heterogeneity; fixed effect may suffice (Story 1)")

if icc > 0.05:
    recommendations.append("✓ Non-negligible ICC; hierarchical structure needed (Story 2/3)")
else:
    recommendations.append("✓ Low ICC; pooling may be acceptable")

if p_val < 0.05:
    recommendations.append("✓ Significant baseline differences across schools (Story 2)")

if abs(corr) > 0.3 and p_value_reg < 0.1:
    recommendations.append("✓ Treatment effect may depend on baseline (Story 5)")

print("\nKey findings:")
for rec in recommendations:
    print(f"  {rec}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: School-level ATEs
ax1 = axes[0, 0]
school_ate_sorted = school_summary_df.sort_values('ate')
colors = ['red' if x < overall_ate else 'green' for x in school_ate_sorted['ate']]
ax1.barh(range(len(school_ate_sorted)), school_ate_sorted['ate'],
         color=colors, alpha=0.7, edgecolor='black')
ax1.axvline(overall_ate, color='blue', linestyle='--', linewidth=2, label=f'Overall ATE={overall_ate:.2f}')
ax1.set_yticks(range(len(school_ate_sorted)))
ax1.set_yticklabels([f"S{int(x)}" for x in school_ate_sorted['school_id']])
ax1.set_xlabel('School-Level ATE')
ax1.set_ylabel('School')
ax1.set_title(f'Treatment Effect Heterogeneity\n(SD={ate_sd:.2f})')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='x')

# Plot 2: ATE vs Baseline
ax2 = axes[0, 1]
ax2.scatter(school_summary_df['baseline'], school_summary_df['ate'], s=100, alpha=0.7, edgecolor='black')
ax2.plot(school_summary_df['baseline'],
         intercept + slope * school_summary_df['baseline'],
         'r--', linewidth=2, label=f'Fit: slope={slope:.3f}')
for _, row in school_summary_df.iterrows():
    ax2.annotate(f"S{int(row['school_id'])}", (row['baseline'], row['ate']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
ax2.set_xlabel('Baseline (Control Mean)')
ax2.set_ylabel('Treatment Effect (ATE)')
ax2.set_title(f'Treatment Effect vs Baseline\n(r={corr:.3f}, p={p_value_reg:.3f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Variance decomposition
ax3 = axes[1, 0]
variance_components = ['Between\nSchools', 'Within\nSchools']
variance_values = [between_school_var, within_school_var]
bars = ax3.bar(variance_components, variance_values, alpha=0.7, edgecolor='black',
               color=['steelblue', 'coral'])
for i, (comp, val) in enumerate(zip(variance_components, variance_values)):
    ax3.text(i, val + 2, f'{val:.1f}', ha='center', fontsize=10)
ax3.set_ylabel('Variance')
ax3.set_title(f'Variance Decomposition\n(ICC={icc:.3f})')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: School means with CIs
ax4 = axes[1, 1]
school_stats = df.groupby('school_id')['score'].agg(['mean', 'sem']).reset_index()
school_stats = school_stats.sort_values('mean')
ax4.errorbar(range(len(school_stats)), school_stats['mean'],
             yerr=1.96*school_stats['sem'], fmt='o', capsize=5, alpha=0.7)
ax4.axhline(grand_mean, color='red', linestyle='--', linewidth=2, label=f'Grand mean={grand_mean:.1f}')
ax4.set_xticks(range(len(school_stats)))
ax4.set_xticklabels([f"S{int(x)}" for x in school_stats['school_id']], rotation=0)
ax4.set_xlabel('School')
ax4.set_ylabel('Mean Score')
ax4.set_title('School Mean Scores (95% CI)')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir + 'generative_stories.png', dpi=150, bbox_inches='tight')
print("\nSaved: generative_stories.png")
plt.close()
