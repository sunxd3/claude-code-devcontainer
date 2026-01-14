"""Calculate variance components and ICC to assess hierarchical structure."""

from pathlib import Path

import numpy as np
import pandas as pd

# Paths
DATA_PATH = Path("/home/user/claude-code-devcontainer/analysis/data/student_scores.csv")
OUTPUT_DIR = Path("/home/user/claude-code-devcontainer/analysis/eda/analyst_2")

# Load data
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("VARIANCE DECOMPOSITION")
print("=" * 60)

# Calculate grand mean
grand_mean = df['score'].mean()
print(f"\nGrand mean score: {grand_mean:.2f}")

# Calculate school means
school_means = df.groupby('school_id')['score'].mean()
school_sizes = df.groupby('school_id').size()

# Total variance (across all students)
total_var = df['score'].var(ddof=1)
print(f"\nTotal variance (all students): {total_var:.2f}")

# Between-school variance (variance of school means, weighted by school size)
# This is the variance in the school means
between_school_var = ((school_means - grand_mean) ** 2 * school_sizes).sum() / (len(df) - 1)

# Within-school variance (pooled within-school variance)
# Calculate variance within each school, then take weighted average
within_school_vars = df.groupby('school_id')['score'].var(ddof=1)
within_school_var = (within_school_vars * (school_sizes - 1)).sum() / (len(df) - len(school_means))

print(f"\nBetween-school variance: {between_school_var:.2f}")
print(f"Within-school variance: {within_school_var:.2f}")
print(f"Sum of components: {between_school_var + within_school_var:.2f}")

# Calculate ICC (Intraclass Correlation Coefficient)
# ICC = between-school variance / total variance
icc = between_school_var / total_var
print("\n" + "=" * 60)
print("INTRACLASS CORRELATION COEFFICIENT (ICC)")
print("=" * 60)
print(f"\nICC = {icc:.4f}")
print(f"  → {icc * 100:.1f}% of variance is between schools")
print(f"  → {(1 - icc) * 100:.1f}% of variance is within schools")

# Interpret ICC
print("\nInterpretation:")
if icc < 0.05:
    print("  → Very weak clustering: pooling may be adequate")
    print("  → But still consider hierarchical model for proper SE estimation")
elif icc < 0.10:
    print("  → Weak to moderate clustering: hierarchical model recommended")
    print("  → Ignoring school structure may bias standard errors")
elif icc < 0.20:
    print("  → Moderate clustering: hierarchical model strongly recommended")
    print("  → Substantial between-school variation to model")
else:
    print("  → Strong clustering: hierarchical model essential")
    print("  → Large proportion of variance is between schools")

# Design effect
# Approximate design effect for cluster randomized designs
# DEFF = 1 + (m-1) * ICC, where m is average cluster size
avg_cluster_size = len(df) / len(school_means)
design_effect = 1 + (avg_cluster_size - 1) * icc
print(f"\nDesign effect (DEFF): {design_effect:.2f}")
print(f"  → Effective sample size: {len(df) / design_effect:.0f} (vs {len(df)} actual)")
print(f"  → Ignoring clustering inflates precision by factor of {np.sqrt(design_effect):.2f}")

# Variance explained by school membership alone
print("\n" + "=" * 60)
print("SCHOOL EFFECTS")
print("=" * 60)

# Calculate R-squared for school membership
# This is the proportion of variance explained by school membership
ss_total = ((df['score'] - grand_mean) ** 2).sum()
ss_between = ((school_means.loc[df['school_id']].values - grand_mean) ** 2).sum()
r_squared_school = ss_between / ss_total

print(f"\nR² for school membership: {r_squared_school:.4f}")
print(f"  → School membership alone explains {r_squared_school * 100:.1f}% of variance")

# Calculate school-level statistics by treatment
print("\n" + "=" * 60)
print("TREATMENT EFFECTS WITHIN AND BETWEEN SCHOOLS")
print("=" * 60)

# Overall treatment effect (ignoring schools)
overall_control = df[df['treatment'] == 0]['score'].mean()
overall_treated = df[df['treatment'] == 1]['score'].mean()
overall_effect = overall_treated - overall_control

print("\nOverall (ignoring schools):")
print(f"  Control mean: {overall_control:.2f}")
print(f"  Treated mean: {overall_treated:.2f}")
print(f"  Treatment effect: {overall_effect:.2f}")

# Within-school treatment effects
school_effects = []
for school_id in sorted(df['school_id'].unique()):
    school_data = df[df['school_id'] == school_id]
    control_mean = school_data[school_data['treatment'] == 0]['score'].mean()
    treated_mean = school_data[school_data['treatment'] == 1]['score'].mean()
    effect = treated_mean - control_mean
    n_control = (school_data['treatment'] == 0).sum()
    n_treated = (school_data['treatment'] == 1).sum()

    school_effects.append({
        'school_id': school_id,
        'control_mean': control_mean,
        'treated_mean': treated_mean,
        'effect': effect,
        'n_control': n_control,
        'n_treated': n_treated
    })

school_effects_df = pd.DataFrame(school_effects)
print("\nWithin-school treatment effects:")
print(school_effects_df.to_string(index=False))

print("\nTreatment effect variability:")
print(f"  Mean effect: {school_effects_df['effect'].mean():.2f}")
print(f"  Std of effects: {school_effects_df['effect'].std():.2f}")
print(f"  Min effect: {school_effects_df['effect'].min():.2f}")
print(f"  Max effect: {school_effects_df['effect'].max():.2f}")
print(f"  Range: {school_effects_df['effect'].max() - school_effects_df['effect'].min():.2f}")

# Check for negative effects
negative_effects = (school_effects_df['effect'] < 0).sum()
print(f"\nSchools with negative treatment effect: {negative_effects}/{len(school_effects_df)}")

# Save variance components
variance_summary = pd.DataFrame([
    {'component': 'Total variance', 'value': f"{total_var:.2f}"},
    {'component': 'Between-school variance', 'value': f"{between_school_var:.2f}"},
    {'component': 'Within-school variance', 'value': f"{within_school_var:.2f}"},
    {'component': 'ICC', 'value': f"{icc:.4f}"},
    {'component': 'Pct variance between schools', 'value': f"{icc * 100:.1f}%"},
    {'component': 'Design effect', 'value': f"{design_effect:.2f}"},
    {'component': 'Effective N', 'value': f"{len(df) / design_effect:.0f}"},
    {'component': 'R² (school membership)', 'value': f"{r_squared_school:.4f}"},
    {'component': 'Overall treatment effect', 'value': f"{overall_effect:.2f}"},
    {'component': 'Mean within-school effect', 'value': f"{school_effects_df['effect'].mean():.2f}"},
    {'component': 'Std within-school effects', 'value': f"{school_effects_df['effect'].std():.2f}"},
])

variance_summary.to_csv(OUTPUT_DIR / "variance_components.csv", index=False)
print("\n✓ Saved variance_components.csv")

# Save school-level treatment effects
school_effects_df.to_csv(OUTPUT_DIR / "school_treatment_effects.csv", index=False)
print("✓ Saved school_treatment_effects.csv")
