"""Analyze school-level hierarchical structure and sample sizes."""

from pathlib import Path

import pandas as pd

# Paths
DATA_PATH = Path("/home/user/claude-code-devcontainer/analysis/data/student_scores.csv")
OUTPUT_DIR = Path("/home/user/claude-code-devcontainer/analysis/eda/analyst_2")

# Load data
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("SCHOOL-LEVEL STRUCTURE")
print("=" * 60)

# School sample sizes
school_counts = df.groupby('school_id').size().reset_index(name='n_students')
school_counts = school_counts.sort_values('n_students', ascending=False)

print("\nSchool sample sizes:")
print(school_counts.to_string(index=False))
print(f"\nTotal schools: {len(school_counts)}")
print(f"Min students per school: {school_counts['n_students'].min()}")
print(f"Max students per school: {school_counts['n_students'].max()}")
print(f"Mean students per school: {school_counts['n_students'].mean():.1f}")
print(f"Median students per school: {school_counts['n_students'].median():.1f}")
print(f"Std students per school: {school_counts['n_students'].std():.1f}")

# Check balance
cv = school_counts['n_students'].std() / school_counts['n_students'].mean()
print(f"\nCoefficient of variation: {cv:.3f}")
if cv < 0.2:
    print("  → Schools are well balanced")
elif cv < 0.5:
    print("  → Schools have moderate imbalance")
else:
    print("  → Schools are highly imbalanced")

# Treatment assignment by school
print("\n" + "=" * 60)
print("TREATMENT ASSIGNMENT BY SCHOOL")
print("=" * 60)

treatment_by_school = df.groupby(['school_id', 'treatment']).size().unstack(fill_value=0)
treatment_by_school.columns = ['control', 'treated']
treatment_by_school['total'] = treatment_by_school['control'] + treatment_by_school['treated']
treatment_by_school['pct_treated'] = (treatment_by_school['treated'] / treatment_by_school['total'] * 100).round(1)
treatment_by_school = treatment_by_school.reset_index()

print("\nTreatment counts by school:")
print(treatment_by_school.to_string(index=False))

# Check for school-level treatment assignment patterns
print("\nPct treated by school:")
print(f"  Min: {treatment_by_school['pct_treated'].min():.1f}%")
print(f"  Max: {treatment_by_school['pct_treated'].max():.1f}%")
print(f"  Mean: {treatment_by_school['pct_treated'].mean():.1f}%")
print(f"  Std: {treatment_by_school['pct_treated'].std():.1f}%")

# Check if treatment varies within schools
within_school_variation = treatment_by_school['pct_treated'].between(1, 99).sum()
print(f"\nSchools with both treatment and control: {within_school_variation}/{len(treatment_by_school)}")

if within_school_variation == len(treatment_by_school):
    print("  → All schools have within-school treatment variation")
elif within_school_variation > 0:
    print("  → Some schools have within-school treatment variation")
    print("  → Some schools are entirely control or treated (confounded with school effects)")
else:
    print("  → WARNING: No within-school treatment variation!")
    print("  → Treatment is confounded with school-level effects")

# School means
print("\n" + "=" * 60)
print("SCHOOL MEANS AND VARIATION")
print("=" * 60)

school_stats = df.groupby('school_id').agg({
    'score': ['mean', 'std', 'min', 'max', 'count']
}).round(2)
school_stats.columns = ['mean', 'std', 'min', 'max', 'n']
school_stats = school_stats.reset_index()

print("\nSchool-level score statistics:")
print(school_stats.to_string(index=False))

print(f"\nRange of school means: {school_stats['mean'].min():.2f} to {school_stats['mean'].max():.2f}")
print(f"Std of school means: {school_stats['mean'].std():.2f}")

# Save school-level summary
school_summary = treatment_by_school.merge(school_stats, on='school_id')
school_summary.to_csv(OUTPUT_DIR / "school_summary.csv", index=False)
print("\n✓ Saved school_summary.csv")
