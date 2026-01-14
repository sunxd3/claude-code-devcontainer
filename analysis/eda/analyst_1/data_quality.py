"""Data quality checks for student scores dataset."""
import pandas as pd

# Load data
data_path = "/home/user/claude-code-devcontainer/analysis/data/student_scores.csv"
df = pd.read_csv(data_path)

print("=" * 80)
print("DATA QUALITY CHECKS")
print("=" * 80)

# 1. Missingness per column
print("\n1. MISSINGNESS PER COLUMN")
print("-" * 40)
missing_counts = df.isnull().sum()
missing_pct = (missing_counts / len(df)) * 100
missing_df = pd.DataFrame({
    'column': missing_counts.index,
    'missing_count': missing_counts.values,
    'missing_pct': missing_pct.values
})
missing_df = missing_df[missing_df['missing_count'] > 0]
if len(missing_df) > 0:
    print(missing_df.to_string(index=False))
else:
    print("No missing values detected in any column.")

# 2. Missingness per row
print("\n2. MISSINGNESS PER ROW")
print("-" * 40)
rows_with_missing = df.isnull().any(axis=1).sum()
print(f"Rows with at least one missing value: {rows_with_missing} ({rows_with_missing/len(df)*100:.1f}%)")

# 3. Duplicates
print("\n3. DUPLICATES")
print("-" * 40)
# Full row duplicates
full_duplicates = df.duplicated().sum()
print(f"Full row duplicates: {full_duplicates}")

# Student ID duplicates
student_id_duplicates = df['student_id'].duplicated().sum()
print(f"Duplicate student IDs: {student_id_duplicates}")

# 4. Invalid values and constants
print("\n4. INVALID VALUES AND CONSTANTS")
print("-" * 40)

# Check for constant columns
for col in df.columns:
    n_unique = df[col].nunique()
    if n_unique == 1:
        print(f"WARNING: {col} is constant (only 1 unique value)")

# Check treatment values
treatment_values = sorted(df['treatment'].unique())
print(f"Treatment values: {treatment_values}")
if set(treatment_values) != {0, 1}:
    print(f"  WARNING: Expected binary (0,1), got {treatment_values}")

# Check score range
score_min, score_max = df['score'].min(), df['score'].max()
print(f"Score range: [{score_min:.2f}, {score_max:.2f}]")
if score_min < 0:
    print(f"  WARNING: Negative scores detected (min={score_min:.2f})")

# Check for sentinel values in score
sentinel_check = df['score'].isin([-999, -99, 999, 9999]).sum()
if sentinel_check > 0:
    print(f"  WARNING: {sentinel_check} potential sentinel values in score")

# 5. Type issues
print("\n5. TYPE CONSISTENCY")
print("-" * 40)
print(f"student_id type: {df['student_id'].dtype}")
student_id_pattern = r'^S\d+_\d+$'
is_valid_pattern = df['student_id'].str.match(student_id_pattern).all()
print(f"  - Check if numeric: {is_valid_pattern}")
print(f"school_id type: {df['school_id'].dtype}")
print(f"  - All integers: {(df['school_id'] == df['school_id'].astype(int)).all()}")
print(f"treatment type: {df['treatment'].dtype}")
print(f"  - All integers: {(df['treatment'] == df['treatment'].astype(int)).all()}")
print(f"score type: {df['score'].dtype}")

# 6. School-level consistency
print("\n6. HIERARCHICAL STRUCTURE CHECKS")
print("-" * 40)
# Check school_id and school_name mapping
school_mapping = df.groupby('school_id')['school_name'].nunique()
inconsistent_schools = school_mapping[school_mapping > 1]
if len(inconsistent_schools) > 0:
    print("WARNING: Inconsistent school_id to school_name mapping:")
    print(inconsistent_schools)
else:
    print("school_id to school_name mapping is consistent")

# Students per school
students_per_school = df.groupby('school_id').size()
print("\nStudents per school:")
print(students_per_school.to_string())

# Create quality summary table
quality_summary = pd.DataFrame({
    'column': df.columns,
    'n_rows': len(df),
    'n_missing': [df[col].isnull().sum() for col in df.columns],
    'pct_missing': [df[col].isnull().sum() / len(df) * 100 for col in df.columns],
    'n_unique': [df[col].nunique() for col in df.columns],
    'dtype': [str(df[col].dtype) for col in df.columns]
})

# Save to CSV
output_path = "/home/user/claude-code-devcontainer/analysis/eda/analyst_1/quality_summary.csv"
quality_summary.to_csv(output_path, index=False)
print("\nQuality summary saved to: quality_summary.csv")
