"""Load data and perform mandatory quality checks."""

from pathlib import Path

import pandas as pd

# Paths
DATA_PATH = Path("/home/user/claude-code-devcontainer/analysis/data/student_scores.csv")
OUTPUT_DIR = Path("/home/user/claude-code-devcontainer/analysis/eda/analyst_2")

# Load data
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATA STRUCTURE")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nColumn types:")
print(df.dtypes)
print("\nFirst few rows:")
print(df.head(10))

# Basic counts
print("\n" + "=" * 60)
print("BASIC COUNTS")
print("=" * 60)
print(f"Total students: {len(df)}")
print(f"Total schools: {df['school_id'].nunique()}")
print(f"Treatment counts: {df['treatment'].value_counts().to_dict()}")

# Data quality checks
print("\n" + "=" * 60)
print("DATA QUALITY CHECKS")
print("=" * 60)

# Missingness
print("\n1. MISSINGNESS")
missing_per_col = df.isnull().sum()
missing_pct_col = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'missing_count': missing_per_col,
    'missing_pct': missing_pct_col
})
print(missing_df)

# Per-row missingness
missing_per_row = df.isnull().sum(axis=1)
print(f"\nRows with any missing: {(missing_per_row > 0).sum()}")
print(f"Max missing values in a row: {missing_per_row.max()}")

# Duplicates
print("\n2. DUPLICATES")
dup_rows = df.duplicated().sum()
dup_student_ids = df['student_id'].duplicated().sum()
print(f"Duplicate rows: {dup_rows}")
print(f"Duplicate student_ids: {dup_student_ids}")

# Invalid values
print("\n3. INVALID VALUES")

# Check for constant columns
for col in df.columns:
    if df[col].nunique() == 1:
        print(f"WARNING: Constant column '{col}' with value {df[col].iloc[0]}")

# Check score range
print("\nScore statistics:")
print(f"  Min: {df['score'].min():.2f}")
print(f"  Max: {df['score'].max():.2f}")
print(f"  Mean: {df['score'].mean():.2f}")
print(f"  Std: {df['score'].std():.2f}")

# Check for sentinel values in score
sentinels = [-999, -99, 999, 9999]
sentinel_count = df['score'].isin(sentinels).sum()
if sentinel_count > 0:
    print(f"WARNING: {sentinel_count} sentinel values found in score")

# Check treatment values
print(f"\nTreatment unique values: {sorted(df['treatment'].unique())}")
if not set(df['treatment'].unique()).issubset({0, 1}):
    print("WARNING: Treatment has values other than 0 and 1")

# Type issues
print("\n4. TYPE ISSUES")
# Check if numerics are actually numeric
try:
    pd.to_numeric(df['score'])
    print("Score: properly numeric")
except (ValueError, TypeError):
    print("WARNING: Score column has non-numeric values")

try:
    pd.to_numeric(df['treatment'])
    print("Treatment: properly numeric")
except (ValueError, TypeError):
    print("WARNING: Treatment column has non-numeric values")

try:
    pd.to_numeric(df['school_id'])
    print("School_id: properly numeric")
except (ValueError, TypeError):
    print("WARNING: School_id column has non-numeric values")

# Save quality summary
quality_summary = pd.DataFrame([
    {'check': 'Total rows', 'value': len(df)},
    {'check': 'Total columns', 'value': len(df.columns)},
    {'check': 'Missing values', 'value': df.isnull().sum().sum()},
    {'check': 'Duplicate rows', 'value': dup_rows},
    {'check': 'Duplicate student_ids', 'value': dup_student_ids},
    {'check': 'Constant columns', 'value': sum(df[col].nunique() == 1 for col in df.columns)},
    {'check': 'Score min', 'value': f"{df['score'].min():.2f}"},
    {'check': 'Score max', 'value': f"{df['score'].max():.2f}"},
])

quality_summary.to_csv(OUTPUT_DIR / "quality_summary.csv", index=False)
print("\n✓ Saved quality_summary.csv")

# Save univariate summary
univariate = pd.DataFrame([
    {
        'column': col,
        'dtype': str(df[col].dtype),
        'n_unique': df[col].nunique(),
        'n_missing': df[col].isnull().sum(),
        'pct_missing': f"{df[col].isnull().sum() / len(df) * 100:.2f}%"
    }
    for col in df.columns
])

# Add numeric stats for numeric columns
for col in ['school_id', 'treatment', 'score']:
    univariate.loc[univariate['column'] == col, 'mean'] = f"{df[col].mean():.2f}"
    univariate.loc[univariate['column'] == col, 'std'] = f"{df[col].std():.2f}"
    univariate.loc[univariate['column'] == col, 'min'] = f"{df[col].min():.2f}"
    univariate.loc[univariate['column'] == col, 'max'] = f"{df[col].max():.2f}"

univariate.to_csv(OUTPUT_DIR / "univariate_summary.csv", index=False)
print("✓ Saved univariate_summary.csv")
