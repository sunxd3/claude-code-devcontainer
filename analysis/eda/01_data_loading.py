"""
Auto-MPG Dataset: Data Loading and Quality Checks
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_PATH = Path("/workspace/analysis/data/auto-mpg.data")
OUTPUT_DIR = Path("/workspace/analysis/eda")

# Column names from the dataset description
columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
           'acceleration', 'model_year', 'origin', 'car_name']

# Read the fixed-width data
# The format is space-separated with tab before car_name
# horsepower has missing values marked as "?"
df = pd.read_csv(
    DATA_PATH,
    delim_whitespace=True,
    names=columns,
    na_values='?',
    quotechar='"'
)

print("="*60)
print("DATA LOADING AND BASIC OVERVIEW")
print("="*60)

print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumn types:")
print(df.dtypes)

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nLast 5 rows:")
print(df.tail())

# Check for parsing issues
print("\n" + "="*60)
print("DATA QUALITY CHECKS")
print("="*60)

# Missingness per column
print("\nMissingness by column:")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'missing_count': missing,
    'missing_pct': missing_pct
})
print(missing_df[missing_df['missing_count'] > 0])

# Check for any rows with multiple missing values
row_missing = df.isnull().sum(axis=1)
print(f"\nRows with any missing: {(row_missing > 0).sum()}")
print(f"Max missing values per row: {row_missing.max()}")

# Duplicates
print(f"\nDuplicate rows (all columns): {df.duplicated().sum()}")
print(f"Duplicate car names: {df['car_name'].duplicated().sum()}")

# Check for constant columns
print("\nColumns with single unique value (constant):")
for col in df.columns:
    if df[col].nunique() == 1:
        print(f"  - {col}")
if all(df[col].nunique() > 1 for col in df.columns):
    print("  None")

# Check numeric ranges for impossible values
print("\nValue ranges for numeric columns:")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    print(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, "
          f"unique={df[col].nunique()}")

# Check for zeros that might be suspicious
print("\nZero value counts:")
for col in numeric_cols:
    zero_count = (df[col] == 0).sum()
    if zero_count > 0:
        print(f"  {col}: {zero_count} zeros")
if all((df[col] == 0).sum() == 0 for col in numeric_cols):
    print("  No zeros in numeric columns")

# Check for negative values
print("\nNegative value counts:")
for col in numeric_cols:
    neg_count = (df[col] < 0).sum()
    if neg_count > 0:
        print(f"  {col}: {neg_count} negative values")
if all((df[col] < 0).sum() == 0 for col in numeric_cols):
    print("  No negative values in numeric columns")

# Check cylinders values
print(f"\nCylinders distribution:")
print(df['cylinders'].value_counts().sort_index())

# Check origin values
print(f"\nOrigin distribution:")
print(df['origin'].value_counts().sort_index())

# Check model year values
print(f"\nModel year distribution:")
print(df['model_year'].value_counts().sort_index())

# Save basic quality summary
quality_summary = pd.DataFrame({
    'column': df.columns,
    'dtype': df.dtypes.values,
    'n_missing': df.isnull().sum().values,
    'pct_missing': (df.isnull().sum() / len(df) * 100).round(2).values,
    'n_unique': df.nunique().values,
    'n_zeros': [(df[col] == 0).sum() if col in numeric_cols else np.nan for col in df.columns],
    'n_negative': [(df[col] < 0).sum() if col in numeric_cols else np.nan for col in df.columns]
})
quality_summary.to_csv(OUTPUT_DIR / 'quality_summary.csv', index=False)
print(f"\nSaved quality summary to {OUTPUT_DIR / 'quality_summary.csv'}")

# Save cleaned data for other scripts
df.to_csv(OUTPUT_DIR / 'auto_mpg_cleaned.csv', index=False)
print(f"Saved cleaned data to {OUTPUT_DIR / 'auto_mpg_cleaned.csv'}")
