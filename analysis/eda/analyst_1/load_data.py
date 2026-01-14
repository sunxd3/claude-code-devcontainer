"""Load and inspect student scores dataset."""
import pandas as pd

# Load data
data_path = "/home/user/claude-code-devcontainer/analysis/data/student_scores.csv"
df = pd.read_csv(data_path)

print("=" * 80)
print("DATA LOADING AND PARSING")
print("=" * 80)

print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nColumn names and types:")
print(df.dtypes)

print("\nFirst few rows:")
print(df.head(10))

print("\nBasic statistics:")
print(df.describe(include='all'))

print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
