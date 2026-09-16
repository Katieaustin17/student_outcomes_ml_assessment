from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


# ---------------------------------------------------------
# Project folders
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Download dataset from UCI Machine Learning Repository
# ---------------------------------------------------------

print("Downloading dataset from UCI...")

student_data = fetch_ucirepo(id=697)

# Predictor variables
X = student_data.data.features.copy()

# Target variable
y = student_data.data.targets.copy()


# ---------------------------------------------------------
# Combine predictors and target into one DataFrame
# ---------------------------------------------------------

df = pd.concat([X, y], axis=1)


# ---------------------------------------------------------
# Save a local copy
# ---------------------------------------------------------

data_path = RAW_DATA_DIR / "student_outcomes.csv"
df.to_csv(data_path, index=False)

# Also save the UCI variable information.
# This will be useful later because several categorical
# variables are represented using numeric codes.
variables_path = RAW_DATA_DIR / "variable_information.csv"
student_data.variables.to_csv(variables_path, index=False)


# ---------------------------------------------------------
# Initial checks
# ---------------------------------------------------------

print("\nDataset downloaded successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nFirst five rows:")
print(df.head())

print("\nTarget distribution:")
print(df["Target"].value_counts())

print("\nMissing values:")
print(df.isnull().sum().sum())

print(f"\nDataset saved to:\n{data_path}")