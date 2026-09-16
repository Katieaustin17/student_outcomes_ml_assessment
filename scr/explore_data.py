from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Project folders
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_outcomes.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())


# ---------------------------------------------------------
# Basic dataset information
# ---------------------------------------------------------

print("\nColumn names:")
for column in df.columns:
    print(column)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ---------------------------------------------------------
# Target variable
# ---------------------------------------------------------

print("\nTarget counts:")
print(df["Target"].value_counts())

print("\nTarget percentages:")
target_percentages = (
    df["Target"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(target_percentages)


# ---------------------------------------------------------
# Plot target distribution
# ---------------------------------------------------------

target_counts = df["Target"].value_counts()

plt.figure(figsize=(8, 5))

target_counts.plot(kind="bar")

plt.title("Distribution of Student Outcomes")
plt.xlabel("Student Outcome")
plt.ylabel("Number of Students")

plt.xticks(rotation=0)

plt.tight_layout()

figure_path = FIGURES_DIR / "target_distribution.png"

plt.savefig(figure_path, dpi=300)
plt.show()

print(f"\nTarget distribution figure saved to:\n{figure_path}")


# ---------------------------------------------------------
# Numerical summary
# ---------------------------------------------------------

print("\nNumerical summary:")
print(df.describe().T)