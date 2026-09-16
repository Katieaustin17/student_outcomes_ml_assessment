from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Project folders
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_outcomes.csv"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)


# ---------------------------------------------------------
# Define feature groups
# ---------------------------------------------------------

categorical_features = [
    "Marital Status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

binary_features = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

numerical_features = [
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

target = "Target"


# ---------------------------------------------------------
# Check that every predictor has been classified
# ---------------------------------------------------------

classified_features = (
    categorical_features
    + binary_features
    + numerical_features
)

predictor_features = [
    column for column in df.columns
    if column != target
]

unclassified = set(predictor_features) - set(classified_features)
unexpected = set(classified_features) - set(predictor_features)


print("Categorical features:")
print(len(categorical_features))
print(categorical_features)

print("\nBinary features:")
print(len(binary_features))
print(binary_features)

print("\nNumerical features:")
print(len(numerical_features))
print(numerical_features)

print("\nTotal predictor features:")
print(len(predictor_features))

print("\nFeatures not classified:")
print(unclassified)

print("\nUnexpected features:")
print(unexpected)


# ---------------------------------------------------------
# Inspect categorical variables
# ---------------------------------------------------------

print("\nNumber of unique values in categorical features:")

for column in categorical_features:
    print(f"{column}: {df[column].nunique()}")


print("\nBinary feature values:")

for column in binary_features:
    print(f"{column}: {sorted(df[column].unique())}")