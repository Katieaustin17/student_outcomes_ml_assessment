from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


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
# Remove second-semester academic variables
#
# The proposed model is intended to make predictions
# after the first semester, so second-semester information
# would not yet be available.
# ---------------------------------------------------------

second_semester_features = [
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
]

df_model = df.drop(columns=second_semester_features)


# ---------------------------------------------------------
# Separate predictors and target
# ---------------------------------------------------------

X = df_model.drop(columns="Target")
y = df_model["Target"]


# ---------------------------------------------------------
# Train/test split
#
# Stratify ensures that Dropout, Enrolled and Graduate
# remain in similar proportions in both datasets.
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("Original dataset:")
print(df.shape)

print("\nModelling dataset:")
print(df_model.shape)

print("\nNumber of predictor features:")
print(X.shape[1])

print("\nTraining set:")
print(X_train.shape)

print("\nTest set:")
print(X_test.shape)


# ---------------------------------------------------------
# Check class distributions
# ---------------------------------------------------------

print("\nTraining target distribution (%):")
print(
    y_train
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nTest target distribution (%):")
print(
    y_test
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)