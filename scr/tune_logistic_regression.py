from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# Project folders
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "student_outcomes.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)


# ---------------------------------------------------------
# Remove second-semester variables
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
# Predictors and target
# ---------------------------------------------------------

X = df_model.drop(columns="Target")
y = df_model["Target"]


# ---------------------------------------------------------
# Same train/test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# ---------------------------------------------------------
# Feature groups
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
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features,
        ),
        (
            "binary",
            "passthrough",
            binary_features,
        ),
    ]
)


# ---------------------------------------------------------
# Logistic Regression pipeline
# ---------------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),
    ]
)


# ---------------------------------------------------------
# Regularisation values to test
# ---------------------------------------------------------

parameter_grid = {
    "classifier__C": [
        0.01,
        0.1,
        1,
        10,
        100,
    ]
}


# ---------------------------------------------------------
# Stratified cross-validation
# ---------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# ---------------------------------------------------------
# Grid search
# ---------------------------------------------------------

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    return_train_score=True,
)


print("Running Logistic Regression grid search...")

grid_search.fit(X_train, y_train)


# ---------------------------------------------------------
# Best result
# ---------------------------------------------------------

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest mean cross-validation macro F1:")
print(round(grid_search.best_score_, 3))


# ---------------------------------------------------------
# Display results for each C value
# ---------------------------------------------------------

results = pd.DataFrame(grid_search.cv_results_)

summary = results[
    [
        "param_classifier__C",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
    ]
].copy()

summary.columns = [
    "C",
    "Mean CV Macro F1",
    "CV Standard Deviation",
    "Mean Training Macro F1",
]

print("\nResults by regularisation strength:")
print(summary.round(3))


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results_path = (
    OUTPUT_DIR
    / "logistic_regression_grid_search_results.csv"
)

summary.to_csv(
    results_path,
    index=False,
)

print(
    f"\nGrid-search results saved to:\n"
    f"{results_path}"
)