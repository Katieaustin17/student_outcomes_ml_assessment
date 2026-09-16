from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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
# Load dataset
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
# Separate predictors and target
# ---------------------------------------------------------

X = df_model.drop(columns="Target")
y = df_model["Target"]


# ---------------------------------------------------------
# Same train/test split as Logistic Regression
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
#
# Random Forest does not require feature scaling.
#
# Categorical variables are one-hot encoded.
# Numerical and binary variables are passed through.
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
            "passthrough",
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
# Random Forest pipeline
# ---------------------------------------------------------

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


# ---------------------------------------------------------
# Stratified 5-fold cross-validation
# ---------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# ---------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------

scoring = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
}


# ---------------------------------------------------------
# Run CV on training data only
# ---------------------------------------------------------

cv_results = cross_validate(
    random_forest_model,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    return_train_score=False,
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

results_df = pd.DataFrame(cv_results)

print("\nRandom Forest - 5-Fold Cross-Validation")
print("------------------------------------------")

for metric in [
    "test_accuracy",
    "test_precision_macro",
    "test_recall_macro",
    "test_f1_macro",
]:
    mean_score = results_df[metric].mean()
    std_score = results_df[metric].std()

    print(
        f"{metric}: "
        f"{mean_score:.3f} "
        f"(+/- {std_score:.3f})"
    )


# ---------------------------------------------------------
# Individual fold results
# ---------------------------------------------------------

print("\nIndividual fold results:")

fold_results = pd.DataFrame(
    {
        "Accuracy": results_df["test_accuracy"],
        "Precision_macro": results_df["test_precision_macro"],
        "Recall_macro": results_df["test_recall_macro"],
        "F1_macro": results_df["test_f1_macro"],
    }
)

print(fold_results.round(3))


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results_path = (
    OUTPUT_DIR
    / "random_forest_cv_results.csv"
)

fold_results.to_csv(
    results_path,
    index=False,
)

print(
    f"\nCross-validation results saved to:\n"
    f"{results_path}"
)