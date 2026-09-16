from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
FIGURES_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


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

X = df_model.drop(columns="Target")
y = df_model["Target"]


# ---------------------------------------------------------
# Reproduce original train/test split
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
# Final Logistic Regression model
# ---------------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                C=1,
                max_iter=2000,
                random_state=42,
            ),
        ),
    ]
)

model.fit(X_train, y_train)


# ---------------------------------------------------------
# Extract transformed feature names
# ---------------------------------------------------------

feature_names = (
    model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)


# Make names easier to read
clean_feature_names = [
    name
    .replace("categorical__", "")
    .replace("numerical__", "")
    .replace("binary__", "")
    for name in feature_names
]


# ---------------------------------------------------------
# Extract coefficients
# ---------------------------------------------------------

classifier = model.named_steps["classifier"]

classes = classifier.classes_
coefficients = classifier.coef_


# ---------------------------------------------------------
# Save coefficients for every class
# ---------------------------------------------------------

all_coefficients = []

for class_index, class_name in enumerate(classes):

    class_df = pd.DataFrame(
        {
            "Feature": clean_feature_names,
            "Coefficient": coefficients[class_index],
            "Class": class_name,
        }
    )

    all_coefficients.append(class_df)


coefficient_df = pd.concat(
    all_coefficients,
    ignore_index=True,
)

coefficient_df.to_csv(
    OUTPUT_DIR / "logistic_regression_coefficients.csv",
    index=False,
)


# ---------------------------------------------------------
# Display strongest coefficients for each class
# ---------------------------------------------------------

for class_name in classes:

    class_results = coefficient_df[
        coefficient_df["Class"] == class_name
    ].sort_values(
        "Coefficient",
        ascending=False,
    )

    print(f"\n{class_name.upper()}")
    print("-" * 50)

    print("\nLargest positive coefficients:")
    print(
        class_results[
            ["Feature", "Coefficient"]
        ].head(10).to_string(index=False)
    )

    print("\nLargest negative coefficients:")
    print(
        class_results[
            ["Feature", "Coefficient"]
        ].tail(10).sort_values(
            "Coefficient"
        ).to_string(index=False)
    )


# ---------------------------------------------------------
# Plot strongest Dropout coefficient associations
# ---------------------------------------------------------

dropout_results = coefficient_df[
    coefficient_df["Class"] == "Dropout"
].copy()

dropout_results["Absolute coefficient"] = (
    dropout_results["Coefficient"].abs()
)

top_dropout = (
    dropout_results
    .sort_values(
        "Absolute coefficient",
        ascending=False,
    )
    .head(12)
    .sort_values("Coefficient")
)


plt.figure(figsize=(10, 7))

plt.barh(
    top_dropout["Feature"],
    top_dropout["Coefficient"],
)

plt.axvline(
    x=0,
    linewidth=1,
)

plt.title(
    "Strongest Logistic Regression Coefficients\n"
    "for the Dropout Outcome"
)

plt.xlabel("Coefficient")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "dropout_logistic_coefficients.png",
    dpi=300,
)

plt.show()