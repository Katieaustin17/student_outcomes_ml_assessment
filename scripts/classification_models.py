from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

from time import perf_counter
from sklearn.base import clone

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



# Load and prepare data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_outcomes.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Second-semester variables are excluded so predictions represent
# information available by the end of the first semester.
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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set:     {len(X_test)} samples (held out until final evaluation)")


# Feature groups and preprocessing


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

logistic_preprocessor = ColumnTransformer(
    [
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numerical", StandardScaler(), numerical_features),
        ("binary", "passthrough", binary_features),
    ]
)

forest_preprocessor = ColumnTransformer(
    [
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numerical", "passthrough", numerical_features),
        ("binary", "passthrough", binary_features),
    ]
)



# Candidate models


models = {
    "Logistic Regression": Pipeline(
        [
            ("preprocessor", logistic_preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    ),
    "Random Forest": Pipeline(
        [
            ("preprocessor", forest_preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
}

# Stratified 5-fold CV mirrors the taught classification workflow and keeps class proportions similar in every fold.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
}



# 5-fold cross-validation model comparison


comparison_rows = []

for name, model in models.items():
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)

    fold_results = pd.DataFrame(
        {
            "Accuracy": scores["test_accuracy"],
            "Precision_macro": scores["test_precision_macro"],
            "Recall_macro": scores["test_recall_macro"],
            "F1_macro": scores["test_f1_macro"],
        }
    )

    output_name = (
        "logistic_regression_cv_results.csv"
        if name == "Logistic Regression"
        else "random_forest_cv_results.csv"
    )
    fold_results.to_csv(OUTPUT_DIR / output_name, index=False)

    comparison_rows.append(
        {
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Macro Precision": scores["test_precision_macro"].mean(),
            "Macro Recall": scores["test_recall_macro"].mean(),
            "Macro F1": scores["test_f1_macro"].mean(),
            "F1 Std Dev": scores["test_f1_macro"].std(),
        }
    )

comparison = pd.DataFrame(comparison_rows)

print("\n5-Fold Stratified Cross-Validation")
print("=" * 78)
print(comparison.round(3).to_string(index=False))
print("=" * 78)



# Hyperparameter tuning with cross-validation


logistic_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
}

logistic_search = GridSearchCV(
    models["Logistic Regression"],
    logistic_grid,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    return_train_score=True,
)
logistic_search.fit(X_train, y_train)

logistic_results = pd.DataFrame(logistic_search.cv_results_)
logistic_summary = logistic_results[
    [
        "param_classifier__C",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
    ]
].copy()
logistic_summary.columns = [
    "C",
    "Mean CV Macro F1",
    "CV Standard Deviation",
    "Mean Training Macro F1",
]
logistic_summary.to_csv(
    OUTPUT_DIR / "logistic_regression_grid_search_results.csv", index=False
)

print("\nLogistic Regression tuning")
print(f"Best parameters: {logistic_search.best_params_}")
print(f"Best CV macro F1: {logistic_search.best_score_:.3f}")

random_forest_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_leaf": [1, 2, 4],
}

forest_search = GridSearchCV(
    models["Random Forest"],
    random_forest_grid,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    return_train_score=True,
)
forest_search.fit(X_train, y_train)

forest_results = pd.DataFrame(forest_search.cv_results_)
forest_results.to_csv(
    OUTPUT_DIR / "random_forest_grid_search_results.csv", index=False
)

best_rf_row = forest_results.loc[forest_results["rank_test_score"] == 1].iloc[0]

print("\nRandom Forest tuning")
print(f"Best parameters: {forest_search.best_params_}")
print(f"Best CV macro F1: {forest_search.best_score_:.3f}")
print(f"Mean training macro F1: {best_rf_row['mean_train_score']:.3f}")
print(f"Mean CV fit time: {best_rf_row['mean_fit_time']:.3f} seconds")


# Class-level cross-validation comparison


class_names = ["Dropout", "Enrolled", "Graduate"]

logistic_cv_pred = cross_val_predict(
    logistic_search.best_estimator_, X_train, y_train, cv=cv, n_jobs=-1
)
forest_cv_pred = cross_val_predict(
    forest_search.best_estimator_, X_train, y_train, cv=cv, n_jobs=-1
)

print("\nLogistic Regression - out-of-fold classification report")
print(classification_report(y_train, logistic_cv_pred, labels=class_names))

print("Random Forest - out-of-fold classification report")
print(classification_report(y_train, forest_cv_pred, labels=class_names))

for name, predictions, csv_name, figure_name in [
    (
        "Logistic Regression",
        logistic_cv_pred,
        "logistic_cv_confusion_matrix.csv",
        "logistic_cv_confusion_matrix.png",
    ),
    (
        "Random Forest",
        forest_cv_pred,
        "random_forest_cv_confusion_matrix.csv",
        "random_forest_cv_confusion_matrix.png",
    ),
]:
    cm = confusion_matrix(y_train, predictions, labels=class_names)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        OUTPUT_DIR / csv_name
    )
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
    plt.title(f"{name} - Cross-Validated Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / figure_name, dpi=300)
    plt.close()


# Final Logistic Regression and Random Forest evaluation on held-out test set

final_model = logistic_search.best_estimator_

start = time()
final_model.fit(X_train, y_train)
training_time = time() - start

start = time()
y_pred = final_model.predict(X_test)
prediction_time = time() - start


# Random Forest timing for comparison
final_rf_model = forest_search.best_estimator_

start = time()
final_rf_model.fit(X_train, y_train)
rf_training_time = time() - start

start = time()
rf_y_pred = final_rf_model.predict(X_test)
rf_prediction_time = time() - start

accuracy = accuracy_score(y_test, y_pred)
macro_precision = precision_score(y_test, y_pred, average="macro")
macro_recall = recall_score(y_test, y_pred, average="macro")
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("\nModel Timing Comparison")
print("=" * 50)
print(f"Logistic Regression training time: {training_time:.3f} seconds")
print(f"Random Forest training time:       {rf_training_time:.3f} seconds")
print()
print(f"Logistic Regression prediction time: {prediction_time:.3f} seconds")
print(f"Random Forest prediction time:       {rf_prediction_time:.3f} seconds")
print("=" * 50)

print("\nFinal Test Set Evaluation")
print("=" * 50)
print(f"Accuracy:        {accuracy:.3f}")
print(f"Macro Precision: {macro_precision:.3f}")
print(f"Macro Recall:    {macro_recall:.3f}")
print(f"Macro F1:        {macro_f1:.3f}")
print(f"Training time:   {training_time:.3f} seconds")
print(f"Prediction time: {prediction_time:.3f} seconds")
print("=" * 50)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=class_names))

report = pd.DataFrame(
    classification_report(
        y_test,
        y_pred,
        labels=class_names,
        output_dict=True,
    )
).transpose()
report.to_csv(OUTPUT_DIR / "final_test_classification_report.csv")

metrics = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
            "Training time (seconds)",
            "Prediction time (seconds)",
        ],
        "Value": [
            accuracy,
            macro_precision,
            macro_recall,
            macro_f1,
            training_time,
            prediction_time,
        ],
    }
)
metrics.to_csv(OUTPUT_DIR / "final_test_metrics.csv", index=False)

cm = confusion_matrix(y_test, y_pred, labels=class_names)
pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
    OUTPUT_DIR / "final_test_confusion_matrix.csv"
)

ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
plt.title("Final Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "final_test_confusion_matrix.png", dpi=300)
plt.close()



# Logistic Regression coefficients


preprocessor = final_model.named_steps["preprocessor"]
classifier = final_model.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()
feature_names = [name.split("__", 1)[-1] for name in feature_names]

coefficient_rows = []
for class_name, coefficients in zip(classifier.classes_, classifier.coef_):
    for feature, coefficient in zip(feature_names, coefficients):
        coefficient_rows.append(
            {
                "Feature": feature,
                "Coefficient": coefficient,
                "Class": class_name,
            }
        )

coefficient_df = pd.DataFrame(coefficient_rows)
coefficient_df.to_csv(
    OUTPUT_DIR / "logistic_regression_coefficients.csv", index=False
)

print("\nSelected Logistic Regression coefficients:")
selected = coefficient_df[
    coefficient_df["Feature"].isin(
        ["Curricular units 1st sem (approved)", "Tuition fees up to date"]
    )
]
print(selected.round(3).to_string(index=False))


# Plot the strongest Dropout coefficient associations.
dropout_coefficients = coefficient_df[
    coefficient_df["Class"] == "Dropout"
].copy()
dropout_coefficients["Absolute Coefficient"] = dropout_coefficients[
    "Coefficient"
].abs()
top_dropout = dropout_coefficients.nlargest(10, "Absolute Coefficient").sort_values(
    "Coefficient"
)

plt.figure(figsize=(8, 6))
plt.barh(top_dropout["Feature"], top_dropout["Coefficient"])
plt.axvline(0, linewidth=1)
plt.xlabel("Coefficient")
plt.title("Strongest Logistic Regression Associations with Dropout")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "dropout_logistic_coefficients.png", dpi=300)
plt.close()

