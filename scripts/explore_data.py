from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_outcomes.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isna().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")


# ---------------------------------------------------------
# Target distribution
# ---------------------------------------------------------

class_order = ["Dropout", "Enrolled", "Graduate"]
class_counts = df["Target"].value_counts().reindex(class_order)
class_percent = df["Target"].value_counts(normalize=True).reindex(class_order) * 100

print("\nClass distribution:")
for outcome in class_order:
    print(f"{outcome:8s}: {class_counts[outcome]:4d} ({class_percent[outcome]:.2f}%)")

plt.figure(figsize=(7, 5))
class_counts.plot(kind="bar")
plt.title("Distribution of Student Outcomes")
plt.xlabel("Student Outcome")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=300)
plt.close()


# ---------------------------------------------------------
# Bivariate analysis
# ---------------------------------------------------------

numerical_features = [
    "Age at enrollment",
    "Admission grade",
    "Previous qualification (grade)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
]

print("\nMean values by student outcome:")
print(df.groupby("Target")[numerical_features].mean().round(2).reindex(class_order))

# Create the three numerical plots used most directly in the report.
plot_features = {
    "Age at enrollment": ("Age at Enrolment", "age_by_outcome.png"),
    "Admission grade": ("Admission Grade", "admission_grade_by_outcome.png"),
    "Curricular units 1st sem (approved)": (
        "First-Semester Approved Units",
        "first_sem_approved_by_outcome.png",
    ),
    "Curricular units 1st sem (grade)": (
        "First-Semester Grade",
        "first_sem_grade_by_outcome.png",
    ),
}

for feature, (title, filename) in plot_features.items():
    values = [df.loc[df["Target"] == outcome, feature] for outcome in class_order]
    plt.figure(figsize=(7, 5))
    plt.boxplot(values, tick_labels=class_order)
    plt.title(f"{title} by Student Outcome")
    plt.xlabel("Student Outcome")
    plt.ylabel(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


# ---------------------------------------------------------
# Selected categorical relationships
# ---------------------------------------------------------

categorical_checks = {
    "Tuition fees up to date": ["Not up to date", "Up to date"],
    "Debtor": ["Not debtor", "Debtor"],
    "Scholarship holder": ["No scholarship", "Scholarship holder"],
}

categorical_filenames = {
    "Tuition fees up to date": "tuition_status_by_outcome.png",
    "Debtor": "debtor_status_by_outcome.png",
    "Scholarship holder": "scholarship_by_outcome.png",
}

for feature, labels in categorical_checks.items():
    table = pd.crosstab(df[feature], df["Target"], normalize="index") * 100
    table.index = labels
    print(f"\nStudent outcomes by {feature.lower()} (%):")
    print(table[class_order].round(2))

    table[class_order].plot(kind="bar", stacked=True, figsize=(7, 5))
    plt.title(f"Student Outcomes by {feature}")
    plt.xlabel(feature)
    plt.ylabel("Percentage of Students")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / categorical_filenames[feature], dpi=300)
    plt.close()