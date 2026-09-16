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


# ---------------------------------------------------------
# 1. Numerical features by student outcome
# ---------------------------------------------------------

selected_numerical = [
    "Age at enrollment",
    "Admission grade",
    "Previous qualification (grade)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
]

print("\nMean values by student outcome:")

numerical_summary = (
    df.groupby("Target")[selected_numerical]
    .mean()
    .round(2)
)

print(numerical_summary)


# ---------------------------------------------------------
# 2. Age at enrolment by outcome
# ---------------------------------------------------------

outcome_order = ["Dropout", "Enrolled", "Graduate"]

age_data = [
    df.loc[df["Target"] == outcome, "Age at enrollment"]
    for outcome in outcome_order
]

plt.figure(figsize=(8, 5))

plt.boxplot(age_data, tick_labels=outcome_order)

plt.title("Age at Enrolment by Student Outcome")
plt.xlabel("Student Outcome")
plt.ylabel("Age at Enrolment")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "age_by_outcome.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 3. Admission grade by outcome
# ---------------------------------------------------------

admission_data = [
    df.loc[df["Target"] == outcome, "Admission grade"]
    for outcome in outcome_order
]

plt.figure(figsize=(8, 5))

plt.boxplot(admission_data, tick_labels=outcome_order)

plt.title("Admission Grade by Student Outcome")
plt.xlabel("Student Outcome")
plt.ylabel("Admission Grade")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "admission_grade_by_outcome.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 4. First-semester approved units by outcome
# ---------------------------------------------------------

approved_data = [
    df.loc[
        df["Target"] == outcome,
        "Curricular units 1st sem (approved)"
    ]
    for outcome in outcome_order
]

plt.figure(figsize=(8, 5))

plt.boxplot(approved_data, tick_labels=outcome_order)

plt.title("First-Semester Approved Units by Student Outcome")
plt.xlabel("Student Outcome")
plt.ylabel("Approved Curricular Units")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "first_sem_approved_by_outcome.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 5. First-semester grade by outcome
# ---------------------------------------------------------

grade_data = [
    df.loc[
        df["Target"] == outcome,
        "Curricular units 1st sem (grade)"
    ]
    for outcome in outcome_order
]

plt.figure(figsize=(8, 5))

plt.boxplot(grade_data, tick_labels=outcome_order)

plt.title("First-Semester Grade by Student Outcome")
plt.xlabel("Student Outcome")
plt.ylabel("Average First-Semester Grade")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "first_sem_grade_by_outcome.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 6. Tuition fee status and outcome
# ---------------------------------------------------------

tuition_table = pd.crosstab(
    df["Tuition fees up to date"],
    df["Target"],
    normalize="index"
).mul(100).round(2)

tuition_table.index = ["Not up to date", "Up to date"]

print("\nStudent outcomes by tuition fee status (%):")
print(tuition_table)

tuition_table[outcome_order].plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5)
)

plt.title("Student Outcomes by Tuition Fee Status")
plt.xlabel("Tuition Fee Status")
plt.ylabel("Percentage of Students")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "tuition_status_by_outcome.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 7. Debtor status and outcome
# ---------------------------------------------------------

debtor_table = pd.crosstab(
    df["Debtor"],
    df["Target"],
    normalize="index"
).mul(100).round(2)

debtor_table.index = ["Not debtor", "Debtor"]

print("\nStudent outcomes by debtor status (%):")
print(debtor_table)

debtor_table[outcome_order].plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5)
)

plt.title("Student Outcomes by Debtor Status")
plt.xlabel("Debtor Status")
plt.ylabel("Percentage of Students")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "debtor_status_by_outcome.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 8. Scholarship status and outcome
# ---------------------------------------------------------

scholarship_table = pd.crosstab(
    df["Scholarship holder"],
    df["Target"],
    normalize="index"
).mul(100).round(2)

scholarship_table.index = [
    "No scholarship",
    "Scholarship holder"
]

print("\nStudent outcomes by scholarship status (%):")
print(scholarship_table)

scholarship_table[outcome_order].plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5)
)

plt.title("Student Outcomes by Scholarship Status")
plt.xlabel("Scholarship Status")
plt.ylabel("Percentage of Students")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "scholarship_by_outcome.png",
    dpi=300
)

plt.show()