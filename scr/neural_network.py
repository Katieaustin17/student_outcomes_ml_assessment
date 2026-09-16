from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)


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
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)


# ---------------------------------------------------------
# Remove second-semester information
#
# Same prediction point used by the classical models:
# end of first semester.
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
# Reproduce the ORIGINAL 80/20 split
#
# The 20% test set has already been used for the final
# Logistic Regression evaluation, so it will NOT be used
# for this neural network experiment.
# ---------------------------------------------------------

X_development, X_unused_test, y_development, y_unused_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
)


# ---------------------------------------------------------
# Split development data into ANN training and validation
#
# 80% of the original training data = ANN training
# 20% of the original training data = ANN validation
# ---------------------------------------------------------

X_ann_train, X_ann_val, y_ann_train, y_ann_val = (
    train_test_split(
        X_development,
        y_development,
        test_size=0.20,
        random_state=42,
        stratify=y_development,
    )
)


print("ANN training set:")
print(X_ann_train.shape)

print("\nANN validation set:")
print(X_ann_val.shape)


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
# Neural networks benefit from scaled numerical inputs.
# Categorical variables are one-hot encoded.
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
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


# Fit preprocessing ONLY to ANN training data
X_ann_train_processed = preprocessor.fit_transform(
    X_ann_train
)

X_ann_val_processed = preprocessor.transform(
    X_ann_val
)


print(
    "\nNumber of processed input features:",
    X_ann_train_processed.shape[1]
)


# ---------------------------------------------------------
# Encode target
#
# Softmax output neurons:
# 0 = Dropout
# 1 = Enrolled
# 2 = Graduate
# ---------------------------------------------------------

target_mapping = {
    "Dropout": 0,
    "Enrolled": 1,
    "Graduate": 2,
}

target_names = [
    "Dropout",
    "Enrolled",
    "Graduate",
]

y_ann_train_encoded = (
    y_ann_train
    .map(target_mapping)
    .to_numpy()
)

y_ann_val_encoded = (
    y_ann_val
    .map(target_mapping)
    .to_numpy()
)


# ---------------------------------------------------------
# Build neural network
# ---------------------------------------------------------

model = Sequential(
    [
        Input(
            shape=(
                X_ann_train_processed.shape[1],
            )
        ),

        Dense(
            64,
            activation="relu",
        ),

        Dropout(0.30),

        Dense(
            32,
            activation="relu",
        ),

        Dropout(0.30),

        Dense(
            3,
            activation="softmax",
        ),
    ]
)


# ---------------------------------------------------------
# Compile model
# ---------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


print("\nNeural Network Architecture")
print("---------------------------")

model.summary()


# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------

history = model.fit(
    X_ann_train_processed,
    y_ann_train_encoded,
    validation_data=(
        X_ann_val_processed,
        y_ann_val_encoded,
    ),
    epochs=40,
    batch_size=32,
    verbose=1,
)


# ---------------------------------------------------------
# Validation predictions
# ---------------------------------------------------------

validation_probabilities = model.predict(
    X_ann_val_processed
)

validation_predictions = np.argmax(
    validation_probabilities,
    axis=1,
)


# ---------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_ann_val_encoded,
    validation_predictions,
)

macro_precision = precision_score(
    y_ann_val_encoded,
    validation_predictions,
    average="macro",
)

macro_recall = recall_score(
    y_ann_val_encoded,
    validation_predictions,
    average="macro",
)

macro_f1 = f1_score(
    y_ann_val_encoded,
    validation_predictions,
    average="macro",
)


print("\nNEURAL NETWORK VALIDATION PERFORMANCE")
print("-------------------------------------")

print(f"Accuracy:        {accuracy:.3f}")
print(f"Macro Precision: {macro_precision:.3f}")
print(f"Macro Recall:    {macro_recall:.3f}")
print(f"Macro F1:        {macro_f1:.3f}")


# ---------------------------------------------------------
# Classification report
# ---------------------------------------------------------

print("\nClassification Report")
print("---------------------")

print(
    classification_report(
        y_ann_val_encoded,
        validation_predictions,
        labels=[0, 1, 2],
        target_names=target_names,
        digits=3,
    )
)


# ---------------------------------------------------------
# Save metrics
# ---------------------------------------------------------

ann_metrics = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
        ],
        "Value": [
            accuracy,
            macro_precision,
            macro_recall,
            macro_f1,
        ],
    }
)

ann_metrics.to_csv(
    OUTPUT_DIR / "ann_validation_metrics.csv",
    index=False,
)


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    y_ann_val_encoded,
    validation_predictions,
    labels=[0, 1, 2],
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=target_names,
)

display.plot()

plt.title(
    "Artificial Neural Network\n"
    "Validation Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "ann_validation_confusion_matrix.png",
    dpi=300,
)

plt.show()


# ---------------------------------------------------------
# Training and validation accuracy
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training accuracy",
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation accuracy",
)

plt.title(
    "ANN Training and Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "ann_accuracy_history.png",
    dpi=300,
)

plt.show()


# ---------------------------------------------------------
# Training and validation loss
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training loss",
)

plt.plot(
    history.history["val_loss"],
    label="Validation loss",
)

plt.title(
    "ANN Training and Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "ann_loss_history.png",
    dpi=300,
)

plt.show()