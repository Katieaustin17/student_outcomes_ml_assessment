from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential



# Reproducibility and folders

np.random.seed(42)
tf.random.set_seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_outcomes.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)



# Load and prepare data

df = pd.read_csv(DATA_PATH)

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

# Reproduce the original 80/20 split. The held-out 20% test set is not
# used for the ANN experiment because it has already been used for the
# final Logistic Regression evaluation.
X_development, _, y_development, _ = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_development,
    y_development,
    test_size=0.20,
    random_state=42,
    stratify=y_development,
)

print(f"ANN training set:   {len(X_train)} samples")
print(f"ANN validation set: {len(X_val)} samples")



# Preprocessing


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

preprocessor = ColumnTransformer(
    [
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features,
        ),
        ("numerical", StandardScaler(), numerical_features),
        ("binary", "passthrough", binary_features),
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

class_names = ["Dropout", "Enrolled", "Graduate"]
target_mapping = {name: i for i, name in enumerate(class_names)}
y_train_encoded = y_train.map(target_mapping).to_numpy()
y_val_encoded = y_val.map(target_mapping).to_numpy()



# Build and train ANN


model = Sequential(
    [
        Input(shape=(X_train_processed.shape[1],)),
        Dense(64, activation="relu"),
        Dropout(0.30),
        Dense(32, activation="relu"),
        Dropout(0.30),
        Dense(3, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

history = model.fit(
    X_train_processed,
    y_train_encoded,
    validation_data=(X_val_processed, y_val_encoded),
    epochs=40,
    batch_size=32,
    verbose=1,
)


# Validation evaluation


probabilities = model.predict(X_val_processed, verbose=0)
y_pred = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(y_val_encoded, y_pred)
macro_precision = precision_score(y_val_encoded, y_pred, average="macro")
macro_recall = recall_score(y_val_encoded, y_pred, average="macro")
macro_f1 = f1_score(y_val_encoded, y_pred, average="macro")

print("\nNeural Network Validation Performance")
print("=" * 45)
print(f"Accuracy:        {accuracy:.3f}")
print(f"Macro Precision: {macro_precision:.3f}")
print(f"Macro Recall:    {macro_recall:.3f}")
print(f"Macro F1:        {macro_f1:.3f}")
print("=" * 45)

print("\nClassification Report:")
print(classification_report(y_val_encoded, y_pred, target_names=class_names))

pd.DataFrame(
    {
        "Metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"],
        "Value": [accuracy, macro_precision, macro_recall, macro_f1],
    }
).to_csv(OUTPUT_DIR / "ann_validation_metrics.csv", index=False)



# Classification report


report = classification_report(
    y_val_encoded,
    y_pred,
    target_names=class_names,
    output_dict=True
)

classification_df = pd.DataFrame(report).transpose()

print("\nClassification Report:")
print(classification_df.round(3))


# Save classification report to CSV
classification_df.to_csv(
    OUTPUT_DIR / "ann_classification_report.csv",
    index=True,
    float_format="%.3f"
)



cm = confusion_matrix(y_val_encoded, y_pred)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
plt.title("ANN Validation Confusion Matrix")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "ann_validation_confusion_matrix.png", dpi=300)
plt.close()



# Training history


plt.figure(figsize=(7, 5))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ANN Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "ann_loss_history.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 5))
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ANN Training and Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "ann_accuracy_history.png", dpi=300)
plt.close()
