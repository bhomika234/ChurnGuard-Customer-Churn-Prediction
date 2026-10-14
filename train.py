
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    classification_report
)

# 1. Dataset location
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}. "
        "Place the CSV file in the project folder."
    )

# 2. Load dataset
df = pd.read_csv(DATA_PATH)

# 3. Clean data
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

# Remove customer identifiers
df = df.drop(columns=["customerID"], errors="ignore")

# Convert TotalCharges to numeric
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

# Prepare target
df = df[df["Churn"].notna()].copy()
y = df["Churn"].astype(str).str.strip().map(
    {"Yes": 1, "No": 0}
)

df = df.loc[y.notna()].copy()
y = y.loc[df.index].astype(int)
X = df.drop(columns=["Churn"])

# 4. Split data: train, validation, test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=42, stratify=y_temp
)

# 5. Identify column types
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 6. Train model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X_train, y_train)

# 7. Select threshold using validation data
val_probabilities = model.predict_proba(X_val)[:, 1]

thresholds = [i / 100 for i in range(10, 91, 5)]
best_threshold = 0.5
best_f1 = -1

for threshold in thresholds:
    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    score = f1_score(
        y_val, val_predictions, zero_division=0
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

# 8. Final test evaluation
test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\n===== CHURNGUARD MODEL RESULTS =====")
print(f"Train records: {len(X_train)}")
print(f"Validation records: {len(X_val)}")
print(f"Test records: {len(X_test)}")
print(f"Selected threshold: {best_threshold:.2f}")
print(f"Accuracy: {accuracy_score(y_test, test_predictions):.4f}")
print(f"Precision: {precision_score(y_test, test_predictions, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, test_predictions, zero_division=0):.4f}")
print(f"F1-score: {f1_score(y_test, test_predictions, zero_division=0):.4f}")
print(f"PR-AUC: {average_precision_score(y_test, test_probabilities):.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test, test_predictions,
    target_names=["Stayed", "Churned"],
    zero_division=0
))

# 9. Save model and threshold
artifact = {
    "model": model,
    "threshold": best_threshold,
    "feature_columns": X.columns.tolist()
}

joblib.dump(artifact, "churn_model.joblib")

print("\nModel saved successfully as churn_model.joblib")
