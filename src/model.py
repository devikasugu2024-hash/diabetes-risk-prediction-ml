import os

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Find CSV files available in the Kaggle environment
csv_files = {}

for root, dirs, files in os.walk("/kaggle/input"):
    for file in files:
        if file.endswith(".csv"):
            csv_files[file] = os.path.join(root, file)


# Load datasets
train = pd.read_csv(csv_files["train.csv"])
test = pd.read_csv(csv_files["test.csv"])


# Identify target column
target_candidates = list(set(train.columns) - set(test.columns))
TARGET = target_candidates[0]

print("Target:", TARGET)
print("Train shape:", train.shape)
print("Test shape:", test.shape)


# Separate features and target
X = train.drop(columns=[TARGET])
y = train[TARGET]


# Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


# Numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# Train-validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Machine learning model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])


# Train model
model.fit(X_train, y_train)


# Validation prediction
y_pred = model.predict(X_valid)


# Evaluation
accuracy = accuracy_score(y_valid, y_pred)

print("\nValidation Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_valid, y_pred))
