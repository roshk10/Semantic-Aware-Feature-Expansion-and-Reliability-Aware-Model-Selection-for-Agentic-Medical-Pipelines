import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

print("\nTraining Heart Disease Model")

df = pd.read_csv(
    "data/datasets/heart_disease.csv"
)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = XGBClassifier(
    eval_metric="logloss"
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

acc = accuracy_score(
    y_test,
    preds
)

print(
    f"Heart Accuracy: {acc:.4f}"
)

joblib.dump(
    model,
    "data/models/trained/heart_xgboost.pkl"
)

print("Heart model saved.")

print("\nTraining Diabetes Model")

df = pd.read_csv(
    "data/datasets/diabetes.csv"
)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = XGBClassifier(
    eval_metric="logloss"
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

acc = accuracy_score(
    y_test,
    preds
)

print(
    f"Diabetes Accuracy: {acc:.4f}"
)

joblib.dump(
    model,
    "data/models/trained/diabetes_xgboost.pkl"
)

print("Diabetes model saved.")
