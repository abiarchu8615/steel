from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELS = BASE / "models"
REPORTS = BASE / "reports"

MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)


def train_model():
    df = pd.read_csv(DATA / "ai4i2020.csv")
    df.columns = df.columns.str.strip()

    df["Temperature difference [K]"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )

    df["Mechanical power proxy"] = (
        df["Torque [Nm]"] * df["Rotational speed [rpm]"]
    )

    target = "Machine failure"

    drop_cols = [
        "UDI",
        "Product ID",
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "classification_report": classification_report(
            y_test,
            pred,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "features": list(X.columns),
    }

    joblib.dump(model, MODELS / "machine_failure_model.joblib")

    with open(REPORTS / "machine_failure_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Model saved to models/machine_failure_model.joblib")
    print("Metrics saved to reports/machine_failure_metrics.json")
    print("Accuracy:", round(metrics["accuracy"], 4))

    return model, metrics


if __name__ == "__main__":
    train_model()