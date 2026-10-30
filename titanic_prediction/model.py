"""Imputation, encoding and logistic regression fitted only on training rows."""

import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from titanic_prediction.data import CATEGORICAL, FEATURES, NUMERIC, load_data, validate_features

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "titanic.csv"
DEFAULT_OUTPUT = ROOT / "artifacts"


def make_pipeline():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)), ("encode", OneHotEncoder(handle_unknown="ignore"))])
    preprocessing = ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)])
    return Pipeline([("preprocess", preprocessing), ("classifier", LogisticRegression(max_iter=1000, random_state=42))])


def train(data_path: Path = DEFAULT_DATA, output_dir: Path = DEFAULT_OUTPUT) -> dict:
    frame = load_data(data_path)
    x_train, x_test, y_train, y_test = train_test_split(frame[FEATURES], frame.Survived, test_size=0.2, random_state=42, stratify=frame.Survived)
    model = make_pipeline().fit(x_train, y_train)
    predicted = model.predict(x_test)
    probability = model.predict_proba(x_test)[:, 1]
    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train).predict(x_test)
    matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
    metrics = {
        "dataset": str(data_path.resolve()), "rows": len(frame),
        "train_rows": len(x_train), "test_rows": len(x_test),
        "test_fraction": 0.2, "random_state": 42, "stratified": True,
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "log_loss": float(log_loss(y_test, probability, labels=[0, 1])),
        "baseline_accuracy": float(accuracy_score(y_test, baseline)),
        "confusion_matrix": matrix.tolist(), "positive_class": "Survived = 1",
        "feature_columns": FEATURES,
        "training_numeric_ranges": {name: [float(x_train[name].min()), float(x_train[name].max())] for name in NUMERIC if x_train[name].notna().any()},
        "train_passenger_ids": frame.loc[x_train.index, "PassengerId"].tolist(),
        "test_passenger_ids": frame.loc[x_test.index, "PassengerId"].tolist(),
        "classification_report": classification_report(y_test, predicted, labels=[0, 1], target_names=["Did not survive", "Survived"], output_dict=True, zero_division=0),
        "scikit_learn_version": sklearn.__version__,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "metrics": metrics, "schema_version": 1}, output_dir / "model.joblib")
    results = x_test.copy()
    results.insert(0, "PassengerId", frame.loc[x_test.index, "PassengerId"])
    results["ActualSurvived"] = y_test
    results["PredictedSurvived"] = predicted
    results["SurvivalProbability"] = probability
    results.to_csv(output_dir / "test_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    from titanic_prediction.plots import save_plots
    save_plots(y_test.to_numpy(), probability, matrix, output_dir)
    return metrics


def predict(pclass: int, sex: str, age: float | None = None, sibsp: int = 0, parch: int = 0, fare: float | None = None, embarked: str | None = "S", model_path: Path = DEFAULT_OUTPUT / "model.joblib") -> dict:
    row = validate_features(pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]], columns=FEATURES))
    # Only load model files you created or trust; joblib uses pickle.
    bundle = joblib.load(model_path)
    if bundle.get("schema_version") != 1:
        raise ValueError("Unsupported model artifact; retrain this project")
    model = bundle["pipeline"]
    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(model.predict(row)[0])
    missing = [name for name in FEATURES if pd.isna(row.iloc[0][name])]
    outside = [name for name, (low, high) in bundle["metrics"]["training_numeric_ranges"].items() if pd.notna(row.iloc[0][name]) and not low <= float(row.iloc[0][name]) <= high]
    return {
        "predicted_survived": prediction,
        "label": "Survived" if prediction else "Did not survive",
        "survival_probability": probability, "decision_threshold": 0.5,
        "imputed_fields": missing, "outside_training_range": outside,
        "note": "Statistical classification of a historical passenger profile, not a verified outcome for a named person.",
    }
