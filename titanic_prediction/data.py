"""Validate historical model features without using outcome-related columns."""

from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Pclass", "Sex", "Embarked"]
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    if set(frame.columns) != set(FEATURES):
        raise ValueError("Features must be exactly: " + ", ".join(FEATURES))
    result = frame[FEATURES].copy()
    bounds = {"Pclass": (1, 3), "Age": (0, 120), "SibSp": (0, 20), "Parch": (0, 20), "Fare": (0, 10000)}
    for column, (low, high) in bounds.items():
        result[column] = pd.to_numeric(result[column], errors="raise")
        present = result[column].dropna()
        if not np.isfinite(present.to_numpy(dtype=float)).all() or not present.between(low, high).all():
            raise ValueError(f"{column} must contain finite values between {low} and {high}")
        if column in ("Pclass", "SibSp", "Parch"):
            if result[column].isna().any() or (present % 1 != 0).any():
                raise ValueError(f"{column} must contain whole numbers without missing values")
    result["Sex"] = result.Sex.map(lambda value: value.strip().lower() if isinstance(value, str) else value)
    if not result.Sex.isin(["male", "female"]).all():
        raise ValueError("Sex must be male or female, matching the historical dataset labels")
    result["Embarked"] = result.Embarked.map(lambda value: value.strip().upper() if isinstance(value, str) else value)
    result["Embarked"] = result.Embarked.replace("", np.nan)
    if not result.Embarked.dropna().isin(["C", "Q", "S"]).all():
        raise ValueError("Embarked must be C, Q, S or missing")
    # scikit-learn's categorical imputer expects np.nan, not pandas nullable NA.
    result["Embarked"] = result.Embarked.astype(object).where(result.Embarked.notna(), np.nan)
    return result


def load_data(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    expected = set(["PassengerId", *FEATURES, "Survived"])
    if set(raw.columns) != expected:
        raise ValueError("CSV must have exactly PassengerId, " + ", ".join(FEATURES) + ", Survived")
    if len(raw) < 20:
        raise ValueError("Use at least 20 historical observations")
    identifiers = pd.to_numeric(raw.PassengerId, errors="raise")
    if identifiers.isna().any() or not np.isfinite(identifiers).all() or (identifiers < 1).any() or (identifiers % 1 != 0).any() or identifiers.duplicated().any():
        raise ValueError("PassengerId must be a unique positive whole number")
    frame = validate_features(raw[FEATURES])
    target = pd.to_numeric(raw.Survived, errors="raise")
    if not target.isin([0, 1]).all() or set(target.unique()) != {0, 1}:
        raise ValueError("Survived must contain both binary classes 0 and 1")
    if target.value_counts().min() < 5:
        raise ValueError("Each survival class needs at least five rows for stratification")
    frame.insert(0, "PassengerId", identifiers.astype(int))
    frame["Survived"] = target.astype(int)
    return frame
