import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from titanic_prediction.data import FEATURES, load_data, validate_features
from titanic_prediction.model import DEFAULT_DATA


def test_historical_dataset_and_provenance():
    frame = load_data(DEFAULT_DATA)
    assert len(frame) == 1309
    assert frame.Survived.value_counts().to_dict() == {0: 809, 1: 500}
    assert frame.Age.isna().sum() == 263
    provenance = json.loads(DEFAULT_DATA.with_name("provenance.json").read_text())
    assert provenance["processed_sha256"] == hashlib.sha256(DEFAULT_DATA.read_bytes()).hexdigest()
    assert set(frame.columns) == set(["PassengerId", *FEATURES, "Survived"])
    assert not set(FEATURES) & {"PassengerId", "Survived", "boat", "body", "name"}


@pytest.mark.parametrize("kind", ["extra_column", "few_rows", "duplicate_id", "missing_id", "bad_class", "missing_class", "bad_sex", "bad_port", "negative_age", "infinite_age", "negative_fare", "fractional_sibsp", "missing_parch", "bad_target", "one_class", "rare_class"])
def test_invalid_training_data(tmp_path, kind):
    frame = pd.read_csv(DEFAULT_DATA)
    if kind == "extra_column":
        frame["boat"] = 1
    elif kind == "few_rows":
        frame = frame.head(10)
    elif kind == "duplicate_id":
        frame.loc[1, "PassengerId"] = frame.loc[0, "PassengerId"]
    elif kind == "missing_id":
        frame["PassengerId"] = frame.PassengerId.astype(float)
        frame.loc[0, "PassengerId"] = np.nan
    elif kind == "bad_class":
        frame.loc[0, "Pclass"] = 4
    elif kind == "missing_class":
        frame["Pclass"] = frame.Pclass.astype(float)
        frame.loc[0, "Pclass"] = np.nan
    elif kind == "bad_sex":
        frame.loc[0, "Sex"] = "invalid"
    elif kind == "bad_port":
        frame.loc[0, "Embarked"] = "X"
    elif kind == "negative_age":
        frame.loc[0, "Age"] = -1
    elif kind == "infinite_age":
        frame.loc[0, "Age"] = np.inf
    elif kind == "negative_fare":
        frame.loc[0, "Fare"] = -1
    elif kind == "fractional_sibsp":
        frame["SibSp"] = frame.SibSp.astype(float)
        frame.loc[0, "SibSp"] = 0.5
    elif kind == "missing_parch":
        frame["Parch"] = frame.Parch.astype(float)
        frame.loc[0, "Parch"] = np.nan
    elif kind == "bad_target":
        frame.loc[0, "Survived"] = 2
    elif kind == "one_class":
        frame["Survived"] = 0
    elif kind == "rare_class":
        frame["Survived"] = 0
        frame.loc[0, "Survived"] = 1
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_data(path)


def test_missing_age_fare_and_port_are_allowed():
    row = pd.DataFrame([[3, " Male ", None, 0, 0, None, ""]], columns=FEATURES)
    result = validate_features(row)
    assert result.Sex.iloc[0] == "male"
    assert result[["Age", "Fare", "Embarked"]].isna().all().all()
