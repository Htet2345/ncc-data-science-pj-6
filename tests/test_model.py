import json

import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageChops
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from titanic_prediction.data import FEATURES, NUMERIC
from titanic_prediction.model import DEFAULT_DATA, make_pipeline, predict, train


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    output = tmp_path_factory.mktemp("titanic")
    return output, train(output_dir=output)


def test_disjoint_stratified_split_and_training_only_preprocessing(trained):
    output, metrics = trained
    training, testing = metrics["train_passenger_ids"], metrics["test_passenger_ids"]
    assert len(training) == 1047 and len(testing) == 262
    assert not set(training) & set(testing)
    frame = pd.read_csv(DEFAULT_DATA).set_index("PassengerId")
    assert set(training) | set(testing) == set(frame.index)
    assert abs(frame.loc[training, "Survived"].mean() - frame.loc[testing, "Survived"].mean()) < 0.01
    pipeline = joblib.load(output / "model.joblib")["pipeline"]
    numeric = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
    assert np.allclose(numeric.named_steps["impute"].statistics_, frame.loc[training, NUMERIC].median().to_numpy())
    assert numeric.named_steps["scale"].n_samples_seen_ == 1047


def test_held_out_metrics_and_probability_exports(trained):
    output, metrics = trained
    rows = pd.read_csv(output / "test_predictions.csv")
    assert list(rows.PassengerId) == metrics["test_passenger_ids"]
    assert rows.SurvivalProbability.between(0, 1).all()
    assert metrics["accuracy"] == pytest.approx(accuracy_score(rows.ActualSurvived, rows.PredictedSurvived))
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(rows.ActualSurvived, rows.SurvivalProbability))
    assert metrics["confusion_matrix"] == confusion_matrix(rows.ActualSurvived, rows.PredictedSurvived, labels=[0, 1]).tolist()
    assert sum(map(sum, metrics["confusion_matrix"])) == 262
    assert metrics["accuracy"] > metrics["baseline_accuracy"]
    assert json.loads((output / "metrics.json").read_text()) == metrics


def test_saved_profile_prediction_and_missing_values(trained):
    result = predict(3, "male", 22, 0, 0, 7.25, "S", trained[0] / "model.joblib")
    assert result["predicted_survived"] in [0, 1]
    assert 0 <= result["survival_probability"] <= 1
    assert result["imputed_fields"] == []
    missing = predict(1, "female", model_path=trained[0] / "model.joblib")
    assert missing["imputed_fields"] == ["Age", "Fare"]
    assert 0 <= missing["survival_probability"] <= 1


@pytest.mark.parametrize("kwargs", [{"pclass": 4}, {"sex": "unknown"}, {"age": -1}, {"age": float("inf")}, {"fare": -1}, {"sibsp": -1}, {"parch": 0.5}, {"embarked": "X"}])
def test_invalid_profiles(trained, kwargs):
    profile = {"pclass": 3, "sex": "male", "age": 22, "fare": 7.25}
    profile.update(kwargs)
    with pytest.raises(ValueError):
        predict(**profile, model_path=trained[0] / "model.joblib")


def test_outside_training_range_reported(trained):
    result = predict(1, "female", 110, fare=9000, model_path=trained[0] / "model.joblib")
    assert "Age" in result["outside_training_range"]
    assert "Fare" in result["outside_training_range"]


def test_chart_images(trained):
    for name in ("confusion_matrix.png", "roc_curve.png"):
        with Image.open(trained[0] / name) as image:
            assert image.format == "PNG" and image.size == (1600, 960)
            assert ImageChops.difference(image.convert("RGB"), Image.new("RGB", image.size, "white")).getbbox()


def test_pipeline_does_not_fit_on_test_features_or_labels(trained):
    output, metrics = trained
    frame = pd.read_csv(DEFAULT_DATA).set_index("PassengerId")
    training = metrics["train_passenger_ids"]
    reference = make_pipeline().fit(frame.loc[training, FEATURES], frame.loc[training, "Survived"])
    saved = joblib.load(output / "model.joblib")["pipeline"]
    assert np.allclose(reference.named_steps["classifier"].coef_, saved.named_steps["classifier"].coef_)
    changed = frame.copy()
    changed.loc[metrics["test_passenger_ids"], "Survived"] = 1 - changed.loc[metrics["test_passenger_ids"], "Survived"]
    # Refit using the exact fixed training IDs; changed test labels never enter fit.
    second = make_pipeline().fit(changed.loc[training, FEATURES], changed.loc[training, "Survived"])
    assert np.allclose(second.named_steps["classifier"].coef_, saved.named_steps["classifier"].coef_)
