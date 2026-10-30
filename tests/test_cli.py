import json
import subprocess
import sys

from titanic_prediction.model import ROOT


def run(*args):
    return subprocess.run([sys.executable, "-m", "titanic_prediction", *args], cwd=ROOT, capture_output=True, text=True, timeout=90)


def test_train_and_predict_separate_processes(tmp_path):
    result = run("train", "--output-dir", str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["rows"] == 1309
    result = run("predict", "--pclass", "3", "--sex", "male", "--age", "22", "--fare", "7.25", "--model", str(tmp_path / "model.joblib"))
    assert result.returncode == 0, result.stderr
    assert 0 <= json.loads(result.stdout)["survival_probability"] <= 1


def test_missing_model_error(tmp_path):
    result = run("predict", "--pclass", "3", "--sex", "male", "--model", str(tmp_path / "missing.joblib"))
    assert result.returncode == 2
    assert "training first" in result.stderr and "Traceback" not in result.stderr


def test_invalid_profile_error():
    result = run("predict", "--pclass", "4", "--sex", "male")
    assert result.returncode == 2
    assert "Pclass" in result.stderr


def test_missing_data_error(tmp_path):
    result = run("train", "--data", str(tmp_path / "missing.csv"))
    assert result.returncode == 2
    assert "File not found" in result.stderr
