# Titanic Survival Prediction - Data Science Project 6

Binary classification using pandas and scikit-learn logistic regression.
The project topic matches Module 5.3, videos **17-19: Titanic Survival Prediction,
Parts 1-3**. This is an original implementation based on those lesson titles;
the teacher's exact code and dataset have not been verified.

The bundled historical dataset is OpenML Titanic 40945, version 1, with 1,309
passenger records. Its provenance, attribution and OpenML license field are
documented under `data/`. It is not synthetic data and is not the same row set
as Kaggle's 891-row training CSV.

## Workflow

1. Validate the dataset and retain only pre-outcome modelling features.
2. Create a fixed, stratified 80/20 split using random seed 42.
3. Learn numeric medians and categorical modes from training rows only.
4. Scale numeric features and one-hot encode categories in a ColumnTransformer.
5. Fit logistic regression inside the same pipeline.
6. Evaluate held-out accuracy, precision, recall, F1, ROC-AUC and log loss.
7. Compare accuracy with a majority-class baseline and save the fitted pipeline.
8. Classify an anonymous historical passenger profile using the saved model.

Names, ticket strings and outcome-related lifeboat/body fields are omitted.
PassengerId is only an audit identifier; it is not a model feature.

## Feature definitions

| Feature | Meaning / accepted values |
| --- | --- |
| Pclass | Passenger class, integer 1, 2 or 3 |
| Sex | Historical dataset labels `male` or `female` |
| Age | Age in years, including fractional infant ages; 0-120 or missing |
| SibSp | Siblings/spouses aboard, integer 0-20 |
| Parch | Parents/children aboard, integer 0-20 |
| Fare | Historical ticket fare, 0-10,000 or missing; not modern USD |
| Embarked | C = Cherbourg, Q = Queenstown, S = Southampton, or missing |
| Survived | Training target: 0 = did not survive, 1 = survived |

Age and fare use median imputation; categorical features use most-frequent
imputation. Sex, Pclass, SibSp and Parch must be supplied. Missing numeric
columns are kept by the imputer; if a training feature is entirely missing,
scikit-learn fills it with zero rather than dropping it. Port categories are
encoded with unknown-category handling. Input validation accepts only the
documented historical labels. These broad bounds are validation checks, not
an assurance that the dataset covers every allowed value.

## Setup on Windows

Requires Python 3.10+; developed using Python 3.13.

Open the folder in VS Code, then **Terminal > New Terminal**:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

The environment is already installed on the original development machine.
The commands above reproduce it on a fresh checkout. The lock file records
tested runtime and testing versions. `requirements.txt` lists direct runtime
dependencies; `requirements-dev.txt` adds pytest.

## Train and evaluate

```powershell
.\.venv\Scripts\python.exe -m titanic_prediction train
```

The bundled data produces **1,047 training rows and 262 test rows**. Stratification
preserves the approximate survival-class balance. All preprocessing is fitted
inside the training pipeline, avoiding imputation/scaling leakage from test
rows. The saved model is the training-fitted model that was evaluated; it is
not refitted on the complete dataset afterwards.

## Predict an example profile

```powershell
.\.venv\Scripts\python.exe -m titanic_prediction predict --pclass 3 --sex male --age 22 --sibsp 0 --parch 0 --fare 7.25 --embarked S
```

The JSON output contains predicted class, survival probability, imputed fields
and numeric features outside the training ranges. The class uses a threshold
of 0.5. The probability is the logistic model's estimate, not a certainty or a
separately calibrated probability.

To demonstrate missing-value handling, omit age and fare:

```powershell
.\.venv\Scripts\python.exe -m titanic_prediction predict --pclass 1 --sex female
```

SibSp and Parch default to zero; Embarked defaults to S. This defaults profile
fields for the demo, rather than discovering their true values. The result is
a statistical classification of a historical profile, not verification of an
actual named person's fate.

Alternative data/output paths:

```powershell
.\.venv\Scripts\python.exe -m titanic_prediction train --data .\data\titanic.csv --output-dir .\artifacts
.\.venv\Scripts\python.exe -m titanic_prediction predict --pclass 3 --sex male --model .\artifacts\model.joblib
```

Run module commands from the project folder. Default paths are relative to the
project; CLI-supplied relative paths follow the current shell directory.
Only load artifacts you created or trust: joblib uses pickle, which can execute
code from an untrusted file. Use the recorded dependency versions when reloading.

## Outputs

| File under `artifacts/` | Contents |
| --- | --- |
| `model.joblib` | Fitted preprocessing/classification pipeline and metadata |
| `metrics.json` | All metrics, classification report and exact split IDs |
| `test_predictions.csv` | Held-out features, actual/predicted classes and probabilities |
| `confusion_matrix.png` | Actual classes on rows, predictions on columns |
| `roc_curve.png` | True-positive rate versus false-positive rate across thresholds |

For confusion-matrix label order [0,1], entries are `[[TN, FP], [FN, TP]]`.
Precision, recall and F1 treat survival (1) as positive. AUC measures ranking,
while log loss evaluates probabilistic errors. The baseline always predicts
the most frequent training class. A random passenger split can share family
or travel-group patterns across partitions, so this is a classroom holdout
evaluation rather than proof of generalization to independent populations.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Tests verify data provenance/checksum, schema, invalid inputs, missing-value
handling, unique IDs, stratified/disjoint splits, training-only medians and
scaler statistics, held-out metrics, probability bounds, saved-model loading,
image decoding and real training/prediction command-line processes.

VS Code **F5** trains the classifier. **Terminal > Run Task** includes training,
sample prediction and tests. No server, account login or PowerShell execution-
policy changes are required.

## Structure

```text
titanic_prediction/
  data.py       Schema and feature validation
  model.py      Preprocessing, classification, evaluation and prediction
  plots.py      Held-out confusion matrix and ROC curve
  __main__.py   Command-line interface
data/           Reduced historical CSV, provenance and attribution
scripts/        Reproducible OpenML download/preparation utility
tests/          Data, pipeline and command-line tests
.vscode/        Interpreter, debugger and tasks
artifacts/      Generated outputs, ignored by Git
```

## GitHub submission

Include source, historical data/provenance, downloader, tests, README, dependency
files, pytest and VS Code configuration. Generated artifacts, environment,
caches and local test report are ignored. No demo screenshots are included.
Repository: https://github.com/Htet2345/ncc-data-science-pj-6

## References

- [OpenML Titanic metadata and attribution](https://www.openml.org/api/v1/json/data/40945)
- [scikit-learn Titanic mixed-type example](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html)
- [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [scikit-learn classification metrics](https://scikit-learn.org/stable/api/sklearn.metrics.html)
