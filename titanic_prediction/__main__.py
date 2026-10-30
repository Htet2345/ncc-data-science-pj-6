"""Train the historical classifier or classify an anonymous profile."""

import argparse
import json
from pathlib import Path

from titanic_prediction.model import DEFAULT_DATA, DEFAULT_OUTPUT, predict, train


def main(argv=None):
    parser = argparse.ArgumentParser(description="Titanic Survival Prediction - logistic regression")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--data", type=Path, default=DEFAULT_DATA)
    training.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--pclass", type=int, required=True)
    prediction.add_argument("--sex", required=True)
    prediction.add_argument("--age", type=float, default=None)
    prediction.add_argument("--sibsp", type=int, default=0)
    prediction.add_argument("--parch", type=int, default=0)
    prediction.add_argument("--fare", type=float, default=None)
    prediction.add_argument("--embarked", default="S")
    prediction.add_argument("--model", type=Path, default=DEFAULT_OUTPUT / "model.joblib")
    args = parser.parse_args(argv)
    try:
        result = train(args.data, args.output_dir) if args.command == "train" else predict(args.pclass, args.sex, args.age, args.sibsp, args.parch, args.fare, args.embarked, args.model)
    except FileNotFoundError as error:
        parser.error(f"File not found: {error.filename}. Prepare data and run training first.")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    # Full metrics are saved in artifacts; keep terminal output readable.
    displayed = {key: value for key, value in result.items() if key not in ("train_passenger_ids", "test_passenger_ids", "classification_report")} if args.command == "train" else result
    print(json.dumps(displayed, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
