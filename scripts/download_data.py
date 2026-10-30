"""Download the public OpenML Titanic dataset and retain only modelling fields."""

import csv
import hashlib
import io
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METADATA_URL = "https://www.openml.org/api/v1/json/data/40945"
CSV_URL = "https://www.openml.org/data/get_csv/16826755/phpMYEkMl"
MAPPING = {"pclass": "Pclass", "sex": "Sex", "age": "Age", "sibsp": "SibSp", "parch": "Parch", "fare": "Fare", "embarked": "Embarked", "survived": "Survived"}


def download(output_dir: Path = ROOT / "data"):
    metadata = json.loads(urllib.request.urlopen(METADATA_URL, timeout=60).read())
    raw = urllib.request.urlopen(CSV_URL, timeout=90).read()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if not set(MAPPING).issubset(set(reader.fieldnames or [])):
        raise ValueError("Source schema changed; inspect the OpenML dataset before proceeding")
    rows = []
    for index, original in enumerate(reader, start=1):
        row = {"PassengerId": index}
        for source, destination in MAPPING.items():
            value = original[source].strip().strip("'")
            row[destination] = "" if value == "?" else value
        rows.append(row)
    if len(rows) != 1309:
        raise ValueError("Unexpected source row count; expected 1309")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "titanic.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["PassengerId", *MAPPING.values()])
        writer.writeheader()
        writer.writerows(rows)
    description = metadata["data_set_description"]
    provenance = {
        "dataset_id": description["id"], "name": description["name"],
        "version": description["version"], "license_as_listed_by_openml": description["licence"],
        "metadata_url": METADATA_URL, "download_url": CSV_URL,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "processed_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "rows": len(rows),
        "transformation": "Selected modelling fields; ? becomes blank; PassengerId is 1-based source row number. Names, tickets, cabins, boat, body and destinations omitted.",
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(rows)} historical records in {destination}")


if __name__ == "__main__":
    download()
