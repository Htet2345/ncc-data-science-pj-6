# Historical Titanic dataset

This project uses OpenML dataset **40945, Titanic, version 1**, with 1,309
historical passenger records. OpenML credits Frank E. Harrell Jr. and Thomas
Cason and lists its license as **Public**. The upstream description identifies
Vanderbilt Biostatistics and Encyclopedia Titanica among the historical sources.

- [OpenML dataset](https://www.openml.org/d/40945)
- [OpenML metadata and license field](https://www.openml.org/api/v1/json/data/40945)
- [Download endpoint used](https://www.openml.org/data/get_csv/16826755/phpMYEkMl)

`titanic.csv` is a reduced modelling table. We select Pclass, Sex, Age, SibSp,
Parch, Fare, Embarked and Survived. Missing `?` values become empty CSV cells.
`PassengerId` is our 1-based source-row number, not an original ticket or
Kaggle passenger identifier. It is retained for split auditing and never used
as a model feature.

Names, tickets, cabin text, home destinations, lifeboat and body-recovery
fields are omitted. Lifeboat/body information would reveal the outcome rather
than provide a valid pre-outcome predictor.

There are 809 records labelled 0 (did not survive) and 500 labelled 1 (survived).
Age, fare and embarkation port contain missing values. Imputation statistics
are learned from the training split only.

The source and processed-file SHA-256 hashes, source URLs and transformation
are recorded in `provenance.json`. `.gitattributes` preserves CSV bytes across
Git checkouts so the processed checksum stays valid.

To refresh the data, run `python -m scripts.download_data` from the project
folder. This requires internet access; normal training uses the bundled CSV
and is offline. The downloader checks the expected schema and row count.

This dataset has not been confirmed as the exact version used in the teacher's
videos. Do not merge its row IDs directly with Kaggle's 891-row training CSV.
