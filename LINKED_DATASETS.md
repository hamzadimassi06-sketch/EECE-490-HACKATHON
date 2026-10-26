# Datasets linked in the hackathon brief

The brief links five Kaggle datasets to show the types of tables we may get.
It says the assessment uses fresh tables. We use these links for additional
local checks of the same program, not as the private assessment datasets.

The five links were checked directly in the Word document. The first one is
Telco Customer Churn. The AI4I link appears once in the document.

## Source files

| Dataset | Main files | Use in this project |
| --- | --- | --- |
| [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) | `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Classification check, target `Churn` |
| [Predictive Maintenance AI4I 2020](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) | `ai4i2020.csv` | Classification check, target `Machine failure` |
| [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) | `train.csv`, `test.csv`, `data_description.txt`, `sample_submission.csv` | Regression check, target `SalePrice`, plus predictions for the unlabeled test file |
| [Bike Sharing Demand](https://www.kaggle.com/competitions/bike-sharing-demand/data) | `train.csv`, `test.csv`, `sampleSubmission.csv` | Regression check, target `count`, plus predictions for the unlabeled test file |
| [Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) | `Mall_Customers.csv` | Clustering check using annual income and spending score |

Telco, AI4I, and Mall were downloaded from the linked Kaggle dataset pages
through Kaggle's public download API. Their original CSVs contain 7,043,
10,000, and 200 rows respectively. The checks record the source URL and a
SHA-256 hash of each CSV so the exact version can be identified.

The House Prices ZIP was supplied separately and contains all four expected
files. Its training table has 1,460 labeled rows, and its test table has 1,459
rows without prices. The supplied `archive.zip` contains Telco; its CSV hash
matches the version already tested.

The Bike Sharing ZIP was also supplied and contains all three expected files:
10,886 labeled training rows and 6,493 unlabeled test rows. All five datasets
are now available and checked. Keep the two competition ZIPs in separate
folders because both contain `train.csv` and `test.csv`.

## Reproduce the Telco AI4I and Mall checks

Download and extract the Telco, AI4I, and Mall CSVs into a local folder. The
evaluation script searches that folder recursively, so separate subfolders
are fine. Run this from the project folder after installing requirements:

```bash
python evaluate_linked_data.py --data-dir /path/to/downloaded/csvs
```

To run just one case, or explicitly set the Mall cluster count:

```bash
python evaluate_linked_data.py --data-dir /path/to/downloaded/csvs --cases ai4i
python evaluate_linked_data.py --data-dir /path/to/downloaded/csvs --cases mall --mall-clusters 5
```

The script prepares inputs under `output/linked_checks`, calls `main.py`,
and saves predictions, profiles, model reports, logs, and measured metrics.
Results from the completed development run are in `LOCAL_RESULTS.md` and
`results/linked_results.json`. Raw source CSVs are not included in the project.

## Preparation choices

### Telco Customer Churn

We predict `Churn`, exclude the unique `customerID`, and mark `SeniorCitizen`
as categorical even though it uses numeric codes. `TotalCharges` is numeric
and has 11 whitespace-only cells in the downloaded CSV. The existing
training-only imputation handles them. We keep the other customer features.

The check uses a stratified 80/20 split with seed 42. The program receives
only the training labels. It makes its own internal validation split, and
the evaluation script scores the outer test rows after predictions are
saved. It reports balanced accuracy, ordinary accuracy, and positive-class
precision, recall, and F1, together with a majority-class baseline.

### Predictive Maintenance AI4I 2020

We predict `Machine failure` from `Type`, air and process temperatures,
rotational speed, torque, and tool wear. `UDI` and `Product ID` are identifiers
and are excluded. `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` describe failure
outcomes, so the evaluation script removes them before creating the training
and test inputs. This prevents the answer from appearing among the features.
The [original UCI documentation](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
describes these failure labels and the six input features.

Only 339 of the 10,000 rows have machine failure. We therefore report balanced
accuracy and failure recall alongside accuracy. The outer split is the same
stratified 80/20 setup as Telco. This is a random-row development check, not a
forecast of future machine failures or evidence of performance on real
industrial data; AI4I is a synthetic dataset.

### Mall Customer Segmentation

The brief specifically mentions annual income and spending score. We use
those two numeric columns and standardize them through the existing pipeline.
`CustomerID`, `Gender`, and `Age` are not part of this two-feature example.

We choose five clusters for this local check. The source has no true cluster
labels and does not specify a required cluster count. The script reports
the number of occupied groups and silhouette on the standardized features.
Silhouette describes separation of the same rows that were clustered; it is
not accuracy or a held-out prediction score. The production program always
takes the required cluster count from the supplied metadata.

### House Prices

Extract the four source files into their own folder, then run:

```bash
python evaluate_house_prices.py --data-dir /path/to/extracted/house-prices
```

We predict the original `SalePrice` values, exclude `Id`, and declare
`MSSubClass` categorical because it codes building classes. The fixed 80/20
outer split uses seed 42: 1,168 rows for training and 292 for scoring. The
program makes a separate internal split to choose a model. It never sees the
outer test labels while fitting or choosing models.

The dictionary uses `NA` as a category code in 14 columns describing alleys,
basements, fireplaces, garages, pools, fences, and miscellaneous features.
The adapter keeps those original codes as `__NA_category__`. It also changes
the valid masonry category `None` to `__no_masonry__`, so the normal CSV reader
does not mistake it for a missing value. These are fixed, row-by-row mappings;
they do not learn statistics from the test table. Some source category codes
conflict with other property fields, so we preserve them without guessing
repairs. Other missing values go through the usual training-only imputers.

We report dollar RMSE, MAE, R squared, and log-price RMSE.
[Kaggle's evaluation](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview/evaluation)
uses RMSE between the logarithms of predicted and actual prices. The general
program still selects models by ordinary RMSE. Log-price RMSE is an additional
evaluation measure; it is not used to tune on the outer test labels. If a run
produces nonpositive prices, the script reports their count and leaves the
log metric unset rather than silently clipping them.

After the holdout check, the script trains on all 1,460 labeled rows and
predicts the 1,459 rows from the supplied test file. It saves
`output/house_prices_checks/house_prices_predictions.csv` with `Id,SalePrice`
in the original test order. The saved development-run copy is
`results/house_prices_predictions.csv`, with metrics and source hashes in
`results/house_prices_results.json`. The sample submission is used only to
check headers and ID order; its prices are examples, not test labels.
No score is claimed for the unlabeled test file, and nothing is submitted to
Kaggle automatically.

### Bike Sharing Demand

Extract the three source files into their own folder, then run:

```bash
python evaluate_bike_sharing.py --data-dir /path/to/extracted/bike-sharing
```

We predict `count`. In every supplied training row it equals `casual` plus
`registered`, so those two columns are removed before creating any model
inputs. They are also absent from the prediction file. The adapter derives
hour, weekday, month, and year from each row's `datetime`, then removes the
raw timestamp from the features. Original timestamps are retained separately
for output. `season` and `weather` are declared categorical. The other inputs
use the ordinary numeric preparation in the model pipeline.

The source training file covers days 1-19 of each month in 2011 and 2012; the
unlabeled test file covers the remaining days. Our local check fits on days
1-15 in each month and scores days 16-19, giving 8,600 training and 2,286
scored rows. This holds out blocks of days within every month. Training still
includes later months than some scored dates, so it is not a strict
future-only forecast test. The core model uses its usual random internal
split within those 8,600 training rows to select a model. No extra hourly
rows are added where the original dataset has gaps.

The target stays in rental-count units and model selection uses ordinary
RMSE. Before scoring or exporting, the adapter clips negative predictions
to zero because rental counts cannot be negative. This fixed rule is applied
to both runs, and the raw minimum and number clipped are recorded. Predictions
are not rounded to integers. The results include RMSE, MAE, R squared, and
[RMSLE](https://www.kaggle.com/c/bike-sharing-demand/overview), calculated using
`log1p` on the true and nonnegative predicted counts. This differs from the
House Prices log metric, which uses `log` without adding 1.

The second run uses all 10,886 labeled rows to predict the 6,493 competition
test rows. It exports `datetime,count` in the original order. The sample
submission's zero counts are placeholders and are never used for scoring.
The saved development-run files are `results/bike_sharing_predictions.csv`
and `results/bike_sharing_results.json`. Their reproducible working copies
are written under `output/bike_sharing_checks`. No error score is available
for the unlabeled test file.
