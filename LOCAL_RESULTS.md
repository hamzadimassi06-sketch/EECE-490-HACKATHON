# Local Results

These are development checks on course data, linked public datasets, and generated examples.
They are not results from the 18 private hackathon datasets or the official checker.

## Environment

- Python 3.12.14 on Linux x86_64.
- Package versions are pinned in requirements.txt.
- One computation thread and a 50-second program budget for each benchmark.
- Cases were run sequentially. Wall time includes Python startup and shutdown.
- The model seed is 42. Timing can change on another machine.

## Functional checks

**All 15 local tests passed.** The full test output is in results/tests.txt.
The checks include training-only preprocessing, unknown categories, constant and
empty columns, declared types, profile values, rare classes, prediction row order,
ID and evaluation-label exclusion, malformed inputs, and an enforced model deadline.

## Supplied course data

| Dataset and task | Train rows | Scored rows | Selected model | Result | Wall time |
| --- | ---: | ---: | --- | --- | ---: |
| Iris classification | 120 | 30 | logistic_regression | Balanced accuracy 93.33% | 10.91 s |
| Body performance classification | 10714 | 2679 | hist_gradient_boosting | Balanced accuracy 75.62% | 24.56 s |
| Life expectancy regression | 2342 | 586 | extra_trees | RMSE 1.606 years; R squared 0.970 | 13.37 s |
| Iris clustering | 150 | 150 | kmeans | Adjusted Rand index 0.620 | 7.35 s |

The supervised cases use an 80/20 outer split with random_state=42. The
classification splits are stratified. The program receives only the outer training
labels and makes its own internal validation split for model selection. The outer
test labels are opened by the evaluation script after predictions are saved.

The supplied Iris file is headerless and contains 150 rows. For clustering, all
150 feature rows are used and the species column is excluded. Species labels are
used only afterward to calculate the adjusted Rand index. This is an external
agreement check on the clustered rows, not a held-out classification accuracy.

The life expectancy file contains 10 rows without a target. These were removed
before splitting; feature missing values are handled by the program. Country and
year records can occur in both splits, so this is a classroom random-row check,
not a future-year or unseen-country forecasting result.

Baseline comparison:

| Dataset | Simple baseline | Program |
| --- | ---: | ---: |
| Iris classification | 33.33% balanced accuracy | 93.33% balanced accuracy |
| Body performance classification | 25.00% balanced accuracy | 75.62% balanced accuracy |
| Life expectancy regression | RMSE 9.308 years | RMSE 1.606 years |

Classification uses the training majority class as its baseline. Regression
uses the training mean. Higher balanced accuracy is better; lower RMSE is better.
The full measured metrics, including accuracy and MAE where applicable, are in
results/course_results.json.

Reproduce these checks using the separately supplied CSV files:

```bash
python evaluate_course_data.py --data-dir /path/to/course/files
```

## Generated examples

The examples use seed 21 for data generation and their outer 80/20 splits.
Classification includes imbalance, missing values, categories, duplicates, and
constant columns. Regression includes different feature scales and missing values.
Clustering uses three separated groups. Reference labels stay separate from inputs.

| Task | Selected model | Local result | Wall time |
| --- | --- | --- | ---: |
| classification | hist_gradient_boosting | Balanced accuracy 83.85% | 7.05 s |
| regression | ridge_regression | RMSE 10.461 target units | 10.20 s |
| clustering | kmeans | Adjusted Rand index 1.000 | 6.28 s |

Reproduce the examples with:

```bash
python evaluate_examples.py
```

## Public datasets linked in the brief

These three source CSVs were downloaded from their linked Kaggle dataset pages.
The same main.py was used without changes to the model or preprocessing code.
The source links, file requirements, and preparation choices are documented in
LINKED_DATASETS.md. CSV hashes and full metrics are in results/linked_results.json.

| Dataset | Train rows | Scored rows | Selected model | Result | Wall time |
| --- | ---: | ---: | --- | --- | ---: |
| Telco churn | 5634 | 1409 | logistic_regression | Balanced accuracy 75.26% | 4.94 s |
| AI4I machine failure | 8000 | 2000 | hist_gradient_boosting | Balanced accuracy 89.11% | 3.83 s |
| Mall customer groups | 200 | 200 | kmeans | Silhouette 0.555; 5 groups | 2.31 s |

Telco and AI4I use stratified 80/20 outer splits with seed 42. Outer test labels
are kept out of the model. The program performs a separate internal validation
split for model selection. Both majority baselines have 50% balanced accuracy.

| Classification dataset | Accuracy | Positive precision | Positive recall | Positive F1 |
| --- | ---: | ---: | ---: | ---: |
| Telco churn | 73.81% | 50.43% | 78.34% | 0.614 |
| AI4I machine failure | 98.15% | 70.13% | 79.41% | 0.745 |

The positive label is Yes for Telco and 1 for AI4I. Balanced accuracy is the
average recall across the two classes. AI4I is strongly imbalanced, so ordinary
accuracy should be read together with failure recall and precision.
The five AI4I failure subtype labels are removed before splitting, and IDs are
excluded from both classification models. The AI4I check is a random-row test
on synthetic machine records, not a real-world future-failure forecast.

Mall uses all 200 rows and only income and spending score, standardized through
the existing pipeline. Five clusters is our local choice. Silhouette is an
internal separation measure on those same rows; no ground-truth labels or
clustering accuracy are available. It is not comparable to classification scores.

Reproduce the Telco, AI4I, and Mall checks using the separately downloaded CSVs:

```bash
python evaluate_linked_data.py --data-dir /path/to/downloaded/csvs
```

## House Prices

The supplied ZIP contains 1,460 labeled training rows and 1,459 unlabeled test
rows. The separate archive.zip is the same Telco CSV already tested above.
The House check uses the same core program and a small source-format adapter.
Id is excluded and MSSubClass is declared categorical. Meaningful NA category
codes and the masonry category None are preserved as safe strings. Other
missing values are handled inside the training-only model pipelines.

### Held-out labeled rows

The outer split uses 80% of the labeled rows for training and 20% for scoring,
with seed 42. The model makes a separate internal split for model selection.
The original SalePrice is used without a target transform. Ordinary RMSE is
the model-selection metric; log-price RMSE is also reported after prediction.

| Item | Measured value |
| --- | ---: |
| Training / scored rows | 1168 / 292 |
| Selected model | extra_trees |
| RMSE | $28,891.90 |
| MAE | $18,004.62 |
| R squared | 0.8912 |
| Log-price RMSE | 0.1593 |
| Training-mean baseline RMSE | $87,619.03 |
| Wall time | 12.02 s |

Log-price RMSE uses natural logarithms of the actual and predicted prices,
without adding 1. These are local holdout results, not Kaggle leaderboard
scores. Nonpositive predictions would be counted and the log score left unset;
this run produced 0 such predictions.

### Predictions for the supplied test file

The program also trained on all 1,460 labeled rows and produced
1,459 test predictions in 15.75 seconds, using extra_trees.
The predictions are saved in results/house_prices_predictions.csv with columns
Id and SalePrice in the original test order. All predictions were checked for
finite values and the export was checked against the sample submission IDs.
The sample submission prices were not used as labels. The test file has no
true prices, so no accuracy or error score is claimed for those predictions.

Full metrics, file hashes, preparation choices, and timing are stored in
results/house_prices_results.json. Reproduce both House runs with:

```bash
python evaluate_house_prices.py --data-dir /path/to/extracted/house-prices
```

## Bike Sharing Demand

The supplied ZIP contains 10,886 labeled rows and 6,493 unlabeled test rows.
The source has no missing cells. Its count equals casual plus registered in
every labeled row, so both components are removed from the input features.
The adapter derives hour, weekday, month, and year from datetime and retains
the original timestamp separately for output. Season and weather are categorical.

### Holdout within each month

The original training file contains days 1-19 in each month across 2011-2012.
Our check fits on days 1-15 and scores days 16-19 in every month. This holds
out blocks of days, but it is not a strict future-only test: training includes
later months than some scored dates. Internal model selection still uses the
core program's random validation split within the outer training rows.

| Item | Measured value |
| --- | ---: |
| Training / scored rows | 8,600 / 2,286 |
| Selected model | hist_gradient_boosting |
| RMSE | 45.291 rentals |
| MAE | 27.355 rentals |
| R squared | 0.9388 |
| RMSLE | 0.4431 |
| Training-mean baseline RMSE | 183.077 rentals |
| Training-mean baseline RMSLE | 1.5313 |
| Wall time | 6.76 s |

Model selection uses ordinary RMSE on the original count target. A fixed
postprocessing rule clips negative predicted counts to zero before scoring
and exporting; fractional counts are retained. RMSLE uses log1p, unlike the
House Prices log metric. The metrics above use the clipped predictions.
The holdout run clipped 34 predictions; the raw minimum was -26.571.
Raw RMSE, MAE, and R squared are also recorded in the results JSON.

### Predictions for the supplied test file

The full run trained on 10,886 labeled rows and predicted
all 6,493 test rows in 8.09 seconds, using hist_gradient_boosting.
It clipped 118 negative predictions to zero.
The saved results/bike_sharing_predictions.csv contains datetime and count,
with original timestamps and row order preserved. The sample submission's
zero counts are placeholders and were not used as labels. No score is claimed
for the unlabeled test rows. Full metrics and source hashes are recorded in
results/bike_sharing_results.json. Reproduce both runs with:

```bash
python evaluate_bike_sharing.py --data-dir /path/to/extracted/bike-sharing
```

All five datasets linked in the brief have now been checked locally.

## Limits of these results

Each supervised dataset was checked with one fixed outer split. These numbers
are useful for checking the program, but they do not estimate a contest score.
The available brief leaves the exact profile schema, per-task quality formula,
dataset bounds, and assessment machine unspecified. This project documents its
CSV and JSON interface and its model-selection metrics. The included local
tests verify that interface; no separate official checker was available.

All 12 scored benchmark cases and the two separate unlabeled prediction runs completed their model search without hitting the model deadline. These checks were added in stages using the same core program.
