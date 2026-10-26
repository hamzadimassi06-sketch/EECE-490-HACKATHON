# Unknown Dataset Hackathon

EECE 490 — Introduction to Machine Learning

## Team

- Jad Kanj
- Brahim Sakka
- Hamza Dimassi
- Richard Jurdi

## What the program does

The idea is to use the same program for different tables. It first checks the
data, deals with missing values and categorical columns, and then tries a few
models. For classification and regression, it compares the models on a small
validation set. For clustering, it uses the number of groups given in the
metadata. It saves the predictions and a JSON data profile.

The hackathon brief is the specification available to our team. It describes
the tasks and outputs but does not give an exact file layout. We use CSV tables,
a small JSON metadata file, and `main.py` as the entry point. These choices form
the `student-v1` interface described below. The examples and checks in this
repository were made locally so the program can be run and reproduced.

## Setup

Use Python 3.12. The exact versions used for the local checks are pinned in
`requirements.txt`. No GPU, internet access, API keys, or downloaded model is
needed when the program runs.

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then install the packages:

```bash
python -m pip install -r requirements.txt
```

## Running the program

Run these commands from the project folder. The examples are small synthetic
datasets made for checking the program. They are not the nine official practice
datasets mentioned in the brief.

Classification:

```bash
python main.py --train examples/classification/train.csv --test examples/classification/test.csv --metadata examples/classification/metadata.json --output output/classification
```

Regression:

```bash
python main.py --train examples/regression/train.csv --test examples/regression/test.csv --metadata examples/regression/metadata.json --output output/regression
```

Clustering:

```bash
python main.py --train examples/clustering/data.csv --metadata examples/clustering/metadata.json --output output/clustering
```

For clustering, leaving out `--test` returns one cluster label for each row of
the input table. If a separate `--test` table is given, the model fits the
training table and assigns clusters to the test rows.

## Input format

Tables are UTF-8 CSV files with a header and unique column names. The supervised
training table includes the target column. The prediction table must include
all training feature columns, but its column order can differ. Prediction row
order is always preserved. Missing values can be empty cells, pandas' usual
CSV missing-value tokens, or blank strings.

Example classification metadata:

```json
{
  "task": "classification",
  "target": "churn",
  "feature_types": {
    "age": "numeric",
    "monthly_charge": "numeric",
    "contract": "categorical"
  },
  "id_columns": ["customer_id"],
  "metric": "balanced_accuracy"
}
```

| Field | Meaning |
| --- | --- |
| `task` | `classification`, `regression`, or `clustering` |
| `target` | Training label column for a supervised task |
| `feature_types` | Optional map from column names to `numeric` or `categorical` |
| `id_columns` | Optional list of columns to exclude from modelling |
| `n_clusters` | Required positive integer for clustering |
| `metric` | Optional model selection metric |

Feature types in metadata take priority. Remaining columns are inferred from
their training values. A text column that is at least 95% numeric among its
nonblank values is treated as numeric. Coded categories and numeric-looking
identifiers should therefore be declared explicitly.

The default selection metrics are balanced accuracy for classification, RMSE
for regression, and silhouette score for clustering. Classification also
accepts `accuracy` and `f1_macro`; regression also accepts `mae` and `r2`.
These are our model-selection defaults. The brief gives the overall point
weights but does not specify the exact quality formula for each task.

Alternatively, provide the task settings directly:

```bash
python main.py --train train.csv --test test.csv --task classification --target label --output output
python main.py --train data.csv --task clustering --clusters 3 --output output
```

`--task`, `--target`, and `--clusters` override the matching metadata values.
`--seed` defaults to 42. Unknown metadata fields and missing prediction features
are rejected to make input mistakes easier to notice.

## How it works

### Preparing the data

Numeric columns use median imputation and categorical columns use the most
common value. Missing indicators are added where missing values occurred in
training. Infinite numbers and invalid numeric entries are treated as missing.
The program removes empty, constant, and exactly duplicated feature columns.
If no usable columns remain, it creates a constant input so a fallback
prediction is still possible.

Numeric features are standardized for models that use distances or linear
coefficients. Most models use one-hot encoding for categories, with at most 40
encoded categories per original column. Unseen categories are handled without
crashing. Histogram gradient boosting uses ordinal encoding to avoid a large
dense one-hot table.
Sparse storage is used for low-density one-hot tables. Mostly numeric tables
stay dense, which avoids slowing down the tree models.

For supervised learning, these steps are fitted inside the model pipeline. During validation,
the program learns the median, scaling values, categories, and columns to drop
from the training part only. The final model is then refitted on the available
training data. Test rows and test labels are not used to select a model.

ID columns are dropped only when listed in the metadata. Unique values alone
are not enough to decide that a column is an ID. If a target column appears in
the test CSV, it is ignored. For clustering, a named target is excluded from
the features and its values are never used.

### Choosing a model

| Task | Models tried |
| --- | --- |
| Classification | Logistic regression, Extra Trees, histogram gradient boosting |
| Regression | Ridge regression, Extra Trees, histogram gradient boosting |
| Clustering | KMeans or MiniBatchKMeans, with a Gaussian mixture candidate when practical |

Extra Trees is a tree ensemble related to random forests. It gives the program
a way to learn nonlinear relationships without needing a neural network.
The classification models use class weights when the largest class has at
least 1.5 times as many rows as the smallest class.

The supervised candidates use the same validation split. Stratification is
used where possible, and very rare classes are kept in training. The program
uses one split rather than a large parameter search because there is a short
time limit. This makes the results somewhat dependent on that split. With fewer
than six rows, validation is skipped and the first model that fits is used.
A majority-class, mean, or median baseline is also checked. It can remain the
selected result if the candidate models do not improve on it.

For clustering, the number of clusters comes from the metadata. Preparation
and any reduction are fitted on the clustering training data only. More than
60 encoded features triggers reduction to at most 40 dimensions, where the
sample size allows it. Silhouette is only an internal selection
measure; it does not prove that the clusters match the private labels.

### Connection with the course

The KMeans part follows Chapter 7: assign each row to the closest center,
update the centers, and repeat. We try several initializations because the
starting centers can affect the result. The brief supplies the number of
clusters, so we use that number directly.

The tree models use the ensemble ideas in Chapter 10, where several trees are
combined for prediction. Extra Trees and histogram gradient boosting are the
scikit-learn variants chosen for this project; the lecture covers the related
ideas of random forests and boosting.

For wide clustering tables, PCA keeps fewer directions of variation, following
Chapter 13. Sparse tables use TruncatedSVD as an implementation choice. The
thresholds of 60 input features and at most 40 components are practical speed
settings chosen for this program.

### Time limit

The default budget is 50 seconds, including loading and profiling as measured
by the script. Set a smaller budget with `--time-limit`; the accepted range is
1 to 55 seconds. The program uses one computation thread and limits model
training to 20,000 rows when the table is larger. Sampling is recorded in
`run_info.json`; it can reduce accuracy on large datasets.
The cap can be exceeded if necessary to retain every class or to fit the
requested number of clusters.

The model work runs in a separate process. Linux uses `fork` to reduce repeated
import time; other platforms use `spawn`. The main process keeps the latest
completed predictions and stops the worker when its remaining budget ends.
It reserves 1.5 seconds for output. If no model finishes, it returns a clearly
reported fallback. These predictions may be weak, especially for clustering.

The model worker has an enforced timeout. CSV loading, full data profiling,
process startup, and output time still depend on the input size and computer.
The brief gives a 60-second assessment limit, but the exact machine and dataset
bounds are unspecified. The local results include measured wall time; timing
can change on a different machine or a much larger table.

## Output files

| File | Contents |
| --- | --- |
| `predictions.csv` | One `prediction` column, or one integer `cluster` column for clustering |
| `profile.json` | Training data profile before model columns are removed |
| `run_info.json` | Selected model, candidate validation results, sampling, timing, and any fallback |

The profile includes row and feature counts, feature types, missing and
unusable values, numeric summaries, category frequencies, empty/constant/
duplicate columns, duplicate rows, possible IDs, and a separate target
summary. Numeric summaries and duplicate checks use the declared or inferred
type conversions. The profile records this distinction.

No clustering labels are supplied to training. Cluster IDs are arbitrary
numbers starting at 0, so cluster 0 does not have a fixed meaning. If the data
has fewer distinct usable patterns than the requested number of clusters, a
model can produce fewer occupied groups; this is recorded rather than hidden.

## Checks and results

Run the local tests:

```bash
python -m unittest discover -s tests -v
```

They check preprocessing leakage, unseen categories, declared feature types,
empty and constant columns, profile accuracy, rare classes, valid JSON, output
row order, excluded IDs and labels, and invalid input files. These are our own
tests, not the official checker.

The example inputs can be recreated with:

```bash
python examples/make_examples.py
```

Run and score all three examples with:

```bash
python evaluate_examples.py
```

`reference_labels.csv` is only for measuring the example results after the
program finishes. It is never passed into the model. Local measured results
and their evaluation setup are in `LOCAL_RESULTS.md`.

To reproduce the checks on the separately supplied course CSVs:

```bash
python evaluate_course_data.py --data-dir /path/to/course/files
```

The course CSVs are not copied into this repository. The script accepts their
original filenames or the numbered attachment filenames.

### Datasets linked in the brief

The Word document links five datasets as examples of the scope. The first
link is Telco Customer Churn; it is a different dataset from AI4I.

| Dataset | Task | Target or clustering inputs |
| --- | --- | --- |
| [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) | Classification | `Churn` |
| [Predictive Maintenance AI4I 2020](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) | Classification | `Machine failure` |
| [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) | Regression | `SalePrice` |
| [Bike Sharing Demand](https://www.kaggle.com/competitions/bike-sharing-demand/data) | Regression | `count` |
| [Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) | Clustering | Annual income and spending score |

The three public dataset CSVs can be checked with:

```bash
python evaluate_linked_data.py --data-dir /path/to/downloaded/csvs
```

The script looks for the original filenames inside that folder and its
subfolders. It prepares the inputs, calls the same `main.py`, and measures
the results. AI4I's failure-type labels are removed before training. The Mall
example uses five clusters as a local choice; the assessment uses the number
given in its metadata.

See `LINKED_DATASETS.md` for the exact files, preparation choices, and current
testing status. The source CSVs are kept separately from this repository.

To check the separately downloaded House Prices files:

```bash
python evaluate_house_prices.py --data-dir /path/to/extracted/house-prices
```

This first measures predictions on an 80/20 split of the labeled training
CSV. It then trains on all labeled rows and predicts the supplied test CSV,
saving `house_prices_predictions.csv` with the original `Id,SalePrice` format.
The sample submission is only used to check the ID order and column names.
Its example prices are never used as labels. The adapter also preserves
meaningful category codes such as masonry `None` when reading the source CSVs.

For Bike Sharing:

```bash
python evaluate_bike_sharing.py --data-dir /path/to/extracted/bike-sharing
```

This removes `casual` and `registered`, derives calendar features from
`datetime`, and tests on days 16-19 of each month after fitting on days 1-15.
It also predicts the supplied test file, preserving the original timestamps
in `bike_sharing_predictions.csv`. Rental predictions below zero are clipped
to zero by the dataset adapter, and the number changed is recorded.
All five linked datasets now have local checks; results and the two regression
prediction files are included under `results`.

## Main files

- `main.py`: command-line arguments, input checks, and output files.
- `preprocessing.py`: cleaning, imputation, scaling, and encoding.
- `data_profile.py`: structured description of the training data.
- `models.py`: model comparison, fitting, and the model timeout.
- `tests/test_project.py`: local checks.
- `evaluate_linked_data.py`: checks on Telco, AI4I, and Mall source CSVs.
- `evaluate_house_prices.py`: a House Prices holdout check and test predictions.
- `evaluate_bike_sharing.py`: a Bike Sharing date-based check and test predictions.
- `LINKED_DATASETS.md`: the five source links and how the examples are used.

## Submission

The commands above reproduce the program using the documented interface.
The team names, dependencies, examples, local tests, and measured results are
included in this folder.

Create a public GitHub repository and upload the contents of this folder,
including the README, Python files, requirements, examples, and tests. The
submission is that repository URL. Generated `output` folders and private
assessment data do not need to be uploaded.

## References

The approach uses the supplied EECE 490 material on data preparation,
classification/regression, and model validation, together with these chapters:

- Chapter 7 KMeans, slides 20–27 and 39–42: the clustering procedure,
  initialization sensitivity, and the Gaussian-model comparison.
- Chapter 10 Decision Trees, slides 99–111 and 113–118: tree ensembles,
  random forests, and boosting.
- Chapter 13 PCA, slides 38–43: mean centering, covariance, component selection,
  and projection into a lower-dimensional space.

Library implementation references:

- [Avoiding data leakage](https://scikit-learn.org/1.8/common_pitfalls.html)
- [Pipelines and column transformers](https://scikit-learn.org/1.8/modules/compose.html)
- [Clustering methods](https://scikit-learn.org/1.8/modules/clustering.html)
- [Tree ensembles](https://scikit-learn.org/1.8/modules/ensemble.html)
