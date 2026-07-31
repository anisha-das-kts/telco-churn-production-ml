# Mini Production ML System: Telco Customer Churn Prediction

## Project Overview

This repository implements an end-to-end production-style machine
learning system for predicting telecom customer churn.

The system includes:

- Batch data ingestion and upsert logic
- Schema and data-quality validation
- Shared feature engineering for training and serving
- Logistic Regression baseline
- Random Forest candidate
- Offline model evaluation and promotion guardrails
- FastAPI online inference service
- Latency and throughput benchmarking
- Data-quality and feature-drift monitoring
- Rule-based retraining triggers
- Automated tests
- Docker configuration

## Problem Definition

The objective is to predict whether an active telecom customer is
likely to churn.

The intended users are members of a telecom customer-retention team.
They can use churn probability and risk level to prioritise customers
for retention campaigns.

| Item | Definition |
|---|---|
| ML task | Binary classification |
| Positive class | Customer churns |
| Intended users | Customer-retention team |
| Input | Demographic, account, service and billing attributes |
| Output | Prediction, churn probability and risk level |
| Inference pattern | Online request-response API |
| p95 latency target | Below 200 ms |
| API error-rate target | Below 1% |
| Primary offline metric | ROC AUC |
| Important guardrail metric | Recall |

## Dataset

The project uses the public IBM Telco Customer Churn dataset.

- Rows: 7,043
- Original columns: 21
- Target: `Churn`
- Non-churn observations: 5,174
- Churn observations: 1,869

Dataset source:

[Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Each row represents one telecom customer. The data includes
demographics, subscribed services, tenure, contract details, payment
method, monthly charges, total charges and churn status.

## Data Quality and Cleaning

Initial validation identified:

- No missing required columns
- No unexpected columns
- No exact duplicate rows
- No duplicate customer IDs
- No negative tenure or charge values
- 11 blank `TotalCharges` values

The blank `TotalCharges` records belong to customers with zero tenure.
The shared preprocessing code replaces these values using:

```text
estimated_total_charges = MonthlyCharges × tenure
```

For zero-tenure customers, this produces a value of zero.

The original CSV is retained unchanged in `data/raw`.

## Engineered Features

The system creates 12 non-trivial features.

| Feature | Description |
|---|---|
| `avg_monthly_spend` | Lifetime charges divided by safe tenure |
| `service_count` | Number of active telecom services |
| `security_support_count` | Number of security, protection and support products |
| `streaming_service_count` | Number of streaming products |
| `charges_per_service` | Monthly charge divided by active-service count |
| `tenure_group` | Non-linear customer-lifecycle category |
| `is_month_to_month` | Month-to-month contract indicator |
| `has_auto_payment` | Automatic payment indicator |
| `has_internet` | Internet-service indicator |
| `support_gap` | Internet customer without technical support |
| `high_charge_short_tenure` | High-cost customer with short tenure |
| `contract_tenure_interaction` | Contract type combined with tenure group |

The final model receives 31 input features: 14 numeric and 17
categorical.

## Offline and Online Features

Raw account attributes such as tenure, contract and monthly charges
would be supplied online by a CRM system.

Derived features such as `service_count`, `support_gap` and
`charges_per_service` are calculated during inference.

In a larger production deployment, historical aggregations could be
precomputed offline and retrieved from a feature store.

## Prevention of Training-Serving Skew

The same function is used in every execution path:

```python
from src.features.build_features import build_features
```

It is called during:

- Offline training
- FastAPI prediction
- Batch prediction
- Monitoring

The complete categorical encoder, numeric imputer and classifier are
also stored together as a Scikit-learn pipeline. This prevents
differences in category handling, scaling, missing-value treatment and
feature formulas.

## Batch Ingestion

The ingestion pipeline reads CSV files from `data/incoming`.

For every new file, it:

1. Validates the required schema.
2. Checks IDs, target values and numeric ranges.
3. Separates valid and rejected records.
4. Adds ingestion timestamp and source filename.
5. Inserts new customers.
6. Updates existing customers using upsert logic.
7. Saves rejected rows for investigation.
8. Records the file hash in an ingestion manifest.

The file hash makes ingestion idempotent. Rerunning the pipeline does
not duplicate previously processed data.

Initial ingestion result:

| Measurement | Result |
|---|---:|
| Rows read | 7,043 |
| Valid rows | 7,043 |
| Rejected rows | 0 |
| Inserted rows | 7,043 |
| Final unique customers | 7,043 |

## Training Pipeline

The repeatable training sequence is:

```text
Load training data
→ Build shared features
→ Stratified train/validation/test split
→ Train baseline
→ Train candidate
→ Evaluate on validation data
→ Apply promotion guardrail
→ Refit selected model
→ Evaluate once on untouched test data
→ Save models and reports
```

Data split:

| Split | Rows |
|---|---:|
| Training | 4,929 |
| Validation | 1,057 |
| Test | 1,057 |

The random seed is fixed at `42`.

## Model Selection

### Baseline

Logistic Regression with:

- Numeric median imputation
- Numeric standardisation
- Categorical most-frequent imputation
- One-hot encoding
- Balanced class weights

### Candidate

Random Forest with:

- 300 trees
- Maximum depth of 12
- Minimum leaf size of 4
- Balanced subsample weights
- Fixed random seed

## Metric Selection

ROC AUC is the primary metric because it measures the model's ability
to rank likely churners above non-churners across classification
thresholds.

Recall is an important guardrail because false negatives represent
customers who churn without receiving a retention intervention.

Precision measures campaign efficiency. Low precision means retention
offers may be sent to customers who would not have churned.

Accuracy alone is insufficient because the target distribution is
imbalanced.

## Validation Results

| Model | Accuracy | ROC AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7540 | 0.8341 | 0.5252 | 0.7794 | 0.6275 |
| Random Forest | 0.7711 | 0.8319 | 0.5549 | 0.7011 | 0.6195 |

The candidate had higher accuracy and precision but lower ROC AUC,
recall and F1-score.

The promotion guardrail required:

- Candidate ROC AUC of at least 0.80
- Candidate ROC AUC to at least match the baseline
- Candidate recall of at least 0.70

Random Forest failed the baseline-comparison guardrail. Logistic
Regression was retained because it performed better on the primary
metric and recall while also being smaller, faster and more
interpretable.

## Final Test Results

The selected Logistic Regression model was refitted on the combined
training and validation data before final test evaluation.

| Metric | Result |
|---|---:|
| Accuracy | 0.7474 |
| ROC AUC | 0.8507 |
| Precision | 0.5145 |
| Recall | 0.8250 |
| F1-score | 0.6337 |
| True negatives | 559 |
| False positives | 218 |
| False negatives | 49 |
| True positives | 231 |

The test set was used only after model selection.

## Serving Pattern

The model is exposed through FastAPI using an online request-response
pattern.

Online inference is appropriate because a retention agent or CRM
workflow may require a score while viewing a customer record. The
payload is small, and the selected model supports low-latency CPU
inference.

For large campaign lists, the same model could also be used through a
batch-scoring script.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service navigation |
| GET | `/health` | Service and model health |
| GET | `/model-info` | Deployed model metadata |
| POST | `/predict` | Customer churn prediction |
| GET | `/docs` | Interactive Swagger documentation |

Example response:

```json
{
  "request_id": "example-request-id",
  "prediction": 1,
  "churn_label": "Likely to churn",
  "churn_probability": 0.93983,
  "risk_level": "high",
  "model_name": "Logistic Regression",
  "model_version": "1.0.0",
  "latency_ms": 34.788
}
```

Risk bands:

| Probability | Risk |
|---:|---|
| Below 0.40 | Low |
| 0.40–0.69 | Medium |
| 0.70 or above | High |

## API Performance

The local API was measured using five warm-up calls followed by 100
sequential requests.

| Measurement | Result |
|---|---:|
| Measured requests | 100 |
| Successful requests | 100 |
| Failed requests | 0 |
| Error rate | 0.00% |
| Average latency | 45.40 ms |
| Median latency | 44.68 ms |
| p95 latency | 55.55 ms |
| Maximum latency | 87.34 ms |
| Throughput | 22.00 requests/second |
| p95 target | 200 ms |
| Target met | Yes |

These measurements represent a local demonstration workload and are
not a substitute for distributed production load testing.

## Monitoring

### Infrastructure and API metrics

- Request count
- Average and p95 latency
- Throughput
- HTTP 4xx and 5xx counts
- Error rate
- Service health

Suggested alerts:

- p95 latency above 200 ms for 10 minutes
- Error rate above 1% for 5 minutes
- Two consecutive failed health checks

### Data and feature metrics

- Incoming row count
- Missing-value rates
- Missing required columns
- Negative or out-of-range values
- Unknown categories
- Numerical feature mean and standard deviation
- Normalised numerical mean shift

### Model and business metrics

- Prediction distribution
- High-risk customer rate
- ROC AUC on delayed labelled feedback
- Recall on recent labels
- Retention-offer acceptance rate
- Churn rate among contacted customers

## Controlled Monitoring Demonstration

Because the public dataset does not provide a genuinely newer
production batch, the repository includes a clearly labelled synthetic
monitoring demonstration.

The controlled batch contains:

- 500 sampled customers
- A 50% increase in `MonthlyCharges`
- An 8% missing rate in `MonthlyCharges`

The monitoring system detected:

- Missing-rate threshold violation
- `MonthlyCharges` z-shift of 1.104
- `charges_per_service` z-shift of 1.026

The synthetic batch is not presented as real recent production data.

## Retraining Strategy

Retraining is considered when any of these signals occurs:

1. At least 30 days have elapsed and at least 500 new labelled rows are
   available.
2. Recent ROC AUC drops by more than 0.05 from the production reference.
3. Numerical drift exceeds the configured threshold.

A monitoring demonstration triggered retraining because the maximum
z-shift was 1.104, exceeding the 0.50 threshold.

A newly trained model is not deployed automatically. It must pass the
same offline evaluation and promotion guardrails.

Critical schema or data-quality failures block retraining because
training on corrupted data could produce an invalid model.

## Incident Scenario

Suppose the billing system changes `MonthlyCharges` to
`monthly_charge` without notifying the ML team.

The ingestion schema check detects that the required column is
missing and rejects the batch instead of silently replacing the feature
with null values. A critical alert is sent to the data engineering and
ML teams.

The API continues using the last valid production model. The team
investigates the upstream schema contract, adds an approved mapping,
reprocesses the rejected batch and verifies data quality. Retraining is
allowed only after the corrected data passes quality and drift checks.

## Project Structure

```text
telco-churn-production-ml/
├── configs/
├── data/
│   ├── raw/
│   ├── incoming/
│   ├── processed/
│   └── monitoring/
├── src/
│   ├── data/
│   ├── features/
│   ├── training/
│   ├── serving/
│   └── monitoring/
├── scripts/
├── models/
├── artifacts/
├── tests/
├── docs/
├── demo/
├── Dockerfile
├── requirements.txt
├── requirements-lock.txt
└── README.md
```

## Local Setup

The project was developed using Python 3.12.10.

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For exact dependency versions:

```powershell
python -m pip install -r requirements-lock.txt
```

## Running the System

Validate raw data:

```powershell
python -m src.data.validate
```

Build feature preview:

```powershell
python -m src.features.build_features
```

Run ingestion:

```powershell
python -m src.data.ingest
```

Train and evaluate models:

```powershell
python -m src.training.train
```

Start the API:

```powershell
uvicorn src.serving.app:app --reload
```

Open Swagger:

```text
http://127.0.0.1:8000/docs
```

Benchmark the API:

```powershell
python scripts/benchmark_api.py --requests 100 --warmup 5
```

Create the controlled monitoring batch:

```powershell
python scripts/create_monitoring_batch.py
```

Run monitoring:

```powershell
python -m src.monitoring.drift
```

Evaluate retraining:

```powershell
python -m src.monitoring.retraining_trigger --days 10 --new-labeled-rows 100 --recent-auc 0.84
```

Run all tests:

```powershell
python -m pytest -q
```

Current result:

```text
29 passed
```

## Docker

A Dockerfile is included for optional container deployment.

```powershell
docker build -t telco-churn-api:1.0.0 .
docker run -p 8000:8000 telco-churn-api:1.0.0
```

Docker configuration was prepared but not executed in the original
local environment because Docker Desktop was unavailable.

## Reproducibility

- Python version documented
- Dependency ranges saved
- Exact dependency lock file saved
- Random seed fixed at 42
- Stratified splits
- Shared feature module
- Idempotent ingestion
- Saved model version
- JSON evaluation reports
- JSON monitoring reports
- Automated tests

## Limitations

- The dataset is fictional and may not represent current telecom users.
- Labels are not time-stamped, so a realistic temporal split is not
  possible.
- Monitoring uses a controlled synthetic recent batch.
- Current drift detection is univariate.
- The model uses a fixed 0.50 classification threshold.
- Prediction logs use local JSONL storage.
- The API benchmark is local and sequential.
- Fairness across demographic groups has not been fully assessed.
- Delayed production labels are not available in this demonstration.

## Future Work

- MLflow model registry
- Feast feature store
- PostgreSQL or object-storage training tables
- Prometheus and Grafana dashboards
- Evidently drift monitoring
- Airflow orchestration
- CI/CD through GitHub Actions
- SHAP explanations
- Probability calibration
- Business-cost-based threshold optimisation
- Champion-challenger deployment
- Canary rollout and automated rollback
- Cloud container deployment
- Fairness monitoring

## Repository

GitHub repository: [telco-churn-production-ml](https://github.com/anisha-das-kts/telco-churn-production-ml)