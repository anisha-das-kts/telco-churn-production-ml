# 🚀 Mini Production ML System: Telco Customer Churn Prediction

An end-to-end production-style machine learning system for predicting telecom customer churn, with data validation, feature engineering, model training, model selection, model registry, online inference, benchmarking, monitoring, drift detection, automated retraining decisions, and reproducible testing.

---

## 📌 Project Overview

This project implements a production-oriented machine learning lifecycle for predicting whether an active telecom customer is likely to churn.

The system is designed for a telecom customer-retention team that can use churn probability and risk level to prioritize customers for retention campaigns.

### Key Capabilities

* Batch data ingestion with idempotent processing
* Schema and data-quality validation
* Duplicate detection and upsert logic
* Shared feature engineering for training and serving
* Logistic Regression baseline
* Random Forest candidate model
* Offline model evaluation
* Model promotion guardrails
* Versioned model registry
* Champion model resolution and integrity checks
* FastAPI online inference service
* Model-level inference benchmarking
* API latency and throughput benchmarking
* Data-quality monitoring
* Numerical feature drift detection
* Rule-based retraining triggers
* Automated promotion and retraining tests
* Docker configuration
* Reproducible local execution
* Exploratory analysis notebook

---

# 🎯 Problem Definition

The objective is to predict whether a telecom customer is likely to churn.

| Item                       | Definition                                           |
| -------------------------- | ---------------------------------------------------- |
| ML task                    | Binary classification                                |
| Positive class             | Customer churns                                      |
| Intended users             | Customer-retention team                              |
| Input                      | Demographic, account, service and billing attributes |
| Output                     | Prediction, churn probability and risk level         |
| Inference pattern          | Online request-response API                          |
| Primary offline metric     | ROC AUC                                              |
| Important guardrail metric | Recall                                               |
| API p95 latency target     | < 200 ms                                             |
| API error-rate target      | < 1%                                                 |

---

# 📊 Dataset

The project uses the public IBM Telco Customer Churn dataset.

* **Rows:** 7,043
* **Original columns:** 21
* **Target:** `Churn`
* **Non-churn observations:** 5,174
* **Churn observations:** 1,869

Dataset source:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Each row represents one telecom customer and contains demographic, service, account, contract, payment and billing information.

---

# 📓 Exploratory Analysis Notebook

The project includes an analysis notebook:

```text
notebooks/telco_churn_analysis.ipynb
```

The notebook provides a reproducible analysis of the Telco Customer Churn dataset and can be used to inspect:

* Dataset structure
* Feature distributions
* Missing values
* Target-class distribution
* Customer tenure patterns
* Contract characteristics
* Billing and service attributes
* Churn-related patterns
* Exploratory visualisations
* Feature relationships relevant to modelling

The notebook complements the production pipeline by providing an analytical view of the underlying dataset before and alongside model development.

---

# 🧹 Data Quality and Cleaning

Initial validation identified:

* No missing required columns
* No unexpected columns
* No exact duplicate rows
* No duplicate customer IDs
* No negative tenure values
* No negative charge values
* 11 blank `TotalCharges` values

The 11 blank `TotalCharges` records correspond to zero-tenure customers.

The preprocessing pipeline replaces them using:

```text
estimated_total_charges = MonthlyCharges × tenure
```

For zero-tenure customers this produces zero.

The original raw dataset is retained unchanged.

---

# ⚙️ Feature Engineering

The system creates 12 engineered features:

| Feature                       | Description                                         |
| ----------------------------- | --------------------------------------------------- |
| `avg_monthly_spend`           | Lifetime charges divided by safe tenure             |
| `service_count`               | Number of active telecom services                   |
| `security_support_count`      | Number of security, protection and support products |
| `streaming_service_count`     | Number of streaming products                        |
| `charges_per_service`         | Monthly charge divided by active-service count      |
| `tenure_group`                | Customer lifecycle category                         |
| `is_month_to_month`           | Month-to-month contract indicator                   |
| `has_auto_payment`            | Automatic payment indicator                         |
| `has_internet`                | Internet-service indicator                          |
| `support_gap`                 | Internet customer without technical support         |
| `high_charge_short_tenure`    | High-cost customer with short tenure                |
| `contract_tenure_interaction` | Contract type combined with tenure group            |

The final model receives 31 input features:

* 14 numeric features
* 17 categorical features

---

# 🔄 Prevention of Training-Serving Skew

The same feature-engineering implementation is reused across all execution paths:

```python
from src.features.build_features import build_features
```

The shared feature function is used for:

* Offline training
* FastAPI prediction
* Batch prediction
* Monitoring

The preprocessing and classifier are also stored together as a Scikit-learn pipeline.

This ensures consistent:

* Missing-value handling
* Numeric scaling
* Categorical encoding
* Feature formulas
* Category handling

between training and inference.

---

# 📥 Batch Data Ingestion

Incoming CSV files are placed in:

```text
data/incoming/
```

The ingestion pipeline:

1. Validates the required schema
2. Checks customer IDs
3. Validates target values
4. Validates numerical ranges
5. Separates valid and rejected records
6. Adds ingestion metadata
7. Inserts new customers
8. Updates existing customers using upsert logic
9. Saves rejected rows
10. Records the source-file hash

The file hash makes ingestion idempotent.

Rerunning the same input does not duplicate previously ingested data.

### Initial Ingestion Result

| Measurement            | Result |
| ---------------------- | -----: |
| Rows read              |  7,043 |
| Valid rows             |  7,043 |
| Rejected rows          |      0 |
| Final unique customers |  7,043 |

---

# 🧠 Training Pipeline

The training workflow is:

```text
Load training data
       ↓
Build shared features
       ↓
Stratified train / validation / test split
       ↓
Train baseline
       ↓
Train candidate
       ↓
Evaluate validation performance
       ↓
Apply promotion guardrails
       ↓
Select production model
       ↓
Refit selected model
       ↓
Evaluate untouched test set
       ↓
Save model and reports
```

### Dataset Split

| Split      |  Rows |
| ---------- | ----: |
| Training   | 4,929 |
| Validation | 1,057 |
| Test       | 1,057 |

Random seed:

```text
42
```

---

# 🤖 Models

## Baseline — Logistic Regression

The baseline uses:

* Numeric median imputation
* Numeric standardisation
* Categorical most-frequent imputation
* One-hot encoding
* Balanced class weights

## Candidate — Random Forest

The candidate uses:

* 300 trees
* Maximum depth of 12
* Minimum leaf size of 4
* Balanced subsample weights
* Fixed random seed

---

# 📏 Metric Selection

## Primary Metric: ROC AUC

ROC AUC is used as the primary metric because the retention team needs to rank customers by churn likelihood across different classification thresholds.

## Guardrail Metric: Recall

Recall is important because a false negative represents a customer who churns without being identified for potential retention intervention.

## Additional Metrics

* Accuracy
* Precision
* F1-score

Accuracy is not used alone because the churn classes are imbalanced.

---

# 📊 Validation Results

| Model               | Accuracy | ROC AUC | Precision | Recall |     F1 |
| ------------------- | -------: | ------: | --------: | -----: | -----: |
| Logistic Regression |   0.7540 |  0.8341 |    0.5252 | 0.7794 | 0.6275 |
| Random Forest       |   0.7711 |  0.8319 |    0.5549 | 0.7011 | 0.6195 |

Although Random Forest achieved higher accuracy and precision, it performed worse on the primary ROC AUC metric and recall.

### Promotion Guardrails

A candidate must satisfy:

* ROC AUC ≥ 0.80
* ROC AUC ≥ baseline ROC AUC
* Recall ≥ 0.70

Random Forest failed the baseline-comparison guardrail.

Therefore:

```text
Selected production model:
Logistic Regression
```

Logistic Regression was retained because it performed better on the primary metric and recall while also providing a smaller, faster and more interpretable production model.

---

# 🏆 Final Test Results

The selected Logistic Regression model was refitted using the combined training and validation data.

The untouched test set was then used for final evaluation.

| Metric          | Result |
| --------------- | -----: |
| Accuracy        | 0.7474 |
| ROC AUC         | 0.8507 |
| Precision       | 0.5145 |
| Recall          | 0.8250 |
| F1-score        | 0.6337 |
| True negatives  |    559 |
| False positives |    218 |
| False negatives |     49 |
| True positives  |    231 |

The test set was used only after model selection.

---

# 🗂️ Model Registry

The project contains a versioned model registry with:

* Model versions
* Model metadata
* Evaluation metrics
* Champion resolution
* Promotion controls
* Champion integrity verification

A newly trained candidate is not automatically deployed.

Instead:

```text
Candidate Model
      ↓
Evaluation
      ↓
Promotion Guardrails
      ↓
Model Registry
      ↓
Champion Model
```

---

# 🚦 Model Promotion Guardrails

Promotion checks include:

* Minimum ROC AUC
* Candidate vs. champion ROC AUC
* Minimum recall
* Inference latency
* API error rate
* Required evaluation metrics

A candidate that fails any required production criterion is rejected.

This prevents an apparently improved model from being deployed if it violates operational or business constraints.

---

# 🌐 FastAPI Serving

The production model is exposed through FastAPI using an online request-response pattern.

Online inference is appropriate for a retention workflow where a churn score may be required while viewing an individual customer record.

## Start the API

```powershell
uvicorn src.serving.app:app --reload
```

The API runs at:

```text
http://127.0.0.1:8000
```

## Swagger Documentation

Open:

```text
http://127.0.0.1:8000/docs
```

---

# 🔌 API Endpoints

| Method | Endpoint      | Purpose                           |
| ------ | ------------- | --------------------------------- |
| GET    | `/`           | Service navigation                |
| GET    | `/health`     | Service and model health          |
| GET    | `/model-info` | Deployed model metadata           |
| POST   | `/predict`    | Customer churn prediction         |
| GET    | `/docs`       | Interactive Swagger documentation |

### Example Response

```json
{
  "request_id": "aba6c0f2-28f3-4ce6-afcc-466a085e1087",
  "prediction": 1,
  "churn_label": "Likely to churn",
  "churn_probability": 0.93983,
  "risk_level": "high",
  "model_name": "Logistic Regression",
  "model_version": "1.0.0",
  "latency_ms": 42.408
}
```

### Risk Bands

| Probability | Risk   |
| ----------: | ------ |
|      < 0.40 | Low    |
|   0.40–0.69 | Medium |
|      ≥ 0.70 | High   |

---

# ⚡ Model Inference Benchmark

The project includes a reusable model-level benchmarking utility:

```text
src/training/benchmark.py
```

It measures:

* Warm-up requests
* Measured requests
* Successful requests
* Failed requests
* Error rate
* Average inference latency
* Median latency
* p95 latency
* Minimum latency
* Maximum latency
* Throughput

The benchmark also validates invalid inputs and records partial inference failures.

---

# 📈 API Performance Benchmark

The API can be benchmarked using:

```powershell
python scripts/benchmark_api.py --requests 100 --warmup 5
```

The benchmark performs:

* 5 warm-up requests
* 100 measured requests
* Health verification before measurement
* Latency measurement
* Success/failure measurement
* Error-rate calculation
* Throughput calculation

Results are saved to:

```text
artifacts/eval/api_benchmark.json
```

## Latest Verified Benchmark Result

| Metric              |             Result |
| ------------------- | -----------------: |
| Requests measured   |                100 |
| Successful requests |                100 |
| Failed requests     |                  0 |
| Error rate          |              0.00% |
| Average latency     |           28.49 ms |
| Median latency      |           28.49 ms |
| P95 latency         |           31.41 ms |
| Minimum latency     |           24.37 ms |
| Maximum latency     |           76.22 ms |
| Throughput          | 35.06 requests/sec |
| P95 latency target  |           < 200 ms |
| Target status       |            **Met** |

### Benchmark Interpretation

The API successfully completed all 100 measured requests with zero failures.

The measured P95 latency of **31.41 ms** is substantially below the configured **200 ms** target.

The measured throughput was **35.06 requests/second** in the local sequential benchmark.

The benchmark represents a local sequential demonstration workload and is not a substitute for distributed production load testing.

---

# 🔍 Monitoring

The monitoring system covers three major areas.

## Infrastructure / API Metrics

* Request count
* Average latency
* p95 latency
* Throughput
* HTTP 4xx/5xx counts
* Error rate
* Service health

## Data and Feature Metrics

* Incoming row count
* Missing-value rates
* Missing required columns
* Invalid numerical values
* Unknown categories
* Numerical means and standard deviations
* Normalised numerical mean shift

## Model and Business Metrics

* Prediction distribution
* High-risk customer rate
* ROC AUC on delayed labels
* Recall on recent labels
* Retention-offer acceptance rate
* Churn rate among contacted customers

---

# 📡 Drift Detection

The project performs numerical feature monitoring against a reference dataset.

The controlled monitoring demonstration intentionally introduces:

* 500 recent rows
* 50% increase in `MonthlyCharges`
* 8% missing `MonthlyCharges`

The monitoring system detects:

* Missing-rate threshold violation
* `MonthlyCharges` z-shift of 1.104
* `charges_per_service` z-shift of 1.026

Configured numerical drift threshold:

```text
0.50
```

The monitoring report is saved to:

```text
artifacts/monitoring/monitoring_report.json
```

> **Note:** The monitoring batch is synthetic demonstration data and is not represented as genuine production data.

---

# 🔁 Automated Retraining Strategy

Retraining is considered when any of the following conditions occurs.

## 1. Scheduled Retraining

```text
At least 30 days elapsed
AND
At least 500 new labelled rows
```

## 2. Performance Degradation

```text
Recent ROC AUC decreases by more than 0.05
```

## 3. Data Drift

```text
Numerical drift exceeds configured threshold
```

The demonstrated monitoring batch triggered retraining because the maximum numerical z-shift exceeded the configured threshold.

### Retraining Safety

A newly trained model is **not automatically deployed**.

The candidate must pass the same:

* Offline evaluation
* Performance checks
* Recall checks
* Latency checks
* Error-rate checks
* Model promotion guardrails

before becoming the champion.

Critical data-quality failures block retraining.

---

# 🚨 Incident Scenario

Suppose an upstream billing system changes:

```text
MonthlyCharges
```

to:

```text
monthly_charge
```

without notifying the ML system.

The ingestion schema check detects the missing required column and rejects the batch rather than silently producing invalid features.

The production system continues using the last valid champion model while the upstream schema issue is investigated.

After the schema contract is corrected:

```text
Correct data
     ↓
Data quality validation
     ↓
Drift checks
     ↓
Retraining
     ↓
Evaluation
     ↓
Promotion guardrails
     ↓
Deployment
```

---

# 🧪 Automated Testing

The project currently contains **48 automated tests** covering:

* API endpoints
* Prediction validation
* Feature engineering
* Data ingestion
* Data-quality handling
* Model benchmarking
* Model registry
* Model promotion
* Retraining triggers
* Champion model serving

Run the complete test suite:

```powershell
python -m pytest -q
```

Expected result:

```text
48 passed
```

---

# 📁 Project Structure

```text
telco-churn-production-ml/
│
├── configs/
│
├── data/
│   ├── raw/
│   ├── incoming/
│   ├── processed/
│   └── monitoring/
│
├── notebooks/
│   └── telco_churn_analysis.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── training/
│   ├── serving/
│   ├── monitoring/
│   └── registry/
│
├── scripts/
│   ├── benchmark_api.py
│   ├── create_monitoring_batch.py
│   └── initialize_model_registry.py
│
├── models/
│
├── artifacts/
│   ├── eval/
│   ├── logs/
│   └── monitoring/
│
├── tests/
│
├── docs/
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── pytest.ini
├── requirements.txt
├── requirements-lock.txt
└── README.md
```

---

# 🛠️ Local Setup

The project was developed using:

```text
Python 3.12.10
```

## Create Virtual Environment

```powershell
py -3.12 -m venv .venv
```

## Activate

```powershell
.\.venv\Scripts\Activate.ps1
```

## Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For exact locked dependency versions:

```powershell
python -m pip install -r requirements-lock.txt
```

---

# ▶️ Running the Complete System

## 1. Validate Raw Data

```powershell
python -m src.data.validate
```

## 2. Build Feature Preview

```powershell
python -m src.features.build_features
```

## 3. Run Ingestion

```powershell
python -m src.data.ingest
```

## 4. Train and Evaluate Models

```powershell
python -m src.training.train
```

## 5. Run Automated Tests

```powershell
python -m pytest -q
```

## 6. Start FastAPI

```powershell
uvicorn src.serving.app:app --reload
```

## 7. Open Swagger

```text
http://127.0.0.1:8000/docs
```

## 8. Benchmark the API

In another terminal:

```powershell
python scripts/benchmark_api.py --requests 100 --warmup 5
```

## 9. Create Controlled Monitoring Batch

```powershell
python scripts/create_monitoring_batch.py
```

## 10. Run Monitoring

```powershell
python -m src.monitoring.drift
```

## 11. Evaluate Retraining

```powershell
python -m src.monitoring.retraining_trigger --days 10 --new-labeled-rows 100 --recent-auc 0.84
```

---

# 🐳 Docker

A Dockerfile is included for optional container deployment.

Build:

```powershell
docker build -t telco-churn-api:1.0.0 .
```

Run:

```powershell
docker run -p 8000:8000 telco-churn-api:1.0.0
```

Docker execution is optional for the local demonstration.

---

# ♻️ Reproducibility

The project includes:

* Python version documentation
* Dependency requirements
* Locked dependency versions
* Fixed random seed (`42`)
* Stratified dataset splitting
* Shared feature-engineering implementation
* Idempotent ingestion
* Versioned model artifacts
* JSON evaluation reports
* JSON monitoring reports
* Automated tests

---

# ⚠️ Limitations

This is a mini production-ML demonstration rather than a full enterprise deployment.

Current limitations include:

* The dataset is a public historical dataset and may not represent current telecom users.
* Labels are not time-stamped, so a realistic temporal split is not available.
* Monitoring uses controlled synthetic recent data.
* Drift detection is currently univariate.
* The classification threshold is fixed at 0.50.
* Prediction logs use local JSONL storage.
* API benchmarking is local and sequential.
* Fairness across demographic groups has not been fully assessed.
* Delayed real-world production labels are unavailable.

---

# 🔮 Future Improvements

Potential production extensions include:

* MLflow-based model lifecycle management
* Feast feature store
* PostgreSQL or object-storage training tables
* Prometheus and Grafana monitoring
* Evidently-based monitoring
* Airflow orchestration
* GitHub Actions CI/CD
* SHAP explanations
* Probability calibration
* Business-cost-based threshold optimisation
* Champion-challenger deployment
* Canary rollout and automated rollback
* Cloud container deployment
* Fairness monitoring

These are intentionally outside the scope of the current mini production-ML implementation.

---

# ✅ Current Validation Status

The current implementation has been validated locally with:

```text
Data validation              ✅
Feature engineering          ✅
Data ingestion               ✅
Model training               ✅
Model selection              ✅
Final evaluation             ✅
Model registry               ✅
Promotion guardrails         ✅
FastAPI serving              ✅
Health endpoint              ✅
Model benchmark              ✅
API benchmark                ✅ 100/100 successful
API P95 latency              ✅ 31.41 ms
API error rate               ✅ 0.00%
API throughput               ✅ 35.06 requests/sec
Monitoring                   ✅
Drift detection              ✅
Retraining decision          ✅
Automated tests              ✅ 48 passed
Docker configuration         ✅
Analysis notebook            ✅
```

---

# 👩‍💻 Repository

GitHub repository:

https://github.com/anisha-das-kts/telco-churn-production-ml

---

## 👤 Author

**Anisha Das**

Master's Student — Data Science & AI
