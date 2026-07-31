# Offline Model Evaluation

## Validation-set comparison

| Model | Accuracy | ROC AUC | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7540 | 0.8341 | 0.5252 | 0.7794 | 0.6275 |
| Random Forest | 0.7711 | 0.8319 | 0.5549 | 0.7011 | 0.6195 |

## Promotion guardrail

- Candidate ROC AUC at least minimum AUC: True
- Candidate ROC AUC at least matches the baseline: False
- Candidate recall at least minimum recall: True

## Decision

**Retain Logistic Regression baseline**

Selected production model: **Logistic Regression**

## Final untouched test-set results

| Metric | Result |
|---|---:|
| Accuracy | 0.7474 |
| ROC AUC | 0.8507 |
| Precision | 0.5145 |
| Recall | 0.8250 |
| F1-score | 0.6337 |

The promotion decision was made using validation data. The test
set was used only once after model selection to estimate final
generalisation performance.
