# FraudLens AI
### Graph-Based Financial Fraud Detection & Intelligence Platform

FraudLens AI is an end-to-end financial transaction fraud analytics project built around an imbalanced, time-ordered transaction dataset. It combines a machine-learning risk scorer with explainability, account/network investigation, drift monitoring, and an API/dashboard deployment workflow.

> **Research/portfolio project:** The model was evaluated on the synthetic PaySim dataset. Its results should not be interpreted as evidence of production performance on real banking data.

## Project highlights

- **Dataset:** PaySim synthetic mobile-money transactions (approximately 6.36 million rows).
- **Model:** XGBoost transaction-level classifier.
- **Explainability:** SHAP global and individual prediction explanations.
- **Investigation:** Account and transaction network exploration in the dashboard.
- **Monitoring:** Evidently-based feature drift checks, with MLflow experiment tracking.
- **Serving:** FastAPI prediction service and Streamlit dashboard.
- **Packaging:** Docker configuration and planned GitHub Actions CI.

## Evaluation

The final transaction-only model was evaluated using a chronological split to reduce temporal leakage.

| Metric | Test result |
|---|---:|
| PR-AUC | 0.554462 |
| Precision | 0.518474 |
| Recall | 0.694888 |
| F1-score | 0.593857 |
| Decision threshold | 0.981278 |

Fraud is rare in this dataset. The test set fraud prevalence was approximately 1.40%, compared with approximately 0.095% in the training set. This substantial temporal distribution shift matters when interpreting the metrics and designing monitoring.

### Important modelling caveats

PaySim is synthetic data and contains dataset-specific patterns. Fraud occurs only in the `CASH_OUT` and `TRANSFER` transaction types in this dataset, and feature audits identified strong synthetic shortcuts in some engineered features. The transaction-only model above is reported separately from the graph investigation functionality: graph features did not improve predictive performance in the experiments. The graph component is therefore positioned as an investigation/intelligence aid, not as a proven predictive-performance improvement.

These results are a reproducible portfolio experiment, not a validated banking fraud system. Real deployment would require representative institution-specific data, leakage and fairness reviews, calibrated thresholds, human oversight, security testing, and ongoing monitoring.

## Architecture

1. **Data preparation** — clean PaySim transactions and create chronological train/validation/test partitions.
2. **Feature engineering** — derive transaction and historical account-behaviour features using prior transaction history.
3. **Model training/evaluation** — train an XGBoost classifier and evaluate with precision-recall metrics.
4. **Explainability** — use SHAP to inspect global feature contributions and individual risk explanations.
5. **Investigation dashboard** — explore transactions, explanations, accounts, and network relationships.
6. **Monitoring** — compare feature distributions and log monitoring runs with MLflow.
7. **Serving** — expose prediction and health endpoints with FastAPI; package the service with Docker.

## Dataset

The project uses the PaySim dataset:

- Kaggle: [PaySim1 – Synthetic Financial Datasets for Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)
- Download the dataset from Kaggle and place the extracted CSV in the project data directory as described by the notebook/configuration.

**Do not commit the raw dataset** to this repository. Follow the dataset's Kaggle terms and attribution requirements when using or redistributing it.

## Repository layout

The intended repository structure is:

```text
FraudLens-AI/
├── api.py
├── app.py
├── monitoring_pipeline.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── artifacts/              # model and metadata; large/private artifacts excluded by default
├── notebooks/              # data preparation and model development
└── .github/
    └── workflows/          # automated checks
```

Some generated model/data artifacts may be too large or unsuitable for Git. Use Git LFS or a versioned artifact store if model distribution is needed; never commit datasets, credentials, tokens, or local tracking databases.

## Local setup

Create and activate a Python virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Run the API (once the model artifact is placed at the path configured by the application):

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open the interactive API documentation at `http://localhost:8000/docs`.

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

## Security and data handling

- Never commit API keys, access tokens, passwords, or `.env` files.
- Never commit the raw PaySim dataset or private transaction data.
- The dashboard-facing dataset is designed to exclude ground-truth fraud labels; do not expose labels in a live prediction interface.
- Treat model probabilities as risk scores, not proof of fraud. Human review is required for consequential decisions.

## Status

This repository is being assembled from the Google Colab project. Training, evaluation, dashboard, monitoring, and API components have been developed and tested in the Colab environment. Docker image build/run and automated GitHub Actions checks remain to be completed and verified.
