# FraudLens AI

End-to-end financial fraud analytics using the synthetic PaySim dataset: XGBoost risk scoring, SHAP explanations, account/network investigation, Evidently drift monitoring, MLflow tracking, FastAPI, Streamlit, Docker, and GitHub Actions.

## Evaluation
Chronological test results for the transaction-only model:
- PR-AUC: 0.554462
- Precision: 0.518474
- Recall: 0.694888
- F1: 0.593857
- Decision threshold: 0.981278

PaySim is synthetic. Test fraud prevalence (~1.40%) was substantially higher than training prevalence (~0.095%), indicating temporal distribution shift. Feature audits identified dataset-specific shortcuts. Graph features are used for investigation, not claimed as improving predictive performance. These metrics are not evidence of production performance on real banking data.

## Run
Install dependencies with `pip install -r requirements.txt`.
API: `uvicorn api:app --host 0.0.0.0 --port 8000`
Dashboard: `streamlit run app.py`

The model and parquet artifacts are intentionally not included. See `artifacts/README.md` for required files. The API requires `artifacts/fraudlens_transaction_model.pkl` and `artifacts/model_metadata.json`. The dashboard additionally requires the dashboard and test-feature parquet files. Monitoring requires the reference parquet file.

Never commit the raw dataset, private transaction data, credentials, or local MLflow databases.
