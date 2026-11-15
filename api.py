
import os
import json
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(
    ARTIFACT_DIR, "fraudlens_transaction_model.pkl"
)

METADATA_PATH = os.path.join(
    ARTIFACT_DIR, "model_metadata.json"
)

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]
DECISION_THRESHOLD = metadata["validation_selected_threshold"]

HIGH_RISK = metadata["risk_category_thresholds"]["high_risk"]
MEDIUM_RISK = metadata["risk_category_thresholds"]["medium_risk"]

app = FastAPI(
    title="FraudLens AI",
    description="AI-powered financial fraud detection API",
    version="1.0.0"
)

class Transaction(BaseModel):
    step: float = Field(ge=0)
    transaction_type: float = Field(ge=0, le=4)
    amount: float = Field(ge=0)
    log_amount: float = Field(ge=0)
    receiver_old_balance: float = Field(ge=0)
    receiver_zero_balance: float = Field(ge=0, le=1)
    sender_previous_txn_count: float = Field(ge=0)
    sender_previous_total_amount: float = Field(ge=0)
    sender_previous_avg_amount: float = Field(ge=0)
    sender_has_history: float = Field(ge=0, le=1)
    receiver_previous_txn_count: float = Field(ge=0)
    receiver_previous_total_amount: float = Field(ge=0)
    receiver_previous_avg_amount: float = Field(ge=0)
    receiver_has_history: float = Field(ge=0, le=1)

@app.get("/")
def home():
    return {
        "project": "FraudLens AI",
        "status": "operational",
        "message": "Financial fraud detection API is running"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict_fraud(transaction: Transaction):

    try:
        transaction_data = transaction.model_dump()

        input_df = pd.DataFrame(
            [transaction_data],
            columns=FEATURES
        )

        probability = float(model.predict_proba(input_df)[0][1])

        if probability >= HIGH_RISK:
            risk_category = "HIGH"
        elif probability >= MEDIUM_RISK:
            risk_category = "MEDIUM"
        else:
            risk_category = "LOW"

        prediction = int(probability >= DECISION_THRESHOLD)

        return {
            "project": "FraudLens AI",
            "fraud_probability": round(probability, 6),
            "fraud_probability_percent": round(probability * 100, 2),
            "predicted_fraud": prediction,
            "risk_category": risk_category,
            "decision_threshold": DECISION_THRESHOLD
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
