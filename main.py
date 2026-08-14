from contextlib import asynccontextmanager
from typing import Dict, List, Any
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from model import load_rf_model

# In-memory storage for loaded models
ml_models: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Triagle Classifier Service...")
    model, feature_importances = load_rf_model()
    ml_models["triage_classifier"] = model
    ml_models["feature_importances"] = feature_importances
    print("Succesfully loaded model and metadata")
    yield
    ml_models.clear()

app = FastAPI(
    title="Triage AI Classifier",
    description="Real-Time Emergency Patient Triage & Risk Analystics API",
    version="1.0.0",
    lifespan=lifespan
)

# --- Request/Response Schema ---
class PatientVitalRequest(BaseModel):
    patient_id: str = Field(..., json_schema_extra={"example":"CHR-0501"}, description="Unique patient identifier")
    heart_rate: float = Field(..., ge=30, le= 250, description= "Beats per minute")
    systolic_bp: float = Field(..., ge=50, le=250, description= "Systolic blood pressure (mmHg)")
    oxygen_saturation: float = Field(..., ge=50, le=100, description="Sp02 percentage")
    temperature_f: float = Field(..., ge=90.0, le=108.0, description="Body temperature in Fahrenheit")

class TriagePredictionResponse(BaseModel):
    patient_id: str
    priority_level: str
    risk_score: int
    risk_factors: List[str]
    global_feature_importances: Dict[str, float]
    status: str

# --- API Routes ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    #liveness probe to check for deadlocks
    model_loaded = "triage_classifier" in ml_models
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "service": "Triage AI Classifier"
    }

@app.post("/predict", response_model=TriagePredictionResponse)
async def predict_triage(payload: PatientVitalRequest):
    model = ml_models.get("triage_classifier")
    importances = ml_models.get("feature_importances", {})

    if not model:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ML Model has not properly been initialized"
        )

    # Set the features to match the column order of the model
    features = np.array([[
        payload.heart_rate,
        payload.systolic_bp,
        payload.oxygen_saturation,
        payload.temperature_f
    ]])

    # Execute the prediction of the payload
    prediction = int(model.predict(features)[0])

    # Rule-based Clinical Factors
    # Can have added features, but will need to add new feature columns in Model.py
    risk_factors = []

    if payload.oxygen_saturation < 92: 
        risk_factors.append("Hypoxia (SpO2 < 92%)")
    if payload.heart_rate > 100 or payload.heart_rate < 50:
        risk_factors.append("Abnormal Heart Rate")
    if payload.systolic_bp < 90 or payload.systolic_bp > 140:
        risk_factors.append("Abnormal Blood Pressure")
    if payload.temperature_f > 100.4: 
        risk_factors.append("Fever")

    priority_map = {0: "ROUTINE", 1: "URGENT", 2: "CRITICAL"}

    return TriagePredictionResponse(
        patient_id = payload.patient_id,
        priority_level = priority_map.get(prediction, "UNKNOWN"),
        risk_score = prediction,
        risk_factors = risk_factors if risk_factors else ["No acute anomalies detected"],
        global_feature_importances=importances,
        status="SUCCESS",
    )