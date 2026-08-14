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
    patient_id: str = Field(..., example="CHR-0501")
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