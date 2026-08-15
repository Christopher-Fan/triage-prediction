from contextlib import asynccontextmanager
from typing import Dict, List, Any
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, status
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter
from datetime import datetime
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from database import engine, Base, get_db
import patient_vital_and_prediction_models
from model import load_rf_model

# Initialize database tables on module loading
Base.metadata.create_all(bind=engine)

# In-memory storage for loaded models
ml_models: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Triage Classifier Service...")
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

# Prometheus FastAPI Instrument to communicate with Grafana visualization
Instrumentator().instrument(app).expose(app)

# Prometheus metric for KTAS Distribution
KTAS_PREDICTIONS = Counter(
    "ktas_predictions_total",
    "Total count of patient predictions broken down by KTAS score (1-5)",
    ["ktas_level"]  # Prometheus label/dimension
)

# --- Request/Response Schema ---
class PatientVitalRequest(BaseModel):
    patient_id: str = Field(..., json_schema_extra={"example":"CHR-0501"}, description="Unique patient identifier")
    heart_rate: float = Field(..., ge=30, le= 250, description= "Beats per minute")
    systolic_bp: float = Field(..., ge=50, le=250, description= "Systolic blood pressure (mmHg)")
    oxygen_saturation: float = Field(..., ge=50, le=100, description="Sp02 percentage")
    temperature_f: float = Field(..., ge=90.0, le=108.0, description="Body temperature in Fahrenheit")

class TriagePredictionResponse(BaseModel):
    record_id: int
    patient_id: str
    priority_level: str
    risk_score: int
    risk_factors: List[str]
    global_feature_importances: Dict[str, float]
    status: str
    created_at: datetime

class PatientHistoryRecord(BaseModel):
    id: int
    patient_id: str
    heart_rate: float
    systolic_bp: float
    oxygen_saturation: float
    temperature_f: float
    predicted_ktas_level: int
    risk_category: str
    created_at: datetime

    class Config: 
        from_attributes = True

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
async def predict_triage(payload: PatientVitalRequest, db: Session = Depends(get_db)):
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
    KTAS_PREDICTIONS.labels(ktas_level=str(prediction)).inc()

    #Map the prediction to priorities
    priority_map = {1: "RESUSCITATION", 2: "EMERGENT", 3: "URGENT", 4: "LESS URGENT", 5: "NON-URGENT"}
    priority_level = priority_map.get(prediction, "UNKNOWN")

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

    all_risk_factors = risk_factors if risk_factors else ["No acute anomalies detected"]

    # Save DB record to PostgreSQL Database
    db_record = patient_vital_and_prediction_models.TriageRecord(
        patient_id = payload.patient_id,
        heart_rate=payload.heart_rate,
        systolic_bp=payload.systolic_bp,
        oxygen_saturation=payload.oxygen_saturation,
        temperature_f=payload.temperature_f,
        predicted_ktas_level=prediction,
        risk_category=priority_level
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    return TriagePredictionResponse(
        record_id=db_record.id,
        patient_id = db_record.patient_id,
        priority_level = priority_level,
        risk_score = prediction,
        risk_factors = all_risk_factors,
        global_feature_importances=importances,
        status="SUCCESS",
        created_at=db_record.created_at
    )

# Fetch all historical triage records for a given patient ID
# Helps check for prior conditions, which is important for Emergency Situations
@app.get("/records/{patient_id}", response_model=List[PatientHistoryRecord])
async def get_patient_history(patient_id: str, db: Session = Depends(get_db)):
    records = db.query(patient_vital_and_prediction_models.TriageRecord).filter(
        patient_vital_and_prediction_models.TriageRecord.patient_id == patient_id
    ).order_by(patient_vital_and_prediction_models.TriageRecord.created_at.desc()).all()

    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"No triage history found for patient ID: {patient_id}"
        )
    return records