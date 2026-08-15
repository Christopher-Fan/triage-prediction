from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.schemas import PredictRequest, PredictResponse
from src.api.metrics import KTAS_PREDICTIONS_COUNTER
from src.models.predictor import TriagePredictor
from src.config import settings

app = FastAPI(title=settings.PROJECT_NAME)
predictor = TriagePredictor()

# Instrument default HTTP metrics and expose endpoint
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

@app.get("/health")
def health_check():
    return {"status": "ok", "environment": settings.ENV}

@app.post("/predict", response_model=PredictResponse)
def predict_triage(payload: PredictRequest):
    try:
        res = predictor.predict(
            heart_rate=payload.heart_rate,
            systolic_bp=payload.systolic_bp,
            oxygen_saturation=payload.oxygen_saturation,
            temperature_f=payload.temperature_f
        )
        
        # Track metric custom counter
        KTAS_PREDICTIONS_COUNTER.labels(ktas_level=str(res["risk_score"])).inc()
        
        return PredictResponse(
            patient_id=payload.patient_id,
            risk_score=res["risk_score"],
            priority_level=res["priority_level"],
            probabilities=res["probabilities"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))