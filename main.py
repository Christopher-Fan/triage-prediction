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