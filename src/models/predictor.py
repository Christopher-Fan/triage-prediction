import joblib
import numpy as np
import pandas as pd
from src.config import settings

PRIORITY_MAP = {
    1: "RESUSCITATION",
    2: "EMERGENCY",
    3: "URGENT",
    4: "LESS-URGENT",
    5: "NON-URGENT"
}

class TriagePredictor:
    def __init__(self, model_path=None):
        self.model_path = model_path or settings.MODEL_PATH
        self.model = None
        self._load_model()

    def _load_model(self):
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
        else:
            self.model = None

    def predict(self, heart_rate: float, systolic_bp: float, oxygen_saturation: float, temperature_f: float) -> dict:
        if self.model is None:
            # Fallback heuristic for uninitialized / testing states
            risk_score = 1 if oxygen_saturation < 88 or heart_rate > 140 else 3
            probs = {f"KTAS_{i}": (0.8 if i == risk_score else 0.05) for i in range(1, 6)}
        else:
            features = pd.DataFrame([{
                'heart_rate': heart_rate,
                'systolic_bp': systolic_bp,
                'oxygen_saturation': oxygen_saturation,
                'temperature_f': temperature_f
            }])
            risk_score = int(self.model.predict(features)[0])
            raw_probs = self.model.predict_proba(features)[0]
            probs = {f"KTAS_{i+1}": float(p) for i, p in enumerate(raw_probs)}

        return {
            "risk_score": risk_score,
            "priority_level": PRIORITY_MAP.get(risk_score, "UNKNOWN"),
            "probabilities": probs
        }