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
        if self.model_path and self.model_path.exists():
            self.model = joblib.load(self.model_path)
        else:
            self.model = None

    def _is_critical_vitals(self, hr: float, sbp: float, spo2: float) -> bool:
        # Clinical escalation rule for life-threatening physiological collapse.
        return spo2 < 88.0 or sbp < 80.0 or hr > 140.0

    def _is_strictly_normal_vitals(self, hr: float, sbp: float, spo2: float, temp: float) -> bool:
        # Clinical baseline guardrail for completely normal resting vital signs.
        return (60.0 <= hr <= 100.0) and (100.0 <= sbp <= 130.0) and (spo2 >= 97.0) and (97.0 <= temp <= 99.5)

    def predict(self, heart_rate: float, systolic_bp: float, oxygen_saturation: float, temperature_f: float) -> dict:
        if self.model is None:
            risk_score = 1 if oxygen_saturation < 88 or heart_rate > 140 else 5
            probs = {f"KTAS_{i}": (0.8 if i == risk_score else 0.05) for i in range(1, 6)}
        else:
            features = pd.DataFrame([{
                'heart_rate': heart_rate,
                'systolic_bp': systolic_bp,
                'oxygen_saturation': oxygen_saturation,
                'temperature_f': temperature_f
            }])
            
            raw_probs = self.model.predict_proba(features)[0]
            
            classes = getattr(self.model, "classes_", np.arange(len(raw_probs)))
            raw_class_pred = classes[np.argmax(raw_probs)]
            
            model_risk_score = int(raw_class_pred) if min(classes) == 1 else int(raw_class_pred) + 1

            # Rule Overrides: Enforce strict safety boundaries at extreme ends
            if self._is_critical_vitals(heart_rate, systolic_bp, oxygen_saturation):
                risk_score = 1
            elif self._is_strictly_normal_vitals(heart_rate, systolic_bp, oxygen_saturation, temperature_f) and model_risk_score <= 3:
                risk_score = 5
            else:
                risk_score = model_risk_score

            probs = {f"KTAS_{i+1}": float(p) for i, p in enumerate(raw_probs)}

        return {
            "risk_score": risk_score,
            "priority_level": PRIORITY_MAP.get(risk_score, "UNKNOWN"),
            "probabilities": probs
        }