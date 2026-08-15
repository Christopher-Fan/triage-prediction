from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    patient_id: str = Field(..., example="PAT-9204")
    heart_rate: float = Field(..., ge=20.0, le=250.0, description="Heart rate in BPM")
    systolic_bp: float = Field(..., ge=40.0, le=260.0, description="Systolic Blood Pressure in mmHg")
    oxygen_saturation: float = Field(..., ge=50.0, le=100.0, description="Oxygen Saturation SpO2 %")
    temperature_f: float = Field(..., ge=90.0, le=110.0, description="Body Temperature in Fahrenheit")

class PredictResponse(BaseModel):
    patient_id: str
    risk_score: int = Field(..., description="KTAS priority level (1=Resuscitation, 5=Non-urgent)")
    priority_level: str
    probabilities: dict[str, float]