from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from database import Base

class TriageRecord(Base):
    __tablename__ = "triage_records"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True, nullable=False)
    heart_rate = Column(Float)
    systolic_bp = Column(Float)
    oxygen_saturation = Column(Float)
    temperature_f = Column(Float)
    predicted_ktas_level = Column(Integer)
    risk_category = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)