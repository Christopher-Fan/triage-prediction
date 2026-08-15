# Triage Prediction System
--------------------------
An end-to-end Machine Learning pipeline and microservice API for automated Emergency Department
patient triage scoring. Built with Scikit-Learn, FastAPI, PostgreSQL / SQLite, and Docker,
this service predicts triage acuity based on patient vital signs using the Korean Triage and Acuity Scale (KTAS 1–5).


This application automates the triage priority assignment process for emergency room admissions by 
processing key patient vitals:
Heart Rate (HR)
Systolic Blood Pressure (SBP)
Oxygen Saturation (SpO2)
Body Temperature (°F)
The machine learning core classifies patients into standardized KTAS priority levels, 
allowing healthcare workflows to dynamically identify high-risk cases.


KTAS Priority MAP for 
PRIORITY_MAP = {
    1: "RESUSCITATION", Immediate life or limb threat
    2: "EMERGENT", Potential threat to life, limb, or organ function
    3: "URGENT", Serious conditions with potential for disease progression
    4: "LESS URGENT", Stable conditions where intervention can be delayed
    5: "NON-URGENT", Chronic or minor complaints
}

===Tech Stack and Architecture ===
Machine Learning: Python, Scikit-Learn, Pandas | Random Forest Classifier predicting KTAS Levels (1-5)
Backend API: FastAPI, Pydantic, Uvicorn | RESTful API endpoint for prediciton and record lookup
Database: SQLAlchemy, PostgreSQL | ORM-backed persistent storage for patient triage records
Containers: Docker Compose | Containerized application deployment for reproducibility

QUICKSTART

Preqrequisites:
Docker Desktop installed and running
Python 3.10+

=== Build Commands ===
docker-compose up -d --build
docker-compose ps

=== API ===
API Live at http://localhost:8000
Interactive at http://localhost:8000/docs