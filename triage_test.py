import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

# Verify health probe return HTTP 200 and indicates a loaded model
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

# Verify /predict processes a crticial patient payload
def test_predict_endpoint_valid_payload(client):
    payload = {
        "patient_id": "TES-TEST-01",
        "heart_rate": 135.0,
        "systolic_bp": 88.0,
        "oxygen_saturation": 89.0,
        "temperature_f": 102.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "TES-TEST-01"
    assert data["status"] == "SUCCESS"
    assert "Hypoxia (SpO2 < 92%)" in data["risk_factors"]


# Verify if the pydantic validation rejects impossible vital inputs (422 Can't Process Entity) 
def test_predict_endpoint_invalid_vitals(client):
    payload = {
        "patient_id": "TES-INVALID",
        "heart_rate": -50.0,  # Invalid heart rate
        "systolic_bp": 120.0,
        "oxygen_saturation": 98.0,
        "temperature_f": 98.6,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422