import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


# -------------------------------------------------------------------
# Health Check Endpoint Tests
# -------------------------------------------------------------------

def test_health_check_status_ok():
    # Verify health check endpoint returns 200 OK and expected structure.
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "environment" in data


# -------------------------------------------------------------------
# Predict Endpoint - Success Tests
# -------------------------------------------------------------------

def test_predict_success_valid_payload():
    # Verify predict endpoint handles valid patient vital signs correctly.
    payload = {
        "patient_id": "TEST_PATIENT_001",
        "heart_rate": 110,
        "systolic_bp": 90,
        "oxygen_saturation": 92,
        "temperature_f": 101.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["patient_id"] == "TEST_PATIENT_001"
    assert "risk_score" in data
    assert "priority_level" in data
    assert "probabilities" in data
    assert isinstance(data["probabilities"], dict)


# -------------------------------------------------------------------
# Predict Endpoint - Data Validation & Failure Edge Cases
# -------------------------------------------------------------------

def test_predict_missing_patient_id():
    # Verify 422 Unprocessable Entity when patient_id is missing.
    payload = {
        "heart_rate": 80,
        "systolic_bp": 120,
        "oxygen_saturation": 98,
        "temperature_f": 98.6,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("patient_id" in err["loc"] for err in errors)


def test_predict_missing_vital_sign():
    # Verify 422 Unprocessable Entity when a vital sign is missing.
    payload = {
        "patient_id": "TEST_PATIENT_002",
        "heart_rate": 80,
        "systolic_bp": 120,
        # missing oxygen_saturation
        "temperature_f": 98.6,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_data_types():
    # Verify 422 Unprocessable Entity when non-numeric values are passed.
    payload = {
        "patient_id": "TEST_PATIENT_003",
        "heart_rate": "invalid_number",
        "systolic_bp": 120,
        "oxygen_saturation": 98,
        "temperature_f": 98.6,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422