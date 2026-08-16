import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


# -------------------------------------------------------------------
# Health Check Endpoint Tests
# -------------------------------------------------------------------

def test_health_check_status_ok():
    """Verify health check endpoint returns 200 OK and expected structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "environment" in data


# -------------------------------------------------------------------
# Predict Endpoint - Schema & Probability Contract Tests
# -------------------------------------------------------------------

def test_predict_success_valid_payload():
    """Verify predict endpoint handles valid patient vital signs and returns valid probability schema."""
    payload = {
        "patient_id": "TEST_PATIENT_001",
        "heart_rate": 110.0,
        "systolic_bp": 90.0,
        "oxygen_saturation": 92.0,
        "temperature_f": 101.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["patient_id"] == "TEST_PATIENT_001"
    assert "risk_score" in data
    assert "priority_level" in data
    assert "probabilities" in data
    
    # Contract Verification: Verify 5 KTAS probability classes summing to ~1.0
    probs = data["probabilities"]
    assert len(probs) == 5
    assert all(f"KTAS_{i}" in probs for i in range(1, 6))
    assert pytest.approx(sum(probs.values()), 0.01) == 1.0


# -------------------------------------------------------------------
# Predict Endpoint - Clinical Acuity Regression Tests
# -------------------------------------------------------------------
def _print_payload_and_response(title: str, payload: dict, response):
    """Helper function to print formatted test input and output."""
    print(f"\n{'='*20} {title} {'='*20}")
    print(f"INPUT PAYLOAD:\n{payload}")
    print(f"STATUS CODE: {response.status_code}")
    print(f"API RESPONSE:\n{response.json()}")
    print(f"{'='*50}\n")

def test_predict_normal_patient_output():
    # Outputs prediction results for a normal/low-acuity patient.
    payload = {
        "patient_id": "PATIENT_NORMAL_003",
        "heart_rate": 72.0,
        "systolic_bp": 120.0,
        "oxygen_saturation": 98.5,
        "temperature_f": 98.6,
    }
    response = client.post("/predict", json=payload)
    _print_payload_and_response("NORMAL PATIENT PREDICTION", payload, response)
    
    assert response.status_code == 200
    data = response.json()
    assert data["risk_score"] in [3, 4, 5]


def test_predict_emergency_patient_output():
    # Outputs prediction results for a high-acuity resuscitation patient.
    payload = {
        "patient_id": "PATIENT_EMERGENCY_001",
        "heart_rate": 145.0,
        "systolic_bp": 72.0,
        "oxygen_saturation": 84.0,
        "temperature_f": 104.2,
    }
    response = client.post("/predict", json=payload)
    _print_payload_and_response("EMERGENCY PATIENT PREDICTION", payload, response)
    
    assert response.status_code == 200
    data = response.json()
    assert data["risk_score"] in [1, 2]


def test_predict_urgent_patient_output():
    # Outputs prediction results for a moderate-acuity patient.
    payload = {
        "patient_id": "PATIENT_URGENT_002",
        "heart_rate": 108.0,
        "systolic_bp": 118.0,
        "oxygen_saturation": 95.0,
        "temperature_f": 101.8,
    }
    response = client.post("/predict", json=payload)
    _print_payload_and_response("URGENT PATIENT PREDICTION", payload, response)
    
    assert response.status_code == 200

# -------------------------------------------------------------------
# Predict Endpoint - Data Validation & Edge Cases
# -------------------------------------------------------------------

def test_predict_missing_patient_id():
    """Verify 422 Unprocessable Entity when patient_id is missing."""
    payload = {
        "heart_rate": 80.0,
        "systolic_bp": 120.0,
        "oxygen_saturation": 98.0,
        "temperature_f": 98.6,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("patient_id" in err["loc"] for err in errors)


def test_predict_missing_vital_sign():
    """Verify 422 Unprocessable Entity when a required vital sign is missing."""
    payload = {
        "patient_id": "TEST_PATIENT_002",
        "heart_rate": 80.0,
        "systolic_bp": 120.0,
        "temperature_f": 98.6,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_data_types():
    """Verify 422 Unprocessable Entity when non-numeric values are passed."""
    payload = {
        "patient_id": "TEST_PATIENT_003",
        "heart_rate": "invalid_number",
        "systolic_bp": 120.0,
        "oxygen_saturation": 98.0,
        "temperature_f": 98.6,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422