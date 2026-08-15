import pytest
from fastapi.testclient import TestClient
from main import app, Base, engine

# Create SQLite tables in memory before running tests
Base.metadata.create_all(bind=engine)

@pytest.fixture(scope="module")
def client():
    # Context manager triggers FastAPI startup/lifespan events (e.g. loading model.pkl)
    with TestClient(app) as test_client:
        yield test_client

def test_health_endpoint(client):
    # Verify that the API server responds cleanly.
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_predict_endpoint_success(client):
    # Test POST /predict with valid vital signs across all KTAS features.
    payload = {
        "patient_id": "TEST-PAT-01",
        "heart_rate": 110.0,
        "systolic_bp": 135.0,
        "oxygen_saturation": 94.0,
        "temperature_f": 101.2
    }
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    # Validate response schema keys
    assert data["status"] == "SUCCESS"
    assert data["patient_id"] == "TEST-PAT-01"
    assert "priority_level" in data
    assert "risk_score" in data
    assert 1 <= data["risk_score"] <= 5

def test_predict_endpoint_invalid_payload(client):
    # Test Pydantic rejection on out-of-bounds/invalid inputs
    payload = {
        "patient_id": "TEST-INVALID",
        "heart_rate": -50.0,
        "systolic_bp": 120.0,
        "oxygen_saturation": 98.0,
        "temperature_f": 98.6
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_get_record_by_patient_id(client):
    # Test GET /records/{patient_id} retrieval after inserting a record
    patient_id = "TEST-LOOKUP-99"
    payload = {
        "patient_id": patient_id,
        "heart_rate": 72.0,
        "systolic_bp": 120.0,
        "oxygen_saturation": 98.0,
        "temperature_f": 98.6
    }
    
    # Insert record
    post_resp = client.post("/predict", json=payload)
    assert post_resp.status_code == 200
    
    # Query record back from DB
    get_resp = client.get(f"/records/{patient_id}")
    assert get_resp.status_code == 200

    fetched_data = get_resp.json()

    # Check that list returned contains at least one record
    assert isinstance(fetched_data, list)
    assert len(fetched_data) > 0
    assert fetched_data[0]["patient_id"] == patient_id