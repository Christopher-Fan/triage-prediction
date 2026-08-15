# Triage AI Classifier — Clinical Triage Prediction Service

A containerized Machine Learning pipeline and microservice API for automated Emergency Department patient triage scoring. Built with FastAPI, Scikit-Learn, Optuna, Docker Compose, and PostgreSQL, this service classifies emergency room patient acuity using the standardized Korean Triage and Acuity Scale (KTAS 1–5).

---

## Clinical Triage & Acuity Mapping

The system processes core patient vital signs and evaluates them against clinical urgency standards to assist medical staff in prioritizing high-risk cases:

| KTAS Level | Priority Category | Clinical Definition |
| :--- | :--- | :--- |
| **KTAS 1** | `RESUSCITATION` | Immediate life or limb threat requiring instant resuscitation |
| **KTAS 2** | `EMERGENT` | Potential threat to life, limb, or organ function requiring rapid intervention |
| **KTAS 3** | `URGENT` | Serious conditions with potential for disease progression or severe discomfort |
| **KTAS 4** | `LESS URGENT` | Stable conditions where medical intervention can be safely delayed |
| **KTAS 5** | `NON-URGENT` | Chronic or minor complaints |

---

## Tech Stack & System Architecture

* **Machine Learning Core:** Python, Scikit-Learn (`RandomForestClassifier`), Pandas, Optuna (Hyperparameter Optimization)
* **Backend API:** FastAPI, Pydantic, Uvicorn (Asynchronous REST API with schema validation)
* **Data Persistence:** SQLAlchemy, PostgreSQL / SQLite (ORM-backed patient audit trail)
* **Testing & Quality:** Pytest, FastAPI `TestClient`
* **Containerization:** Docker Compose (Multi-container orchestration for API, database, and observability)

```text
triage/
├── src/
│   ├── api/          # FastAPI routers, app factory, and middleware
│   ├── models/       # Training, Optuna tuning, and inference pipelines
│   └── config.py     # Pydantic-validated environment configuration
├── tests/            # Comprehensive pytest integration and unit test suite
├── reports/          # Generated Optuna convergence plots and feature importances
├── docker-compose.yml# Multi-service container orchestration
└── requirements.txt  # Project dependencies
```

## Observability & Monitoring (Prometheus & Grafana)

The microservice exposes production metrics via Prometheus middleware, visualized in real-time through Grafana dashboards.

* **API Health & Traffic:** Monitored via HTTP request count, latency histogram, and status code distributions (`200 OK` vs `422 Unprocessable Entity`).
* **Model Inference Metrics:** Real-time tracking of prediction throughput, KTAS acuity class distributions, and latency percentiles ($P_{50}$, $P_{95}$, $P_{99}$).
* **Database Connections:** Connection pool health and query latency tracking for persistent triage record audits.

| Service | Endpoint / URL | Purpose |
| :--- | :--- | :--- |
| **FastAPI Service** | `http://localhost:8000` | REST API for triage predictions |
| **Prometheus Metrics** | `http://localhost:8000/metrics` | Scraped metrics endpoint |
| **Prometheus UI** | `http://localhost:9090` | Targets & metric querying |
| **Grafana Dashboard** | `http://localhost:3000` | Real-time visual monitoring (`admin` / `admin`) |

## ML Pipeline & Hyperparameter Optimization

To prevent majority classes (non-urgent visits) from overshadowing high-acuity life threats, the model optimizes for Macro F1-Score using Stratified 5-Fold Cross-Validation.

### Optuna Tuning Configuration
Trials: 15 automated iterations searching hyperparameter space (n_estimators, max_depth, min_samples_split, min_samples_leaf).
Feature Importances: Ranked Gini importance highlighting Oxygen Saturation and Systolic Blood Pressure as primary clinical drivers of acute classification.

```text
# Build containers and start services in detached mode
docker-compose up -d --build

# Verify running containers
docker-compose ps
```

### Testing and Verification Suite
Local unit runtest is available using pytest:
```text
pytest tests/test_api.py -v
```

### API Usage Example
```text
$body = @{
    patient_id = "PATIENT_101"
    heart_rate = 110
    systolic_bp = 90
    oxygen_saturation = 92
    temperature_f = 101.2
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"
```
