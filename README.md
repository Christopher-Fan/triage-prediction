# Triage AI Classifier — Clinical Emergency Department Triage Classifier

An end-to-end Data Science pipeline and microservice API for automated Emergency Department patient triage scoring. Built with FastAPI, Scikit-Learn, Optuna, Docker Compose, and PostgreSQL, this service classifies patient acuity using the standardized Korean Triage and Acuity Scale (KTAS 1–5).

---

## Key Exploratory Data Analysis (EDA) Insights

Data analysis of patient vital sign distributions revealed critical non-linear relationships with clinical urgency:

* **Oxygen Saturation ($\text{SpO}_2$):** Primary driver of acute classification. $\text{SpO}_2 < 90\%$ exhibits an exponential correlation with KTAS 1 (Resuscitation) and KTAS 2 (Emergent) levels.
* **Systolic Blood Pressure (SBP):** Demonstrates a distinct **U-shaped risk profile**—both extreme hypotension ($\text{SBP} < 90\text{ mmHg}$, risk of shock) and hypertensive crises ($\text{SBP} > 180\text{ mmHg}$) trigger high-acuity scoring, whereas moderate ranges remain non-urgent.
* **Vital Sign Interaction:** Strong interaction between heart rate and core temperature capturing compensatory tachycardia and septic response patterns.

> *For full distribution plots, data provenance details, and feature interaction analysis, see [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb).*

---

## Model Training & Evaluation Results

To prevent majority classes (KTAS 3/4) from obscuring life-threatening presentations, hyperparameter optimization prioritizes **Macro F1-Score** evaluated via **Stratified 5-Fold Cross-Validation**.

| Model / Configuration | Cross-Validation Scheme | Macro F1-Score | Resuscitation (KTAS 1) Sensitivity |
| :--- | :--- | :--- | :--- |
| Logistic Regression (Baseline) | Stratified 5-Fold | 0.2366 | 50.0% |
| Default Random Forest | Stratified 5-Fold | 0.2547 | 0.0% |
| **Optuna-Tuned Random Forest** | **Stratified 5-Fold** | **0.2715** | **30.0%** |

> *Detailed validation methodology and metric ablation available in [`reports/validation_ablation.md`](reports/validation_ablation.md).*

---

## Clinical Triage & Acuity Mapping

The system processes core patient vital signs and evaluates them against clinical urgency standards:

| KTAS Level | Priority Category | Clinical Definition |
| :--- | :--- | :--- |
| **KTAS 1** | `RESUSCITATION` | Immediate life or limb threat requiring instant resuscitation |
| **KTAS 2** | `EMERGENT` | Potential threat to life, limb, or organ function requiring rapid intervention |
| **KTAS 3** | `URGENT` | Serious conditions with potential for disease progression or severe discomfort |
| **KTAS 4** | `LESS URGENT` | Stable conditions where medical intervention can be safely delayed |
| **KTAS 5** | `NON-URGENT` | Chronic or minor complaints |

---

## Tech Stack & System Architecture

* **Machine Learning & Analysis:** Python, Scikit-Learn, Pandas, Optuna, Matplotlib, Plotly
* **Backend API:** FastAPI, Pydantic, Uvicorn (Asynchronous REST API with schema validation)
* **Data Persistence:** SQLAlchemy, PostgreSQL / SQLite (ORM-backed patient audit trail)
* **Testing & Quality:** Pytest, FastAPI `TestClient`
* **Containerization & Observability:** Docker Compose, Prometheus, Grafana

```text
triage/
├── docs/             # Validation scheme ablation study and methodology docs
├── notebooks/        # Cleaned EDA and feature analysis notebooks
├── reports/          # Human-readable study reports and exported visual plots
├── src/              # Modular source code
│   ├── api/          # FastAPI routers, app factory, and middleware
│   ├── models/       # Training, Optuna tuning, and inference pipelines
│   └── config.py     # Pydantic-validated environment configuration
├── tests/            # Comprehensive pytest integration and unit test suite
├── docker-compose.yml# Multi-service container orchestration
└── requirements.txt  # Project dependencies
```

## ML Pipeline & Hyperparameter Optimization
To prevent majority classes (non-urgent visits) from overshadowing high-acuity life threats, the model optimizes for Macro F1-Score using Stratified 5-Fold Cross-Validation.

### Optuna Tuning Configuration
Trials: 15 automated iterations searching hyperparameter space (n_estimators, max_depth, min_samples_split, min_samples_leaf).
Feature Importances: Ranked Gini importance highlighting Oxygen Saturation and Systolic Blood Pressure as primary clinical drivers of acute classification.

## Quickstart and Installation
Prerequisites:
- Docker Desktop installed and running
- Python 3.10+

Build & Run via Docker Compose
```text
# Build containers and start services in detached mode
docker-compose up -d --build

# Verify running containers
docker-compose ps
```

API is live at http://localhost:8000
Grafana Observability at http://localhost:3000 "admin"/"admin"
Prometheus Query at http://localhost:9090

## Testing and Verification
Testing suite to test for edge cases in pytest and training showcase
```text
pytest tests/test_api.py -v

python -m src.models.train
```

## API USAGE EXAMPLE
BASH COMMAND:
```text
$body = @{
    patient_id = "PATIENT_101"
    heart_rate = 110
    systolic_bp = 90
    oxygen_saturation = 92
    temperature_f = 101.2
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"```
```

