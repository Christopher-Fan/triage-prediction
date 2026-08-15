import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "Triage AI Classifier"
    APP_NAME: str = "Triage AI Classifier"
    APP_VERSION: str = "1.1.0"
    ENV: str = "production"
    DEBUG: bool = False
    
    # Path & File Configuration
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODEL_PATH: Path = BASE_DIR / "models" / "ktas_model.pkl"
    METADATA_PATH: Path = BASE_DIR / "models" / "model_metadata.pkl"
    DATA_PATH: Path = BASE_DIR / "data" / "data.csv"
    
    # ML & Training Settings
    RANDOM_SEED: int = 42
    N_OPTUNA_TRIALS: int = 15
    CV_FOLDS: int = 5
    
    # Feature Schema Baseline
    FEATURE_NAMES: list[str] = [
        "heart_rate",
        "systolic_bp",
        "oxygen_saturation",
        "temperature_f",
    ]
    
    # Database / Observability (if used with docker compose)
    POSTGRES_USER: str = "triage_user"
    POSTGRES_PASSWORD: str = "triage_pass"
    POSTGRES_DB: str = "triage_db"
    POSTGRES_HOST: str = "db"
    POSTGRES_PORT: int = 5432

    # Pydantic Settings Config (Loads .env files automatically)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Instantiate single global settings instance
settings = Settings()