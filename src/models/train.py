import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
)
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score

from src.config import settings
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = settings.BASE_DIR / "data" / "data.csv"
MODEL_SAVE_PATH = settings.BASE_DIR / "models" / "ktas_model.pkl"
METADATA_SAVE_PATH = settings.BASE_DIR / "models" / "model_metadata.pkl"
REPORTS_DIR = settings.BASE_DIR / "reports"


def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.Series]:
    # Load semicolon-delimited KTAS raw dataset, map feature names, and convert units.
    if not DATA_PATH.exists():
        print(f"  Dataset file '{DATA_PATH}' not found. Generating representative sample...")
        np.random.seed(42)
        n = 1500
        df = pd.DataFrame({
            "HR": np.random.normal(82, 20, n),
            "SBP": np.random.normal(122, 24, n),
            "Saturation": np.clip(np.random.normal(97, 4, n), 70, 100),
            "BT": np.random.normal(37.0, 0.8, n),
            "KTAS_expert": np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.04, 0.12, 0.44, 0.28, 0.12])
        })
    else:
        print(f"📄 Loading dataset from {DATA_PATH}...")
        try:
            df = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8")
        except (UnicodeDecodeError, pd.errors.ParserError):
            df = pd.read_csv(DATA_PATH, sep=";", encoding="latin1")

    feature_cols = ["HR", "SBP", "Saturation", "BT"]
    target_col = "KTAS_expert"

    # Filter required columns and enforce numeric types
    df_clean = df[feature_cols + [target_col]].copy()
    for col in feature_cols + [target_col]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    df_clean = df_clean.dropna(subset=feature_cols + [target_col])

    # Map to schema-aligned names & convert Celsius to Fahrenheit
    X = pd.DataFrame({
        "heart_rate": df_clean["HR"],
        "systolic_bp": df_clean["SBP"],
        "oxygen_saturation": df_clean["Saturation"],
        "temperature_f": (df_clean["BT"] * 9 / 5) + 32,
    })

    # Adjust KTAS 1-5 scale to 0-indexed (0-4) for ML modeling
    y = df_clean[target_col].astype(int) - 1

    return X, y


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    # Optuna objective function targeting Macro-F1 across Stratified K-Folds.
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "random_state": settings.RANDOM_SEED,
        "n_jobs": -1
    }

    clf = RandomForestClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=settings.RANDOM_SEED)
    
    # Target Macro F1-score to prioritize under-represented high-urgency cases
    scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")
    return scores.mean()


def generate_study_visualizations(study: optuna.Study, importances: Dict[str, float]):
    """Export interactive HTML and static PNG plots summarizing Optuna study results."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating study visualizations in reports/...")

    # 1. Optimization History Plot
    fig_history = plot_optimization_history(study)
    fig_history.write_html(str(REPORTS_DIR / "optuna_optimization_history.html"))
    try:
        fig_history.write_image(str(REPORTS_DIR / "optuna_optimization_history.png"))
    except Exception:
        pass  # Requires kaleido; html export acts as fallback

    # 2. Hyperparameter Importances Plot
    fig_param_imp = plot_param_importances(study)
    fig_param_imp.write_html(str(REPORTS_DIR / "optuna_param_importances.html"))
    try:
        fig_param_imp.write_image(str(REPORTS_DIR / "optuna_param_importances.png"))
    except Exception:
        pass

    # 3. Feature Importance Bar Chart (Matplotlib)
    plt.figure(figsize=(8, 5))
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_features]
    values = [item[1] for item in sorted_features]

    plt.barh(names[::-1], values[::-1], color="#2b5c8f")
    plt.xlabel("Gini Importance Score")
    plt.title("Random Forest Clinical Vitals Feature Importance")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importances.png", dpi=300)
    plt.close()

    print(f"✅ Visualizations successfully exported to {REPORTS_DIR}")


def run_training_pipeline():
    print("================================================")
    print("Starting Triage AI Classifier Optuna Optimization Study")
    print("================================================")

    X, y = load_and_preprocess_data()
    print(f"✅ Processed {len(X)} clinical records across 4 vital sign features.")

    # Execute hyperparameter tuning study
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=15)

    print("\n Best Optuna Trial:")
    print(f"  - Macro F1-Score: {study.best_value:.4f}")
    print("  - Hyperparameters:")
    for key, val in study.best_params.items():
        print(f"    * {key}: {val}")

    # Train final production model on full dataset using best params
    best_params = study.best_params
    best_params.update({"random_state": settings.RANDOM_SEED, "n_jobs": -1})
    
    final_model = RandomForestClassifier(**best_params)
    final_model.fit(X, y)

    # Compute Feature Importances
    importances = {
        feature: float(imp)
        for feature, imp in zip(X.columns, final_model.feature_importances_)
    }

    # Save visual artifacts
    generate_study_visualizations(study, importances)

    # Ensure output directories exist
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save model artifacts
    joblib.dump(final_model, MODEL_SAVE_PATH)
    joblib.dump({"feature_importances": importances, "macro_f1": study.best_value}, METADATA_SAVE_PATH)

    print(f"\n Model successfully exported to {MODEL_SAVE_PATH}")
    print("================================================\n")


if __name__ == "__main__":
    run_training_pipeline()