from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
)
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import settings

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = settings.BASE_DIR / "data" / "data.csv"
MODEL_SAVE_PATH = settings.BASE_DIR / "models" / "ktas_model.pkl"
METADATA_SAVE_PATH = settings.BASE_DIR / "models" / "model_metadata.pkl"
REPORTS_DIR = settings.BASE_DIR / "reports"


def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.Series]:
    # Load semicolon-delimited KTAS raw dataset, map feature names, and convert units.
    if not DATA_PATH.exists():
        print(
            f" Dataset file '{DATA_PATH}' not found. Generating clinically realistic representative sample..."
        )
        np.random.seed(settings.RANDOM_SEED)
        n = 2000

        # Generate KTAS levels according to clinical distribution
        ktas = np.random.choice(
            [1, 2, 3, 4, 5], size=n, p=[0.05, 0.15, 0.45, 0.25, 0.10]
        )

        hr, sbp, spo2, bt = [], [], [], []

        for level in ktas:
            if level == 1:  # Resuscitation (Severe shock/hypoxia/tachycardia)
                hr.append(np.random.normal(135, 18))
                sbp.append(np.random.normal(78, 12))
                spo2.append(np.clip(np.random.normal(82, 6), 65, 91))
                bt.append(np.random.normal(38.2, 1.2))
            elif level == 2:  # Emergent
                hr.append(np.random.normal(115, 15))
                sbp.append(np.random.normal(95, 15))
                spo2.append(np.clip(np.random.normal(91, 3), 85, 95))
                bt.append(np.random.normal(37.8, 1.0))
            elif level == 3:  # Urgent
                hr.append(np.random.normal(92, 12))
                sbp.append(np.random.normal(118, 14))
                spo2.append(np.clip(np.random.normal(96, 2), 92, 99))
                bt.append(np.random.normal(37.2, 0.8))
            elif level == 4:  # Less Urgent
                hr.append(np.random.normal(78, 10))
                sbp.append(np.random.normal(124, 12))
                spo2.append(np.clip(np.random.normal(98, 1), 95, 100))
                bt.append(np.random.normal(36.8, 0.5))
            else:  # Non-Urgent (KTAS 5)
                hr.append(np.random.normal(72, 8))
                sbp.append(np.random.normal(120, 10))
                spo2.append(np.clip(np.random.normal(99, 1), 96, 100))
                bt.append(np.random.normal(36.6, 0.4))

        df = pd.DataFrame({
            "HR": hr,
            "SBP": sbp,
            "Saturation": spo2,
            "BT": bt,
            "KTAS_expert": ktas,
        })
    else:
        print(f" Loading dataset from {DATA_PATH}...")
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


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def run_ablation_study(
    X: pd.DataFrame, y: pd.Series, best_rf_params: Dict[str, Any]
) -> pd.DataFrame:
    # Executes Stratified 5-Fold ablation study comparing baseline and tuned models.
    print("\nRunning Ablation & Benchmarking Study across Stratified 5-Fold CV...")

    # Ensure class_weight="balanced" is included in tuned RF
    tuned_rf_params = best_rf_params.copy()
    tuned_rf_params.update({"class_weight": "balanced", "random_state": settings.RANDOM_SEED, "n_jobs": -1})

    models = {
        "Logistic Regression (Baseline)": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000, random_state=settings.RANDOM_SEED, class_weight="balanced"
            )
        ),
        "Default Random Forest": RandomForestClassifier(
            random_state=settings.RANDOM_SEED, n_jobs=-1, class_weight="balanced"
        ),
        "Optuna-Tuned Random Forest": RandomForestClassifier(**tuned_rf_params),
    }

    skf = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=settings.RANDOM_SEED
    )
    ablation_results = []

    for model_name, clf in models.items():
        macro_f1s = []
        ktas1_sensitivities = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            clf.fit(X_train, y_train)
            preds = clf.predict(X_val)

            macro_f1s.append(f1_score(y_val, preds, average="macro"))

            # Evaluate KTAS 1 Recall (Class Index 0)
            recalls = recall_score(
                y_val, preds, average=None, labels=[0], zero_division=0
            )
            ktas1_sensitivities.append(recalls[0])

        avg_macro_f1 = float(np.mean(macro_f1s))
        avg_ktas1_sens = float(np.mean(ktas1_sensitivities))

        ablation_results.append({
            "Model / Configuration": model_name,
            "Cross-Validation": "Stratified 5-Fold",
            "Macro F1-Score": round(avg_macro_f1, 4),
            "Resuscitation (KTAS 1) Sensitivity": f"{avg_ktas1_sens * 100:.1f}%",
        })

    results_df = pd.DataFrame(ablation_results)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(REPORTS_DIR / "ablation_study.csv", index=False)

    print("\n--- Ablation Study Results ---")
    print(results_df.to_string(index=False))
    print("---------------------------------\n")

    return results_df


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "random_state": settings.RANDOM_SEED,
        "n_jobs": -1,
        "class_weight": "balanced"  # Critical addition for real, imbalanced data
    }

    clf = RandomForestClassifier(**params)
    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=settings.RANDOM_SEED
    )

    scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")
    return float(scores.mean())


def generate_study_visualizations(
    study: optuna.Study, importances: Dict[str, float]
):
    # Export interactive HTML and static PNG plots summarizing Optuna study results.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating study visualizations in reports/...")

    # 1. Optimization History Plot
    fig_history = plot_optimization_history(study)
    fig_history.write_html(
        str(REPORTS_DIR / "optuna_optimization_history.html")
    )
    try:
        fig_history.write_image(
            str(REPORTS_DIR / "optuna_optimization_history.png")
        )
    except Exception:
        pass  # Requires kaleido; html export acts as fallback

    # 2. Hyperparameter Importances Plot
    fig_param_imp = plot_param_importances(study)
    fig_param_imp.write_html(
        str(REPORTS_DIR / "optuna_param_importances.html")
    )
    try:
        fig_param_imp.write_image(
            str(REPORTS_DIR / "optuna_param_importances.png")
        )
    except Exception:
        pass

    # 3. Feature Importance Bar Chart (Matplotlib)
    plt.figure(figsize=(8, 5))
    sorted_features = sorted(
        importances.items(), key=lambda x: x[1], reverse=True
    )
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
    print(
        f"✅ Processed {len(X)} clinical records across 4 vital sign features."
    )

    # Execute hyperparameter tuning study
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=15)

    print("\n Best Optuna Trial:")
    print(f"  - Macro F1-Score: {study.best_value:.4f}")
    print("  - Hyperparameters:")
    for key, val in study.best_params.items():
        print(f"    * {key}: {val}")

    # Prepare best params dictionary
    best_params = study.best_params.copy()
    best_params.update({"random_state": settings.RANDOM_SEED, "n_jobs": -1})

    # Run Ablation & Benchmarking Study
    ablation_df = run_ablation_study(X, y, best_params)

    # Train final production model on full dataset using best params
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

    # Save model artifacts & metadata
    joblib.dump(final_model, MODEL_SAVE_PATH)
    joblib.dump(
        {
            "feature_importances": importances,
            "macro_f1": study.best_value,
            "ablation_summary": ablation_df.to_dict(orient="records"),
        },
        METADATA_SAVE_PATH,
    )

    print(f"\n Model successfully exported to {MODEL_SAVE_PATH}")
    print("================================================\n")


if __name__ == "__main__":
    run_training_pipeline()