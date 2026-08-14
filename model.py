import os
from typing import Any, Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_PATH = "triage_rf_model.joblib"
METADATA_PATH = "model_metadata.joblib"


# Feature Engineering
# ===================
# These 4 were chosen based off the standard Emergency Severity Index (ESI) intake protocols.

feature_cols = [
    "heart_rate",
    "systolic_bp",
    "oxygen_saturation",
    "temperature_f",
]

#reads kaggle dataset as input, outputs the Random Forest Classifier and Dictionary of important features
def train_random_forest(csv_path: str = "data.csv", ) -> Tuple[RandomForestClassifier, Dict[str, float]]: 

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset file '{csv_path}' not found."
        )

    print(f"Found dataset, loading '{csv_path}")
    df = pd.read_csv(csv_path)

    target_col = "triage_score" # Lables: 0 = Routine, 1 = Urgent, 2 = Critical

    # This removes incomplete records (any record with NaN data)
    df = df.dropna(subset= feature_cols + [target_col])
    
    x = df[feature_cols]
    y = df[target_col]

    # Split to training and testing splits
    # 80/20 targeting the wanted variables using stratification to prevent bias
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)

    # Create and fit the ensemble training model for Random Forest
    # Aiming for higher accuracy and generalization by using ensemble methods
    # This will average predictions across 100 decorrelated trees (Using a set seed for testing purposes)
    rf_classifier = RandomForestClassifier(
        n_estimators=100,                 # Makes 100 distinct trees (Better for generalization and accuracy)
        max_depth=8,                      # Noise Reduction by limiting depth
        random_state=1,                   # Using a fixed random state for reproducibility (if in production this sould be removed)
        n_jobs=1,                         # Speed up training using parallelization to use all CPU cores
    )

    rf_classifier.fit(x_train, y_train)

    # Evaluate model performance metrics
    train_acc = rf_classifier.score(x_train, y_train)
    test_acc = rf_classifier.score(x_test, y_test)
    print("=== Random Forest Classifier Training Finished ===")
    print(f"  - Train Accuracy: {train_acc:.2%}")
    print(f"  - Test Accuracy:  {test_acc:.2%}")

    # Extract the important features using Gini scores
    # Useful for healthcare, as Random Forest means that it operates as a Blackbox
    importances = rf_classifier.feature_importances_
    feature_importance_dict = {
        col: float(importance) for col, importance in zip(feature_cols, importances)
    }

    # Save the model and metadata for FastAPI usage
    # Decoupled the API serve from model training to have instant startup features
    joblib.dump(rf_classifier, MODEL_PATH)
    joblib.dump(feature_importance_dict, METADATA_PATH)
    print(f"Saved Random Forest model artifact to {MODEL_PATH}")

    return rf_classifier, feature_importance_dict
