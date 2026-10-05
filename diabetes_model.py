"""Training and inference helpers for the diabetes risk prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
FEATURE_LABELS = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Glucose",
    "BloodPressure": "Blood pressure",
    "SkinThickness": "Skin thickness",
    "Insulin": "Insulin",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "Diabetes pedigree",
    "Age": "Age",
}
ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
ARTIFACT_PATH = Path("artifacts/diabetes_model.joblib")
DATASET_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"


def prepare_features(features: pd.DataFrame) -> pd.DataFrame:
    """Select the canonical feature order and mark known invalid zeros missing."""
    prepared = features.reindex(columns=FEATURES).copy()
    for column in FEATURES:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared[ZERO_IS_MISSING] = prepared[ZERO_IS_MISSING].replace(0, np.nan)
    return prepared


def load_pima_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Fetch and validate the 768-row Pima dataset from its public CSV mirror."""
    dataset = pd.read_csv(DATASET_URL, header=None, names=[*FEATURES, "Outcome"])
    if len(dataset) != 768 or dataset[FEATURES].shape[1] != len(FEATURES):
        raise ValueError("Dataset download did not match the expected 768-row Pima schema.")
    labels = pd.to_numeric(dataset["Outcome"], errors="coerce")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError("Dataset outcome must contain only binary 0/1 labels.")
    return prepare_features(dataset[FEATURES]), labels.astype(int).rename("Outcome")


def build_models() -> dict[str, Pipeline]:
    """Create comparable pipelines with imputation and scaling fit on training data."""
    def pipeline(classifier: Any) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )

    return {
        "Logistic Regression": pipeline(LogisticRegression(max_iter=1500, class_weight="balanced")),
        "Random Forest": pipeline(
            RandomForestClassifier(
                n_estimators=350,
                class_weight="balanced",
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        ),
        "K-Nearest Neighbors": pipeline(KNeighborsClassifier(n_neighbors=9, weights="distance")),
        "Neural Network (ANN)": pipeline(
            MLPClassifier(
                hidden_layer_sizes=(48, 24),
                alpha=0.001,
                early_stopping=True,
                max_iter=600,
                random_state=42,
            )
        ),
    }


def train_and_save(artifact_path: Path = ARTIFACT_PATH) -> dict[str, Any]:
    features, labels = load_pima_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    models = build_models()
    selection_scores = {
        name: float(np.mean(cross_val_score(model, x_train, y_train, cv=5, scoring="f1")))
        for name, model in models.items()
    }
    metrics: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics[name] = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        }

    selected_name = max(selection_scores, key=selection_scores.get)
    selected_model = models[selected_name]
    importance = permutation_importance(
        selected_model,
        x_test,
        y_test,
        scoring="f1",
        n_repeats=12,
        random_state=42,
        n_jobs=-1,
    )
    global_importance = [
        {
            "feature": feature,
            "label": FEATURE_LABELS[feature],
            "importance": round(max(0.0, float(value)), 5),
        }
        for feature, value in zip(FEATURES, importance.importances_mean)
    ]
    global_importance.sort(key=lambda item: item["importance"], reverse=True)

    bundle = {
        "model": selected_model,
        "selected_model": selected_name,
        "selection_cv_f1": {name: round(score, 4) for name, score in selection_scores.items()},
        "metrics": metrics,
        "feature_importance": global_importance,
        "reference_values": x_train.median().to_dict(),
        "feature_names": FEATURES,
        "training_rows": int(len(features)),
        "test_rows": int(len(x_test)),
        "threshold": 0.5,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)
    return bundle


def explain_prediction(
    model: Any,
    features: pd.DataFrame,
    reference_values: dict[str, float],
) -> list[dict[str, Any]]:
    """Estimate local sensitivity by replacing one input with its training median."""
    baseline_probability = float(model.predict_proba(features)[0, 1])
    explanations = []
    for feature in FEATURES:
        comparison = features.copy()
        comparison.loc[:, feature] = reference_values[feature]
        comparison_probability = float(model.predict_proba(comparison)[0, 1])
        explanations.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS[feature],
                "delta_points": round((baseline_probability - comparison_probability) * 100, 1),
            }
        )
    return sorted(explanations, key=lambda item: abs(item["delta_points"]), reverse=True)