from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from diabetes_model import ARTIFACT_PATH, FEATURES, ZERO_IS_MISSING, explain_prediction, prepare_features

app = Flask(__name__)
_bundle: dict[str, Any] | None = None
RANGES = {
    "Pregnancies": (0, 20),
    "Glucose": (20, 300),
    "BloodPressure": (20, 150),
    "SkinThickness": (0, 100),
    "Insulin": (0, 900),
    "BMI": (10, 80),
    "DiabetesPedigreeFunction": (0, 3),
    "Age": (18, 100),
}
OPTIONAL_FEATURES = {"SkinThickness", "Insulin", "DiabetesPedigreeFunction"}


def get_bundle() -> dict[str, Any]:
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(ARTIFACT_PATH)
    return _bundle


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/model-info")
def model_info():
    try:
        bundle = get_bundle()
    except FileNotFoundError:
        return jsonify({"error": "Model not trained yet. Run python train.py first."}), 503
    return jsonify(
        {
            "selected_model": bundle["selected_model"],
            "metrics": bundle["metrics"],
            "selection_cv_f1": bundle["selection_cv_f1"],
            "feature_importance": bundle["feature_importance"],
            "training_rows": bundle["training_rows"],
            "test_rows": bundle["test_rows"],
        }
    )


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send patient measurements as a JSON object."}), 400

    values: dict[str, float] = {}
    for feature in FEATURES:
        raw_value = payload.get(feature)
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()
        if raw_value is None or raw_value == "":
            if feature in OPTIONAL_FEATURES:
                values[feature] = np.nan
                continue
            return jsonify({"error": f"{feature} is required."}), 400
        if isinstance(raw_value, bool):
            return jsonify({"error": f"{feature} must be a number."}), 400
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return jsonify({"error": f"{feature} must be a number."}), 400
        if not np.isfinite(value):
            return jsonify({"error": f"{feature} must be a finite number."}), 400
        minimum, maximum = RANGES[feature]
        if not minimum <= value <= maximum:
            return jsonify({"error": f"{feature} must be between {minimum} and {maximum}."}), 400
        values[feature] = value

    features = prepare_features(pd.DataFrame([values]))
    try:
        bundle = get_bundle()
    except FileNotFoundError:
        return jsonify({"error": "Model not trained yet. Run python train.py first."}), 503

    probability = float(bundle["model"].predict_proba(features)[0, 1])
    prediction = int(probability >= bundle["threshold"])
    return jsonify(
        {
            "probability": round(probability, 4),
            "prediction": prediction,
            "selected_model": bundle["selected_model"],
            "explanation": explain_prediction(bundle["model"], features, bundle["reference_values"]),
            "missing_measurements": [feature for feature in ZERO_IS_MISSING if pd.isna(features.iloc[0][feature])],
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)