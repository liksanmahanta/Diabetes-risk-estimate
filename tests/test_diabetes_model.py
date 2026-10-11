import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import app as web_app
from diabetes_model import FEATURES, build_models, prepare_features


def test_zero_values_in_measurements_are_marked_missing():
    row = {feature: 1.0 for feature in FEATURES}
    row.update({"Glucose": 0, "BloodPressure": 0, "BMI": 0, "Pregnancies": 0})

    prepared = prepare_features(pd.DataFrame([row]))

    assert np.isnan(prepared.loc[0, "Glucose"])
    assert np.isnan(prepared.loc[0, "BloodPressure"])
    assert np.isnan(prepared.loc[0, "BMI"])
    assert prepared.loc[0, "Pregnancies"] == 0


def test_model_pipelines_fit_with_missing_measurements():
    rows = []
    for index in range(40):
        rows.append(
            {
                "Pregnancies": index % 5,
                "Glucose": np.nan if index == 0 else 80 + index,
                "BloodPressure": 60 + index % 20,
                "SkinThickness": np.nan,
                "Insulin": np.nan,
                "BMI": 20 + index % 15,
                "DiabetesPedigreeFunction": 0.2 + index / 100,
                "Age": 20 + index % 40,
            }
        )
    features = pd.DataFrame(rows, columns=FEATURES)
    labels = np.array([index % 2 for index in range(len(features))])
    model = build_models()["Logistic Regression"]

    model.fit(features, labels)

    assert model.predict_proba(features[:2]).shape == (2, 2)


def test_prediction_endpoint_returns_probability_and_sensitivity(monkeypatch):
    class FixedProbabilityModel:
        def predict_proba(self, features):
            return np.array([[0.35, 0.65]])

    monkeypatch.setattr(
        web_app,
        "_bundle",
        {
            "model": FixedProbabilityModel(),
            "selected_model": "Test model",
            "threshold": 0.5,
            "reference_values": {feature: 1.0 for feature in FEATURES},
        },
    )
    payload = {
        "PatientName": "Test Patient",
        "Gender": "Female",
        "Pregnancies": 2,
        "Glucose": 110,
        "BloodPressure": 72,
        "SkinThickness": "   ",
        "Insulin": "   ",
        "BMI": 28.5,
        "DiabetesPedigreeFunction": "   ",
        "Age": 42,
    }

    response = web_app.app.test_client().post("/api/predict", json=payload)

    assert response.status_code == 200
    result = response.get_json()
    assert result["probability"] == 0.65
    assert result["prediction"] == 1
    assert result["gender"] == "female"
    assert len(result["explanation"]) == len(FEATURES)
    assert result["missing_measurements"] == ["SkinThickness", "Insulin"]


def test_prediction_endpoint_rejects_boolean_measurements():
    payload = {
        "PatientName": "Test Patient",
        "Gender": "female",
        "Pregnancies": 2,
        "Glucose": 110,
        "BloodPressure": 72,
        "SkinThickness": 30,
        "Insulin": 80,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.4,
        "Age": True,
    }

    response = web_app.app.test_client().post("/api/predict", json=payload)

    assert response.status_code == 400
    assert response.get_json() == {"error": "Age must be a number."}


def test_prediction_endpoint_requires_supported_gender():
    payload = {
        "PatientName": "Test Patient",
        "Gender": "other",
    }
    response = web_app.app.test_client().post(
        "/api/predict",
        json=payload,
    )

    assert response.status_code == 400
    assert response.get_json() == {"error": "Gender must be either male or female."}