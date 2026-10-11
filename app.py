from __future__ import annotations

import json
import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from diabetes_model import ARTIFACT_PATH, FEATURES, ZERO_IS_MISSING, explain_prediction, prepare_features

DATABASE_PATH = Path(__file__).resolve().parent / "patient_records.db"
OWNER_AUTH_PATH = Path(__file__).resolve().parent / ".patient_owner.json"
PASSWORD_HASH_ROUNDS = 600_000

app = Flask(__name__)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax", PERMANENT_SESSION_LIFETIME=28800)
_bundle: dict[str, Any] | None = None


def read_owner_config() -> dict[str, str] | None:
    try:
        config = json.loads(OWNER_AUTH_PATH.read_text(encoding="utf-8"))
        if all(key in config for key in ("salt", "password_hash", "secret_key")):
            return config
    except (OSError, ValueError, TypeError):
        pass
    return None


def hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_HASH_ROUNDS)


def save_owner_config(password: str) -> None:
    salt = secrets.token_bytes(16)
    config = {
        "salt": salt.hex(),
        "password_hash": hash_password(password, salt).hex(),
        "secret_key": secrets.token_hex(32),
    }
    OWNER_AUTH_PATH.write_text(json.dumps(config), encoding="utf-8")
    try:
        os.chmod(OWNER_AUTH_PATH, 0o600)
    except OSError:
        pass
    app.secret_key = config["secret_key"]


_owner_config = read_owner_config()
app.secret_key = _owner_config["secret_key"] if _owner_config else secrets.token_hex(32)


@app.before_request
def require_owner():
    public_endpoints = {"owner_access", "owner_logout", "static", "index", "model_info", "predict"}
    if request.endpoint in public_endpoints:
        return None
    if request.endpoint != "patient_records":
        return None
    if read_owner_config() is None:
        return jsonify({"error": "Owner password setup is required to view patient records.", "login_url": url_for("owner_access")}), 401
    if not session.get("owner_authenticated"):
        return jsonify({"error": "Owner sign-in is required to view patient records.", "login_url": url_for("owner_access")}), 401
    return None


@app.route("/owner", methods=["GET", "POST"])
def owner_access():
    configured = read_owner_config()
    error = ""
    setup_mode = configured is None
    if request.method == "POST":
        password = request.form.get("password", "")
        if setup_mode:
            confirmation = request.form.get("confirm_password", "")
            if len(password) < 12:
                error = "Choose a password with at least 12 characters."
            elif password != confirmation:
                error = "The passwords do not match."
            else:
                try:
                    save_owner_config(password)
                except OSError:
                    app.logger.exception("Could not save owner authentication settings")
                    error = "Could not save owner settings. Check project folder permissions."
                else:
                    session["owner_authenticated"] = True
                    session.permanent = True
                    return redirect(url_for("index") + "#patient-records")
        else:
            salt = bytes.fromhex(configured["salt"])
            candidate = hash_password(password, salt).hex()
            if hmac.compare_digest(candidate, configured["password_hash"]):
                session["owner_authenticated"] = True
                session.permanent = True
                return redirect(url_for("index") + "#patient-records")
            error = "Incorrect owner password."
    return render_template("owner.html", setup_mode=setup_mode, error=error)


@app.post("/owner/logout")
def owner_logout():
    session.clear()
    return redirect(url_for("index"))


def init_db() -> None:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS patient_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            gender TEXT NOT NULL,
            created_at TEXT NOT NULL,
            glucose REAL,
            blood_pressure REAL,
            skin_thickness REAL,
            insulin REAL,
            bmi REAL,
            diabetes_pedigree_function REAL,
            age INTEGER,
            pregnancies INTEGER,
            probability REAL,
            prediction TEXT,
            explanation TEXT
        )
        """
    )
    conn.commit()
    conn.close()


init_db()
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


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid numeric value.")
    if value is np.nan or pd.isna(value):
        return None
    return float(value)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid numeric value.")
    if value is np.nan or pd.isna(value):
        return None
    return int(float(value))


def save_patient_record(payload: dict[str, Any], probability: float, prediction: str, explanation: str) -> dict[str, Any]:
    """Persist a patient prediction record to SQLite."""
    record = {
        "patient_name": (payload.get("PatientName") or "").strip() or "Unknown patient",
        "gender": (payload.get("Gender") or "").strip() or "unknown",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "glucose": _coerce_float(payload.get("Glucose")),
        "blood_pressure": _coerce_float(payload.get("BloodPressure")),
        "skin_thickness": _coerce_float(payload.get("SkinThickness")),
        "insulin": _coerce_float(payload.get("Insulin")),
        "bmi": _coerce_float(payload.get("BMI")),
        "diabetes_pedigree_function": _coerce_float(payload.get("DiabetesPedigreeFunction")),
        "age": _coerce_int(payload.get("Age")),
        "pregnancies": _coerce_int(payload.get("Pregnancies")),
        "probability": float(probability),
        "prediction": prediction,
        "explanation": explanation,
    }

    conn = sqlite3.connect(DATABASE_PATH)
    try:
        cursor = conn.execute(
            """
            INSERT INTO patient_records (
                patient_name, gender, created_at, glucose, blood_pressure, skin_thickness,
                insulin, bmi, diabetes_pedigree_function, age, pregnancies, probability,
                prediction, explanation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["patient_name"],
                record["gender"],
                record["created_at"],
                record["glucose"],
                record["blood_pressure"],
                record["skin_thickness"],
                record["insulin"],
                record["bmi"],
                record["diabetes_pedigree_function"],
                record["age"],
                record["pregnancies"],
                record["probability"],
                record["prediction"],
                record["explanation"],
            ),
        )
        conn.commit()
        record["id"] = cursor.lastrowid
    finally:
        conn.close()

    return record


@app.get("/api/patient-records")
def patient_records():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, patient_name, gender, created_at, glucose, blood_pressure, skin_thickness,
               insulin, bmi, diabetes_pedigree_function, age, pregnancies, probability,
               prediction, explanation
        FROM patient_records
        ORDER BY id DESC
        """
    ).fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])


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

    raw_patient_name = payload.get("PatientName")
    patient_name = raw_patient_name.strip() if isinstance(raw_patient_name, str) else ""
    if not patient_name:
        return jsonify({"error": "Patient name is required."}), 400

    raw_gender = payload.get("Gender")
    gender = raw_gender.strip().lower() if isinstance(raw_gender, str) else ""
    if gender not in {"male", "female"}:
        return jsonify({"error": "Gender must be either male or female."}), 400

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
    explanation = explain_prediction(bundle["model"], features, bundle["reference_values"])

    try:
        saved_record = save_patient_record(payload, probability, prediction, json.dumps(explanation))
    except sqlite3.Error:
        app.logger.exception("Could not save patient prediction record")
        return jsonify({"error": "The prediction was calculated, but the patient record could not be saved. Check the database file permissions and try again."}), 500

    response = {
        "patient_name": patient_name,
        "probability": round(probability, 4),
        "prediction": prediction,
        "gender": gender,
        "selected_model": bundle["selected_model"],
        "explanation": explanation,
        "missing_measurements": [feature for feature in ZERO_IS_MISSING if pd.isna(features.iloc[0][feature])],
    }
    response["record_id"] = saved_record["id"]

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
