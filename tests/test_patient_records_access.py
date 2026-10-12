import app as web_app


def test_patient_records_are_hidden_from_anonymous_visitors(monkeypatch, tmp_path):
    monkeypatch.setattr(web_app, "DATABASE_PATH", tmp_path / "patient_records.db")
    monkeypatch.setattr(web_app, "read_owner_config", lambda: {"configured": "true"})
    web_app.init_db()
    web_app.save_patient_record(
        {"PatientName": "Private Patient", "Gender": "female"},
        0.65,
        "1",
        "[]",
    )

    client = web_app.app.test_client()

    response = client.get("/api/patient-records")
    assert response.status_code == 401
    assert response.get_json()["login_url"] == "/owner"

    page = client.get("/")
    assert b"Patient records are private." in page.data
    assert b"Private Patient" not in page.data
    assert b'id="patient-records-body"' not in page.data


def test_patient_records_are_visible_to_authenticated_owner(monkeypatch, tmp_path):
    monkeypatch.setattr(web_app, "DATABASE_PATH", tmp_path / "patient_records.db")
    monkeypatch.setattr(web_app, "read_owner_config", lambda: {"configured": "true"})
    web_app.init_db()
    web_app.save_patient_record(
        {"PatientName": "Private Patient", "Gender": "female"},
        0.65,
        "1",
        "[]",
    )

    client = web_app.app.test_client()
    with client.session_transaction() as owner_session:
        owner_session["owner_authenticated"] = True

    response = client.get("/api/patient-records")
    assert response.status_code == 200
    assert response.get_json()[0]["patient_name"] == "Private Patient"

    page = client.get("/")
    assert b'id="patient-records-body"' in page.data
    assert b"Sign out" in page.data
