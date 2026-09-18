"""API boundaries, upload cleanup and model contract."""

from pathlib import Path

import pytest
from backend.api import create_app
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def client():
    with TestClient(create_app()) as session:
        yield session


def test_health_and_frontend_routes(client):
    assert client.get("/api/health").json()["status"] == "ready"
    assert client.get("/api/not-a-route").status_code == 404
    # Built assets are optional when running just the Python development tests.
    if (ROOT / "app/frontend/dist/index.html").exists():
        assert client.get("/rail").status_code == 200
        assert '<div id="root">' in client.get("/").text


@pytest.mark.parametrize(
    "name,data",
    [
        ("empty.csv", b""),
        ("wrong.txt", b"1,2"),
        ("../bad.csv", b"1"),
        ("bad.csv", b"a,b\n1,2\n"),
    ],
)
def test_invalid_inputs_return_structured_errors(client, name, data):
    response = client.post(
        "/api/predict/rail", files={"file": (name, data, "text/csv")}
    )
    assert response.status_code == 422
    assert response.json()["error"]["message"]


def test_upload_size_is_enforced():
    with TestClient(create_app(maximum_bytes=8)) as client:
        response = client.post(
            "/api/predict/rail", files={"file": ("large.csv", b"x" * 9)}
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "file_too_large"
        response = client.post(
            "/api/predict/rail",
            content=b"x",
            headers={"content-length": str(2 * 1024 * 1024)},
        )
        assert response.status_code == 413


def test_missing_model_is_not_a_normal_prediction(tmp_path):
    with TestClient(create_app(model_path=tmp_path / "missing.joblib")) as client:
        assert client.get("/api/health").json()["status"] == "unavailable"
        response = client.post("/api/predict/rail", files={"file": ("one.csv", b"1")})
        assert response.status_code == 503
        assert "prediction" not in response.json()


def test_original_name_and_temporary_file_cleanup(client, monkeypatch):
    from backend.models.rail.types import RailPrediction

    seen = []

    def predict(path):
        assert path.is_file()
        assert path.name == "recording.csv"
        seen.append(path)
        return RailPrediction(
            path.name, "Side I", {"Normal": 0.1, "Side I": 0.8, "Side II": 0.1}, {}, {}
        )

    monkeypatch.setattr(client.app.state.predictor, "predict_file", predict)
    response = client.post(
        "/api/predict/rail",
        files={"file": ("Original Name.csv", b"uploaded recording")},
    )
    assert response.json()["file_id"] == "Original Name.csv"
    assert response.json()["prediction"] == "Side I"
    assert not seen[0].exists()

    def broken(path):
        seen.append(path)
        raise RuntimeError("internal failure")

    monkeypatch.setattr(client.app.state.predictor, "predict_file", broken)
    response = client.post("/api/predict/rail", files={"file": ("two.csv", b"content")})
    assert response.status_code == 500
    assert not seen[-1].exists()
    assert "internal failure" not in response.text
