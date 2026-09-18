"""API contracts for the integrated Door, ACV and SHM models."""

from pathlib import Path

import pytest
from backend.api import create_app
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as session:
        yield session


@pytest.mark.parametrize(
    "subsystem,relative,field",
    [
        ("door", "Door/Test.csv", "cycles"),
        ("acv", "ACV/Test/acv_test_case.xlsx", "ranked_cars"),
        ("shm", "SHM/Test/test01.csv", "prediction"),
    ],
)
def test_real_model_matches_api_and_preserves_filename(
    client, subsystem, relative, field
):
    path = ROOT / "PS3/02_Datasets" / relative
    predictor = client.app.state.predictors[subsystem]
    expected = (
        [c.to_dict() for c in predictor.predict_stream(path)]
        if subsystem == "door"
        else predictor.predict_file(path).to_dict()[field]
    )
    with path.open("rb") as stream:
        response = client.post(
            f"/api/predict/{subsystem}", files={"file": (path.name, stream)}
        )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["file_id"] == path.name
    assert result["subsystem"] == subsystem
    # Tuples serialize as arrays in JSON.
    import json

    assert result[field] == json.loads(json.dumps(expected))


@pytest.mark.parametrize(
    "subsystem,filename", [("door", "bad.csv"), ("shm", "bad.csv"), ("acv", "bad.xlsx")]
)
def test_invalid_recordings_and_unavailable_models(
    client, monkeypatch, subsystem, filename
):
    response = client.post(
        f"/api/predict/{subsystem}", files={"file": (filename, b"not a recording")}
    )
    assert response.status_code == 422
    assert response.json()["error"]["message"]
    monkeypatch.setitem(client.app.state.predictors, subsystem, None)
    assert client.get("/api/health").json()["subsystems"][subsystem] is False
    assert (
        client.post(
            f"/api/predict/{subsystem}", files={"file": (filename, b"data")}
        ).status_code
        == 503
    )


def test_all_direct_routes_and_wrong_extension(client):
    for subsystem in ("rail", "door", "acv", "shm"):
        assert client.get("/api/health").json()["subsystems"][subsystem]
        if (ROOT / "app/frontend/dist/index.html").exists():
            assert client.get("/" + subsystem).status_code == 200
    assert (
        client.post(
            "/api/predict/acv", files={"file": ("bad.csv", b"data")}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/predict/unknown", files={"file": ("bad.csv", b"data")}
        ).status_code
        == 404
    )
