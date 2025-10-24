from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_root_create_job_compat_route_returns_job_id():
    r = client.post("/jobs", json={"title": "Compat"})
    assert r.status_code in (200, 201), r.text
    body = r.json()
    # When FastAPI serializes HTTPException details, it's under "detail";
    # success path should include job_id
    assert "job_id" in body
    assert isinstance(body["job_id"], str)
