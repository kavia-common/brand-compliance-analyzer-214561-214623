from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_create_job_and_status_datetime_is_iso_string():
    # Create job
    r = client.post("/api/v1/jobs", json={"title": "Test"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert "job_id" in body
    job_id = body["job_id"]
    assert isinstance(job_id, str)

    # Get status
    rs = client.get(f"/api/v1/jobs/{job_id}/status")
    assert rs.status_code == 200, rs.text
    status = rs.json()
    assert status["job_id"] == job_id
    # updated_at should be an ISO-8601 string, not a dict or number
    assert isinstance(status["updated_at"], str)
    # quick ISO shape check: contains 'T'
    assert "T" in status["updated_at"]
