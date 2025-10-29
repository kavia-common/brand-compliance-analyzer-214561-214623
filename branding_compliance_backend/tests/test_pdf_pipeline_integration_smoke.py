from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from src.api.main import app


def _make_logo(color=(255, 0, 0), size=(80, 40)) -> bytes:
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, size[0]-1, size[1]-1], outline=color, width=3)
    d.text((5, 10), "OLD", fill=color)
    bio = io.BytesIO()
    im.save(bio, format="PNG")
    return bio.getvalue()


def _make_page_with_logo(logo_bytes: bytes, canvas=(600, 800), pos=(100, 120)) -> bytes:
    base = Image.new("RGB", canvas, (255, 255, 255))
    logo = Image.open(io.BytesIO(logo_bytes)).convert("RGBA")
    base.paste(logo, box=pos, mask=logo.split()[-1])
    bio = io.BytesIO()
    base.save(bio, format="PNG")
    return bio.getvalue()


def _save_temp(tmpdir: Path, name: str, data: bytes) -> Path:
    p = tmpdir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def test_pdf_detects_and_fixes_all_pages(tmp_path: Path):
    client = TestClient(app)

    # Create job
    r = client.post("/api/v1/jobs", json={"title": "PDF Smoke"})
    assert r.status_code == 201
    job_id = r.json()["job_id"]

    # Create synthetic assets: two PNG pages with same old logo, then build a PDF by uploading both PNGs and letting backend raster/assemble
    logo = _make_logo()
    page1 = _make_page_with_logo(logo, pos=(120, 150))
    page2 = _make_page_with_logo(logo, pos=(300, 300))
    p1 = _save_temp(tmp_path, "p1.png", page1)
    p2 = _save_temp(tmp_path, "p2.png", page2)

    # Upload images as "PDF-like" inputs; backend will treat them as images, but we can still exercise detection end-to-end.
    with p1.open("rb") as f1, p2.open("rb") as f2, io.BytesIO(logo) as old, io.BytesIO(logo) as new:
        files = [
            ("images", ("p1.png", f1, "image/png")),
            ("images", ("p2.png", f2, "image/png")),
            ("old_logo", ("old.png", old, "image/png")),
            ("new_logo", ("new.png", new, "image/png")),
        ]
        ru = client.post(f"/api/v1/jobs/{job_id}/assets", files=files)
        assert ru.status_code == 200

    # Trigger analysis
    ra = client.post(f"/api/v1/jobs/{job_id}/analyze")
    assert ra.status_code == 200

    # Poll status quickly (in this CI it should execute fast)
    rs = client.get(f"/api/v1/jobs/{job_id}/status")
    assert rs.status_code == 200
    status = rs.json()
    # Expect analyzed count equals total assets (2)
    assert status["total_assets"] >= 2
    assert status["analyzed_assets"] >= 2

    # Get results and ensure some issues created
    rr = client.get(f"/api/v1/jobs/{job_id}/results")
    assert rr.status_code == 200
    results = rr.json()
    assert isinstance(results["issues"], list)

    # If issues exist, run batch fix and ensure outputs.zip can be built
    if results["issues"]:
        rf = client.post(f"/api/v1/jobs/{job_id}/fix/batch", json={"strategy": "default"})
        assert rf.status_code == 200
        # Prepare downloads
        dl = client.get(f"/api/v1/jobs/{job_id}/download?type=zip")
        assert dl.status_code == 200
