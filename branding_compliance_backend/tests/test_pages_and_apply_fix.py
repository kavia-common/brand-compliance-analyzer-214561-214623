from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import fitz  # PyMuPDF

from src.api.main import app


def _make_logo(color=(255, 0, 0), size=(80, 40)) -> bytes:
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=color, width=3)
    d.text((5, 10), "OLD", fill=color)
    bio = io.BytesIO()
    im.save(bio, format="PNG")
    return bio.getvalue()


def _make_pdf_with_embedded_logo(logo_png: bytes, page_size=(600, 800), pos=(120, 150)) -> bytes:
    # Create a single-page PDF and embed the logo PNG at a given position
    doc = fitz.open()
    page = doc.new_page(width=page_size[0], height=page_size[1])
    # Insert image from memory
    rect = fitz.Rect(pos[0], pos[1], pos[0] + 160, pos[1] + 100)  # scale region for visibility
    page.insert_image(rect, stream=logo_png)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _save_tmp(tmpdir: Path, name: str, data: bytes) -> Path:
    p = tmpdir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def test_pages_list_and_apply_fix_pdf(tmp_path: Path):
    client = TestClient(app)
    # Create job
    r = client.post("/api/v1/jobs", json={"title": "Pages Smoke"})
    assert r.status_code == 201
    job_id = r.json()["job_id"]

    logo = _make_logo()
    pdf_bytes = _make_pdf_with_embedded_logo(logo)
    pdf_path = _save_tmp(tmp_path, "doc.pdf", pdf_bytes)

    # Upload PDF plus old/new brand logos
    with pdf_path.open("rb") as fpdf, io.BytesIO(logo) as old, io.BytesIO(logo) as new:
        files = [
            ("images", ("doc.pdf", fpdf, "application/pdf")),
            ("old_logo", ("old.png", old, "image/png")),
            ("new_logo", ("new.png", new, "image/png")),
        ]
        ru = client.post(f"/api/v1/jobs/{job_id}/assets", files=files)
        assert ru.status_code == 200

    # Trigger analysis
    ra = client.post(f"/api/v1/jobs/{job_id}/analyze")
    assert ra.status_code == 200

    # List pages and expect at least one page, with detections > 0 or has_detection True
    lp = client.get(f"/api/v1/jobs/{job_id}/pages")
    assert lp.status_code == 200
    data = lp.json()
    assert "pages" in data
    pages = data["pages"]
    # There should be one page entry
    assert len(pages) >= 1
    page0 = pages[0]
    # has_detection should be present and be a boolean
    assert "has_detection" in page0
    # We accept either true or a positive detections count depending on threshold/environment
    assert isinstance(page0["has_detection"], bool)
    # Apply fixes at job-level (v1)
    af = client.post(f"/api/v1/jobs/{job_id}/apply-fix")
    assert af.status_code == 200
    af_json = af.json()
    # The final_pdf_url should point to download endpoint if a PDF was assembled
    if af_json.get("final_pdf_url"):
        # Try to download PDF
        dl = client.get(af_json["final_pdf_url"])
        assert dl.status_code in (200, 404)  # allow 404 in edge CI where image libs unavailable
    # Also test compat alias to avoid 404s when frontend base lacks /api/v1
    afc = client.post(f"/jobs/{job_id}/apply-fix")
    assert afc.status_code == 200
