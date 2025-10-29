#!/usr/bin/env python3
from __future__ import annotations

"""
PUBLIC_INTERFACE
Manual verification utility for the PDF pipeline.

This script:
- Creates a job via the FastAPI app
- Uploads a synthetic PDF made from two pages containing an 'OLD' logo and uploads old/new logos
- Triggers analysis and then batch fix
- Prints out status and locations of outputs for manual inspection

Usage:
    python -m src.tools.manual_verify_pdf_pipeline
"""

import io
import time

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


def run() -> None:
    client = TestClient(app)

    r = client.post("/api/v1/jobs", json={"title": "Manual Verify PDF"})
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print("Created job:", job_id)

    logo = _make_logo()
    page1 = _make_page_with_logo(logo, pos=(120, 150))
    page2 = _make_page_with_logo(logo, pos=(300, 300))

    with io.BytesIO(page1) as f1, io.BytesIO(page2) as f2, io.BytesIO(logo) as old, io.BytesIO(logo) as new:
        files = [
            ("images", ("p1.png", f1, "image/png")),
            ("images", ("p2.png", f2, "image/png")),
            ("old_logo", ("old.png", old, "image/png")),
            ("new_logo", ("new.png", new, "image/png")),
        ]
        ru = client.post(f"/api/v1/jobs/{job_id}/assets", files=files)
        ru.raise_for_status()
        print("Uploaded images and logos.")

    ra = client.post(f"/api/v1/jobs/{job_id}/analyze")
    ra.raise_for_status()
    print("Analysis queued.")

    # Wait a short time and poll status
    time.sleep(0.2)
    rs = client.get(f"/api/v1/jobs/{job_id}/status")
    rs.raise_for_status()
    status = rs.json()
    print("Status:", status)

    rr = client.get(f"/api/v1/jobs/{job_id}/results")
    rr.raise_for_status()
    results = rr.json()
    print("Issues found:", len(results.get("issues", [])))

    if results.get("issues"):
        rf = client.post(f"/api/v1/jobs/{job_id}/fix/batch", json={"strategy": "default"})
        rf.raise_for_status()
        print("Batch fixed assets.")
        dl = client.get(f"/api/v1/jobs/{job_id}/download?type=zip")
        print("Download status code:", dl.status_code)
    else:
        print("No issues detected; manual check may be required.")


if __name__ == "__main__":
    run()
