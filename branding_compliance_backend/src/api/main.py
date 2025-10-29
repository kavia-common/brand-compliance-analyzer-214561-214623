from __future__ import annotations

import io
import json
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from src.models.schemas import (
    StartRequestParams,
    StatusResponse,
)
from src.tasks.pdf_pipeline import PdfLogoReplaceManager
from src.storage.paths import ensure_storage_root, load_metadata_safely

# FastAPI app with metadata and OpenAPI tags
app = FastAPI(
    title="Brand Compliance Backend",
    description="APIs to analyze and replace old logos in PDFs. Upload a PDF with old logo templates and a new logo to produce a corrected PDF, along with previews and status.",
    version="0.1.0",
    openapi_tags=[
        {
            "name": "PDF Logo Replace",
            "description": "Upload PDFs for logo replacement, check status, view previews, and download results.",
        }
    ],
)

# Allow CORS (broadly for MVP)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, narrow this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize storage root on startup
ensure_storage_root()


# PUBLIC_INTERFACE
@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint."""
    return {"message": "Healthy"}


# PUBLIC_INTERFACE
@app.post(
    "/pdf/logo-replace/start",
    tags=["PDF Logo Replace"],
    summary="Start PDF logo replacement job",
    description="Upload a PDF, one or more old logo images, and a new logo image. Returns a job_id used to query status, previews, and download.",
    response_model=dict,
)
async def start_logo_replace_job(
    pdf: UploadFile = File(..., description="The input PDF file"),
    old_logos: list[UploadFile] = File(..., description="One or more images of the old logo to detect"),
    new_logo: UploadFile = File(..., description="The new logo image to overlay in place of detected old logos"),
    dpi: int = Form(250, description="Rasterization DPI for detection (200-300 recommended)"),
    max_pages: Optional[int] = Form(None, description="Optional cap on number of pages to process"),
    match_threshold: float = Form(0.8, description="Template matching threshold (0.0-1.0)"),
):
    # Basic validations
    if not pdf.filename or not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid or missing PDF file.")
    if not old_logos or any((not f.filename for f in old_logos)):
        raise HTTPException(status_code=400, detail="At least one old_logo image is required.")
    if not new_logo or not new_logo.filename:
        raise HTTPException(status_code=400, detail="New logo image is required.")
    if dpi < 100 or dpi > 600:
        raise HTTPException(status_code=400, detail="dpi must be between 100 and 600.")
    if match_threshold <= 0 or match_threshold > 1:
        raise HTTPException(status_code=400, detail="match_threshold must be in (0, 1].")

    # Build params model (multipart handled manually, but keep strong typing)
    try:
        params = StartRequestParams(
            dpi=dpi,
            max_pages=max_pages,
            match_threshold=match_threshold,
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=json.loads(e.json()))

    # Delegate to manager to create job and spawn processing
    job_id = PdfLogoReplaceManager.create_job_and_start(
        pdf_file=pdf,
        old_logo_files=old_logos,
        new_logo_file=new_logo,
        params=params,
    )
    return {"job_id": job_id}


# PUBLIC_INTERFACE
@app.get(
    "/pdf/logo-replace/status/{job_id}",
    tags=["PDF Logo Replace"],
    summary="Get job status",
    description="Returns status, progress, and findings summary for the specified job.",
    response_model=StatusResponse,
)
def get_status(job_id: str):
    status = PdfLogoReplaceManager.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return status


# PUBLIC_INTERFACE
@app.get(
    "/pdf/logo-replace/preview/{job_id}/{page_index}",
    tags=["PDF Logo Replace"],
    summary="Get preview image for a page",
    description="Returns a PNG preview for the given page. type=detected shows detection boxes; type=replaced shows the page with overlays.",
)
def get_preview(job_id: str, page_index: int, type: str = "detected"):
    # Validate preview type
    if type not in ("detected", "replaced"):
        raise HTTPException(status_code=400, detail="type must be 'detected' or 'replaced'.")

    preview_bytes = PdfLogoReplaceManager.get_preview(job_id, page_index, type)
    if preview_bytes is None:
        raise HTTPException(status_code=404, detail="Preview not found.")
    return StreamingResponse(io.BytesIO(preview_bytes), media_type="image/png")


# PUBLIC_INTERFACE
@app.post(
    "/pdf/logo-replace/confirm/{job_id}",
    tags=["PDF Logo Replace"],
    summary="Confirm results (stub)",
    description="Optional confirmation step. For MVP this is a stub that marks job as confirmed if possible.",
)
def confirm_job(job_id: str):
    ok = PdfLogoReplaceManager.confirm(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"status": "confirmed"}


# PUBLIC_INTERFACE
@app.get(
    "/pdf/logo-replace/download/{job_id}",
    tags=["PDF Logo Replace"],
    summary="Download replaced PDF",
    description="Downloads the corrected/replaced PDF for the given job_id.",
)
def download_replaced_pdf(job_id: str):
    data = PdfLogoReplaceManager.get_download(job_id)
    if data is None:
        # Try to give a more descriptive message if available in metadata
        md = load_metadata_safely(job_id)
        if md and md.get("status") != "completed":
            raise HTTPException(status_code=409, detail="Job is not yet completed.")
        raise HTTPException(status_code=404, detail="Replaced PDF not found.")
    filename, payload = data
    return Response(
        content=payload,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )
