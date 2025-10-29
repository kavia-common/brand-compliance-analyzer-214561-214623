from __future__ import annotations

import re
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile, Request
from typing import List
from fastapi.responses import FileResponse, JSONResponse, Response
import mimetypes
import logging
import os
import json
import io
from pydantic import BaseModel, Field, StrictStr

from src.api.errors import error_response
from src.models.job import Job, JobStatus
from src.services.analyzer import AnalyzerService
from src.services.fixer import FixerService
from src.services.report import ReportService
from src.services.state_store import StateStore
from src.storage.workspace import get_job_workspace


def get_state_store() -> StateStore:
    return StateStore()


router = APIRouter(
    prefix="/api/v1",
    tags=["jobs", "assets", "report"],
)

_UUID_RE = re.compile(r"^[0-9a-fA-F-]{32,36}$")


def _validate_job_id(job_id: str) -> None:
    """Lightweight validation for job_id format to catch obvious errors early."""
    if not job_id or not isinstance(job_id, str):
        raise HTTPException(
            status_code=400,
            detail=error_response(
                "invalid_job_id",
                "job_id must be a non-empty string",
                {"job_id": job_id},
                status_code=400,
            ),
        )
    if not _UUID_RE.match(job_id):
        # We accept hyphenated UUIDs; avoid over-strict parsing, just ensure safe charset/length
        raise HTTPException(
            status_code=400,
            detail=error_response(
                "invalid_job_id_format",
                "job_id format appears invalid",
                {"job_id": job_id},
                status_code=400,
            ),
        )


def _build_public_url_for_output(request: Request, output_path: Path) -> str | None:
    """
    Given an absolute Path to a file under the workspace jobs/<job_id>/outputs,
    construct a browser-accessible URL served by the /outputs static mount.

    It will attempt to:
      1) Use BASE_URL env var if provided (e.g., https://host:3001)
      2) Otherwise, derive from request.base_url

    Returns None if the path is not under the expected outputs directory layout.
    """
    try:
        # Expect structure: .../jobs/<job_id>/outputs/<file>
        parts = output_path.parts
        if "jobs" not in parts:
            return None
        idx = parts.index("jobs")
        # require at least jobs/<job_id>/outputs/<file>
        if len(parts) < idx + 4:
            return None
        job_id = parts[idx + 1]
        if parts[idx + 2] != "outputs":
            return None
        rel_in_job = Path(*parts[idx + 2:])  # outputs/<file...>
        # Our StaticFiles mount serves jobs root at /outputs, so URL becomes:
        # /outputs/{job_id}/{rel_in_job}
        path_suffix = f"/outputs/{job_id}/{rel_in_job.as_posix()}"
        base = os.getenv("BASE_URL")
        if base:
            return f"{base.rstrip('/')}{path_suffix}"
        # fallback to request base_url
        return f"{str(request.base_url).rstrip('/')}{path_suffix}"
    except Exception:
        return None


def _png_placeholder(message: str, width: int = 800, height: int = 450) -> Response:
    """
    Generate a simple PNG with the given message for cases where a preview image is not available.

    Returns:
        Starlette Response with image/png content.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGBA", (width, height), (245, 245, 245, 255))
        draw = ImageDraw.Draw(img)
        title = "Preview not available"
        # Attempt to use a default font; fallback to basic
        try:
            font_title = ImageFont.load_default()
            font_msg = ImageFont.load_default()
        except Exception:
            font_title = None
            font_msg = None

        # Center the text roughly
        tw, th = draw.textsize(title, font=font_title)
        draw.text(((width - tw) / 2, height * 0.35), title, fill=(80, 80, 80, 255), font=font_title)
        # Multi-line message
        lines = [message]
        y = height * 0.35 + th + 16
        for line in lines:
            lw, lh = draw.textsize(line, font=font_msg)
            draw.text(((width - lw) / 2, y), line, fill=(100, 100, 100, 255), font=font_msg)
            y += lh + 4

        bio = io.BytesIO()
        img.convert("RGB").save(bio, format="PNG")
        bio.seek(0)
        headers = {"X-Placeholder": "1"}
        return Response(content=bio.getvalue(), media_type="image/png", headers=headers)
    except Exception:
        # Fallback: return a minimal 1x1 PNG if Pillow is unavailable
        minimal_png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c\xf8\x0f"
            b"\x00\x01\x01\x01\x00\x18\xdd\x8d\xf7\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return Response(content=minimal_png, media_type="image/png", headers={"X-Placeholder": "1"})


# Request/Response Models

# PUBLIC_INTERFACE
class CreateJobRequest(BaseModel):
    """Request to create a job.

    All fields are optional and default to None.
    """
    owner: Optional[StrictStr] = Field(None, description="Owner/user id")
    title: Optional[StrictStr] = Field(None, description="Optional job title")
    description: Optional[StrictStr] = Field(None, description="Job description")


# PUBLIC_INTERFACE
class CreateJobResponse(BaseModel):
    """Response containing created job ID."""
    job_id: StrictStr = Field(..., description="Created job id")


# PUBLIC_INTERFACE
class StatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    total_assets: int
    analyzed_assets: int
    failed_assets: int
    progress_percent: float
    issues_total: int
    issues_high_or_above: int
    updated_at: str


class FixAssetRequest(BaseModel):
    """Request to fix a single asset."""
    strategy: Optional[str] = Field(None, description="Fix strategy (stubbed)")


class BatchFixRequest(BaseModel):
    """Request to run batch fixes."""
    strategy: Optional[str] = Field(None, description="Batch fix strategy (stubbed)")


# PUBLIC_INTERFACE
class PageEntry(BaseModel):
    """Represents one PDF page preview entry."""
    index: int = Field(..., description="0-based page index")
    asset_id: Optional[str] = Field(None, description="Owning asset id if known (PDF)")
    status: str = Field(..., description="Page status: detected|fixed|skipped|pending")
    has_detection: bool = Field(..., description="True if one or more detections above threshold were found on this page")
    original_url: str = Field(..., description="Preview URL for original rasterized page")
    overlay_url: str = Field(..., description="Preview URL with detection overlay if available")
    fixed_url: str = Field(..., description="Preview URL after applying fix if available")
    detections: int = Field(..., description="Number of detections found on this page above threshold")


# Routes

# PUBLIC_INTERFACE
@router.post(
    "/jobs",
    summary="Create Job",
    description="Create a new job and initialize workspace JSON files.",
    response_model=CreateJobResponse,
    tags=["jobs"],
    status_code=201,
)
def create_job(data: CreateJobRequest, state: StateStore = Depends(get_state_store)):
    """Create a job with optional title and metadata.

    Returns:
        CreateJobResponse with job_id.
    """
    logger = logging.getLogger("api.create_job")
    # (a) job_id generation and validation
    try:
        job_id = str(uuid.uuid4())
    except Exception as e:
        logger.exception("Failed to generate job_id: %s", e)
        raise HTTPException(status_code=500, detail="Failed to generate job id")

    # Build Job model and validate input
    try:
        job = Job(
            id=job_id,
            status=JobStatus.created,
            owner=data.owner,
            title=data.title,
            description=data.description,
        )
    except Exception as e:
        # Pydantic validation or enum error
        logger.exception("Invalid job payload: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

    # (a) workspace root resolution with verbose logs
    try:
        from src.storage.workspace import get_root_workspace, get_job_workspace
        ws_root = get_root_workspace()
        job_ws = get_job_workspace(job_id)
        logger.info(
            "Resolved workspace: root=%s job_dir=%s",
            ws_root, job_ws["job"]
        )
        # temp disk write check in job dir to ensure permissions
        probe = job_ws["job"] / ".create_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except HTTPException:
        # rethrow http exceptions if any
        raise
    except ValueError as ve:
        # input-related path issues should be 400 if job_id invalid chars (unlikely)
        logger.exception("Invalid path parameters for job: %s", ve)
        raise HTTPException(status_code=400, detail=str(ve))
    except PermissionError as pe:
        logger.exception("Permission error preparing workspace: %s", pe)
        raise HTTPException(status_code=409, detail="Workspace not writable")
    except Exception as e:
        logger.exception("Error resolving workspace: %s", e)
        # keep going; state.create_job will likely fail and we log there too

    # (c) atomic JSON writes via state store
    try:
        state.create_job(job)
    except ValueError as ve:
        # Duplicate or state-related validation
        logger.warning("Job creation failed (bad request): %s", ve)
        raise HTTPException(status_code=400, detail=str(ve))
    except PermissionError as pe:
        logger.exception("Permission error writing job JSON: %s", pe)
        raise HTTPException(status_code=409, detail="Workspace not writable")
    except Exception as e:
        logger.exception("Unexpected error creating job (IO/JSON): %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    # (d) response model serialization with guard
    try:
        resp = CreateJobResponse(job_id=job_id)
    except Exception as e:
        logger.exception("Failed to serialize CreateJobResponse: %s", e)
        raise HTTPException(status_code=500, detail="Failed to serialize response")

    logger.info("Job created: %s", job_id)
    return resp


# PUBLIC_INTERFACE
@router.delete(
    "/jobs/{job_id}",
    summary="Delete Job",
    description="Delete a job and its workspace folder.",
    tags=["jobs"],
)
def delete_job(job_id: str, state: StateStore = Depends(get_state_store)):
    """Delete the job workspace directory and JSON state."""
    w = get_job_workspace(job_id)
    if not w["job"].exists():
        raise HTTPException(status_code=404, detail="Job not found")
    # Best-effort removal
    shutil.rmtree(w["job"], ignore_errors=True)
    return {"message": "Deleted", "job_id": job_id}


def _save_upload_to(path: Path, file: UploadFile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: write to tmp in same dir then replace
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    tmp_path.replace(path)


# PUBLIC_INTERFACE
@router.post(
    "/jobs/{job_id}/upload/assets",
    summary="Upload Assets Zip",
    description="Upload a zip file of assets. Contents are extracted to uploads/.",
    tags=["assets"],
)
async def upload_assets_zip(job_id: str, file: UploadFile = File(...), state: StateStore = Depends(get_state_store)):
    """Accepts a ZIP and extracts into uploads/ for the job."""
    if state.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")

    w = get_job_workspace(job_id)
    uploads = w["uploads"]
    uploads.mkdir(parents=True, exist_ok=True)
    tmp_zip = uploads / f"upload_{uuid.uuid4().hex}.zip"
    _save_upload_to(tmp_zip, file)

    try:
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(uploads)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip file")
    finally:
        tmp_zip.unlink(missing_ok=True)

    state.update_job_status(job_id, JobStatus.uploading)
    return {"message": "Assets uploaded", "job_id": job_id}

# PUBLIC_INTERFACE
@router.post(
    "/jobs/{job_id}/assets",
    summary="Upload Assets (multipart)",
    description=(
        "Accepts multipart form-data for images/documents and brand logos.\n"
        "Fields:\n"
        "- images[]: one or more files to analyze, saved under uploads/\n"
        "- old_logo: optional file saved under analysis/old_brand/\n"
        "- new_logo: optional file saved under analysis/new_brand/\n"
        "Responds with counts and saved relative paths."
    ),
    tags=["assets"],
)
async def upload_assets_multipart(
    job_id: str,
    request: Request,
    images: List[UploadFile] = File(default_factory=list, description="One or more input assets (images/docs)"),
    old_logo: UploadFile | None = File(default=None, description="Old brand/logo image"),
    new_logo: UploadFile | None = File(default=None, description="New brand/logo image"),
    state: StateStore = Depends(get_state_store),
):
    """
    Handle multipart uploads for assets and optional logo references.

    Saves:
      - images[] to jobs/{job_id}/uploads/
      - old_logo to jobs/{job_id}/analysis/old_brand/
      - new_logo to jobs/{job_id}/analysis/new_brand/
    Returns JSON with saved paths and counts. Errors include structured JSON messages.
    """
    # Validate job exists
    if state.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail=error_response("job_not_found", "Job not found", {"job_id": job_id}, 404))

    w = get_job_workspace(job_id)
    saved_images: list[str] = []
    saved_old: str | None = None
    saved_new: str | None = None

    # Save images[]
    if images:
        for f in images:
            try:
                target = w["uploads"] / f.filename
                _save_upload_to(target, f)
                saved_images.append(str(target.relative_to(w["job"])))
            except Exception as e:
                # Continue saving others but note failure
                logging.getLogger("api.upload").exception("Failed to save image %s: %s", f.filename, e)

    # Save old_logo
    if old_logo is not None:
        try:
            target_old = w["analysis"] / "old_brand" / old_logo.filename
            _save_upload_to(target_old, old_logo)
            saved_old = str(target_old.relative_to(w["job"]))
        except Exception as e:
            raise HTTPException(status_code=500, detail=error_response("save_failed", "Failed to save old_logo", {"error": str(e)}, 500))

    # Save new_logo
    if new_logo is not None:
        try:
            target_new = w["analysis"] / "new_brand" / new_logo.filename
            _save_upload_to(target_new, new_logo)
            saved_new = str(target_new.relative_to(w["job"]))
        except Exception as e:
            raise HTTPException(status_code=500, detail=error_response("save_failed", "Failed to save new_logo", {"error": str(e)}, 500))

    # Update status if any images were uploaded
    if saved_images:
        try:
            state.update_job_status(job_id, JobStatus.uploading)
        except Exception:
            pass

    return {
        "message": "Upload complete",
        "job_id": job_id,
        "saved": {
            "images": saved_images,
            "old_logo": saved_old,
            "new_logo": saved_new,
        },
        "counts": {
            "images": len(saved_images),
            "logos": int(saved_old is not None) + int(saved_new is not None),
        },
    }


# PUBLIC_INTERFACE
@router.post(
    "/jobs/{job_id}/upload/old-brand",
    summary="Upload Old Brand Image",
    description="Upload an image of the old brand to aid detection.",
    tags=["assets"],
)
async def upload_old_brand(job_id: str, file: UploadFile = File(...), state: StateStore = Depends(get_state_store)):
    """Upload a single old brand reference image into analysis/old_brand/."""
    if state.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")

    w = get_job_workspace(job_id)
    target = w["analysis"] / "old_brand" / file.filename
    _save_upload_to(target, file)
    return {"message": "Old brand uploaded", "path": str(target.relative_to(w["job"]))}


# PUBLIC_INTERFACE
@router.post(
    "/jobs/{job_id}/upload/new-brand",
    summary="Upload New Brand Image",
    description="Upload an image of the new brand for fixing overlays.",
    tags=["assets"],
)
async def upload_new_brand(job_id: str, file: UploadFile = File(...), state: StateStore = Depends(get_state_store)):
    """Upload a single new brand reference image into analysis/new_brand/."""
    if state.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    w = get_job_workspace(job_id)
    target = w["analysis"] / "new_brand" / file.filename
    _save_upload_to(target, file)
    return {"message": "New brand uploaded", "path": str(target.relative_to(w["job"]))}


# PUBLIC_INTERFACE
@router.post(
    "/jobs/{job_id}/analyze",
    summary="Trigger Analysis",
    description="Run logo detection pipeline. Persists detections to work/detections.json and generates overlays. Progress available via status.",
    tags=["jobs"],
)
def analyze_job(job_id: str, background: BackgroundTasks, request: Request, state: StateStore = Depends(get_state_store)):
    """Trigger background analysis and preview generation.

    Validates job_id and job state, ensures required inputs exist, and schedules background work.
    Returns:
        200 on queued with a structured payload,
        4xx with structured error JSON for invalid job/state/inputs,
        5xx only for unexpected errors (structured).
    """
    log = logging.getLogger("api.analyze")
    origin = request.headers.get("origin")
    phase = "validate"
    try:
        # Structured entry log
        log.info("analyze:start job_id=%s phase=%s origin=%s", job_id, phase, origin)

        # Validate job_id format early (unit-friendly and avoids path traversal)
        _validate_job_id(job_id)

        # Load job
        phase = "load_job"
        job = state.get_job(job_id)
        if job is None:
            log.warning("analyze:missing_job job_id=%s phase=%s", job_id, phase)
            return JSONResponse(
                status_code=404,
                content=error_response("job_not_found", "Job not found", {"job_id": job_id}, status_code=404),
            )

        # Check inputs existence to avoid heavy processing when uploads are missing
        phase = "check_inputs"
        w = get_job_workspace(job_id)
        uploads = w["uploads"]
        if not uploads.exists():
            log.info("analyze:no_uploads job_id=%s phase=%s", job_id, phase)
            # Early return: unit-friendly lightweight check
            return JSONResponse(
                status_code=400,
                content=error_response(
                    "missing_inputs",
                    "No assets uploaded yet. Please upload a zip before analyzing.",
                    {"job_id": job_id},
                    status_code=400,
                ),
            )
        # Guard against empty uploads folder
        has_files = any(p.is_file() for p in uploads.rglob("*"))
        if not has_files:
            log.info("analyze:empty_uploads job_id=%s phase=%s", job_id, phase)
            return JSONResponse(
                status_code=400,
                content=error_response(
                    "missing_inputs",
                    "Uploads folder is empty. Ensure your zip contained files.",
                    {"job_id": job_id},
                    status_code=400,
                ),
            )

        # Ensure job state is appropriate; allow re-queue from created/uploading/failed/cancelled
        phase = "validate_state"
        if job.status not in {
            JobStatus.created,
            JobStatus.uploading,
            JobStatus.failed,
            JobStatus.cancelled,
            JobStatus.completed,
            JobStatus.summarizing,
        }:
            # If already queued/analyzing/generating_previews/fixing, block duplicate trigger
            if job.status in {JobStatus.queued, JobStatus.analyzing, JobStatus.generating_previews, JobStatus.fixing}:
                log.info("analyze:already_in_progress job_id=%s state=%s", job_id, job.status.value)
                return JSONResponse(
                    status_code=409,
                    content=error_response(
                        "job_in_progress",
                        f"Job is already in progress: {job.status.value}",
                        {"job_id": job_id, "status": job.status.value},
                        status_code=409,
                    ),
                )

        # Schedule background work
        phase = "schedule"
        analyzer = AnalyzerService(state)

        def run():
            sublog = logging.getLogger("api.analyze.worker")
            try:
                sublog.info("analyze:worker_start job_id=%s", job_id)
                state.update_job_status(job_id, JobStatus.analyzing)
                analyzer.analyze_job(job_id)
                state.update_job_status(job_id, JobStatus.generating_previews)
                analyzer.generate_previews(job_id)
                state.update_job_status(job_id, JobStatus.summarizing)
                # Build summary to ensure status/results have content
                ReportService(state).build_summary(job_id)
                state.update_job_status(job_id, JobStatus.completed)
                sublog.info("analyze:worker_done job_id=%s", job_id)
            except ValueError as ve:
                # Map known bad-state errors to failed
                sublog.exception("analyze:worker_value_error job_id=%s err=%s", job_id, ve)
                try:
                    state.update_job_status(job_id, JobStatus.failed)
                except Exception:
                    pass
            except FileNotFoundError as fe:
                sublog.exception("analyze:worker_file_not_found job_id=%s err=%s", job_id, fe)
                try:
                    state.update_job_status(job_id, JobStatus.failed)
                except Exception:
                    pass
            except Exception as e:
                sublog.exception("analyze:worker_unexpected job_id=%s err=%s", job_id, e)
                try:
                    state.update_job_status(job_id, JobStatus.failed)
                except Exception:
                    pass

        background.add_task(run)
        state.update_job_status(job_id, JobStatus.queued)
        resp = {"message": "Analysis queued", "job_id": job_id}
        log.info("analyze:queued job_id=%s phase=%s resp=%s", job_id, phase, resp)
        return resp

    except HTTPException as he:
        # If detail was built via error_response, pass it through; otherwise wrap
        log.exception("analyze:http_exception job_id=%s phase=%s", job_id, phase)
        content = he.detail if isinstance(he.detail, dict) else error_response(
            "http_exception", str(he.detail), {"job_id": job_id}, status_code=he.status_code
        )
        return JSONResponse(status_code=he.status_code, content=content)
    except ValueError as ve:
        log.exception("analyze:value_error job_id=%s phase=%s", job_id, phase)
        return JSONResponse(
            status_code=400,
            content=error_response("invalid_request", str(ve), {"job_id": job_id}, status_code=400),
        )
    except FileNotFoundError as fe:
        log.exception("analyze:file_not_found job_id=%s phase=%s", job_id, phase)
        return JSONResponse(
            status_code=404,
            content=error_response("file_not_found", "Required file not found", {"job_id": job_id, "error": str(fe)}, status_code=404),
        )
    except Exception as e:
        # Catch unexpected server errors to avoid opaque 500 without JSON
        log.exception("analyze:unexpected job_id=%s phase=%s err=%s", job_id, phase, e)
        return JSONResponse(
            status_code=500,
            content=error_response("internal_error", "Internal Server Error", {"job_id": job_id}, status_code=500),
        )


# PUBLIC_INTERFACE
@router.get(
    "/jobs/{job_id}/status",
    summary="Get Job Status",
    description="Retrieve job processing status and progress.",
    response_model=StatusResponse,
    tags=["jobs"],
)
def get_status(job_id: str, state: StateStore = Depends(get_state_store)):
    """Return job progress details."""
    try:
        progress = state.compute_progress(job_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Job not found")
    return StatusResponse(**progress)


# PUBLIC_INTERFACE
@router.get(
    "/jobs/{job_id}/results",
    summary="Get Results",
    description="List assets and issues for a job.",
    tags=["jobs", "assets"],
)
def get_results(job_id: str, state: StateStore = Depends(get_state_store)):
    """Return current assets and issues arrays."""
    from fastapi.encoders import jsonable_encoder

    if state.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    # Optionally enrich assets with simple detections/page info for UI
    assets_list = state.list_assets(job_id)
    # Build a lightweight map of page counts and leave detections to overlays/previews; issues already returned
    payload = {
        "assets": [a.model_dump(mode="json") for a in assets_list],
        "issues": [i.model_dump(mode="json") for i in state.list_issues(job_id)],
        "summary": state.get_summary(job_id).model_dump(mode="json") if state.get_summary(job_id) else None,
    }
    # Final guard to ensure JSON-compatibility
    return jsonable_encoder(payload)


# PUBLIC_INTERFACE
@router.get(
    "/jobs/{job_id}/assets/{asset_id}/preview",
    summary="Get Asset Preview",
    description="Returns a preview file for the given asset. Query params: view=original|overlay|fixed, page (for PDFs).",
    tags=["assets"],
)
def get_asset_preview(
    job_id: str,
    asset_id: str,
    view: Literal["original", "overlay", "fixed"] = Query("original", description="Preview type"),
    page: int | None = Query(default=None, description="0-based page index (PDF only)"),
    state: StateStore = Depends(get_state_store),
):
    """Serve a preview file based on requested view.

    Parameters:
        job_id: Job identifier (UUID).
        asset_id: Asset identifier.
        view: One of original|overlay|fixed.
        page: Optional, 0-based page index when the asset is a PDF.

    PDF behavior:
      - original: If page is provided, returns the rasterized original page image at /jobs/{job_id}/pdf/pages/{page:04d}.png;
                  otherwise returns the original PDF.
      - overlay: If page is provided and overlay exists, returns overlay image; otherwise 404 JSON so the frontend can fallback to original.
      - fixed: If page is provided and fixed page exists, returns it; otherwise 404 JSON so the frontend can fallback to original.
               If page is None and fixed.pdf exists, returns it; else 404 JSON.

    Returns:
        200 FileResponse for existing previews, or 404 JSON with a clear reason string.

    Notes:
        - Use ?page=N to access specific pages of PDFs.
        - Overlays are only available after analysis completes and only for detected pages.
        - Fixed previews/pages are available after apply-fix runs.
    """
    if state.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")

    log = logging.getLogger("api.asset_preview")

    assets = state.list_assets(job_id)
    asset = next((a for a in assets if a.id == asset_id), None)
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    w = get_job_workspace(job_id)

    # Consider both Enum string and value for robustness
    is_pdf = str(getattr(asset, "type", "")) in ("AssetType.pdf", "pdf")

    # Validate page range if provided and page_count known
    if is_pdf and page is not None:
        if page < 0:
            raise HTTPException(status_code=400, detail="Invalid page")
        if getattr(asset, "page_count", None) is not None and page >= int(asset.page_count):
            # The provided page refers to local asset page index; do not 404 yet, let mapping adjust below
            pass

    # For PDFs, map an asset-local page index to a global page index using page_map if provided
    if is_pdf and page is not None:
        try:
            job_dir = w["job"]
            pdf_root = job_dir / "pdf"
            page_map_path = pdf_root / "page_map.json"
            if page_map_path.exists():
                pm = json.loads(page_map_path.read_text(encoding="utf-8")) or {}
                # Find global index mapped to this asset_id and local page index
                for gidx_str, info in pm.items():
                    if isinstance(info, dict) and str(info.get("asset_id")) == str(asset_id) and int(info.get("local_index", -1)) == int(page):
                        # remap page to global index
                        page = int(gidx_str)
                        break
        except Exception:
            # best-effort; fallback to using provided page as global if mapping fails
            pass

    if is_pdf:
        job_dir = w["job"]
        pdf_root = job_dir / "pdf"
        pages_dir = pdf_root / "pages"
        fixed_pages_dir = pdf_root / "fixed_pages"
        final_dir = pdf_root / "final"

        if view == "original":
            if page is None:
                path = job_dir / asset.rel_path
            else:
                cand = pages_dir / f"{page:04d}.png"
                if not cand.exists():
                    log.warning("asset_preview:missing_page_png job_id=%s asset_id=%s page=%s", job_id, asset_id, page)
                    return _png_placeholder(f"job {job_id} - page {page} not rasterized yet")
                path = cand
        elif view == "overlay":
            if page is None:
                path = job_dir / asset.rel_path
            else:
                overlay = w["previews"] / f"overlay_{(pages_dir / f'{page:04d}.png').stem}_p{page:04d}.png"
                if overlay.exists():
                    path = overlay
                else:
                    # Return a JSON 404 so frontend can fallback to original explicitly
                    log.info("asset_preview:overlay_missing job_id=%s asset_id=%s page=%s", job_id, asset_id, page)
                    return JSONResponse(
                        status_code=404,
                        content=error_response(
                            "view_not_available",
                            "Overlay preview not available for this page",
                            {"job_id": job_id, "asset_id": asset_id, "view": "overlay", "page": page},
                            status_code=404,
                        ),
                    )
        else:  # fixed
            if page is None:
                fixed_pdf = final_dir / "fixed.pdf"
                alt = w["outputs"] / f"fixed_{Path(asset.original_filename).stem}.pdf"
                if fixed_pdf.exists():
                    path = fixed_pdf
                elif alt.exists():
                    path = alt
                else:
                    # Return JSON 404 so callers can react (e.g., show badge)
                    log.info("asset_preview:fixed_pdf_missing job_id=%s asset_id=%s", job_id, asset_id)
                    return JSONResponse(
                        status_code=404,
                        content=error_response(
                            "view_not_available",
                            "Fixed PDF not available yet",
                            {"job_id": job_id, "asset_id": asset_id, "view": "fixed"},
                            status_code=404,
                        ),
                    )
            else:
                cand = fixed_pages_dir / f"{page:04d}.png"
                if cand.exists():
                    path = cand
                else:
                    # If page not yet fixed, respond with 404 JSON; frontend will fallback to original
                    op = pages_dir / f"{page:04d}.png"
                    if not op.exists():
                        log.warning("asset_preview:missing_page_png job_id=%s asset_id=%s page=%s", job_id, asset_id, page)
                        return JSONResponse(
                            status_code=404,
                            content=error_response(
                                "page_not_rasterized",
                                "Page image not rasterized yet",
                                {"job_id": job_id, "asset_id": asset_id, "page": page},
                                status_code=404,
                            ),
                        )
                    log.info("asset_preview:fixed_missing job_id=%s asset_id=%s page=%s", job_id, asset_id, page)
                    return JSONResponse(
                        status_code=404,
                        content=error_response(
                            "view_not_available",
                            "Fixed preview not available for this page",
                            {"job_id": job_id, "asset_id": asset_id, "view": "fixed", "page": page},
                            status_code=404,
                        ),
                    )
    else:
        if view == "original":
            path = w["job"] / asset.rel_path
        elif view == "overlay":
            overlay = w["previews"] / f"overlay_{Path(asset.original_filename).stem}.png"
            path = overlay if overlay.exists() else (w["job"] / asset.rel_path)
        else:  # fixed
            fixed_path = w["outputs"] / f"fixed_{asset.original_filename}"
            path = fixed_path if fixed_path.exists() else (w["job"] / asset.rel_path)

    if not path.exists():
        log.warning("asset_preview:path_missing job_id=%s asset_id=%s view=%s path=%s", job_id, asset_id, view, path)
        raise HTTPException(status_code=404, detail="Preview not available")
    guessed, _ = mimetypes.guess_type(str(path))
    media_type = guessed or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)


# PUBLIC_INTERFACE
@router.post(
    "/jobs/{job_id}/assets/{asset_id}/fix",
    summary="Fix Single Asset",
    description="Run an automatic fix for a specific asset.",
    tags=["assets"],
)
def fix_single_asset(job_id: str, asset_id: str, req: FixAssetRequest, request: Request, state: StateStore = Depends(get_state_store)):
    """Apply a stubbed automatic fix for an asset and mark related issues as fixed."""
    fixer = FixerService(state)
    try:
        output = fixer.fix_asset(job_id, asset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    public_url = _build_public_url_for_output(request, Path(output))
    return {"message": "Fixed", "output": str(output), "public_url": public_url}


# PUBLIC_INTERFACE
@router.post(
    "/jobs/{job_id}/fix/batch",
    summary="Batch Fix",
    description="Run automatic fixes across all assets with issues.",
    tags=["assets"],
)
def batch_fix(job_id: str, req: BatchFixRequest, request: Request, state: StateStore = Depends(get_state_store)):
    """Apply fixes across all assets that have issues."""
    fixer = FixerService(state)
    outputs = fixer.fix_all(job_id)
    outputs_list = [str(p) for p in outputs]
    public_urls = [_build_public_url_for_output(request, Path(p)) for p in outputs]
    return {"message": "Batch fixed", "outputs": outputs_list, "public_urls": public_urls}


# PUBLIC_INTERFACE
@router.post(
    "/jobs/{job_id}/apply-fix",
    summary="Apply Fix (job-level)",
    description="Apply fixes across all detected pages and assets; builds final PDFs for documents.",
    tags=["jobs", "assets"],
)
def apply_fix(job_id: str, request: Request, state: StateStore = Depends(get_state_store)):
    """Run job-level apply fix and return final PDF URL if present.

    Returns:
        JSON with list of outputs and optional final_pdf_url (if a PDF asset was fixed).
    """
    # Validate job existence for clearer 404s instead of silent no-op
    job = state.get_job(job_id)
    if job is None:
        from src.api.errors import error_response
        raise HTTPException(
            status_code=404,
            detail=error_response("job_not_found", "Job not found", {"job_id": job_id}, 404),
        )

    fixer = FixerService(state)
    outputs = fixer.fix_all(job_id)
    w = get_job_workspace(job_id)
    final_pdf = (w["job"] / "pdf" / "final" / "fixed.pdf")
    final_pdf_url = None
    if final_pdf.exists():
        final_pdf_url = f"/api/v1/jobs/{job_id}/download?type=pdf"

    # Provide hints when no outputs produced to aid debugging in UI/QA
    hints = {}
    if not outputs:
        hints = {
            "note": "No outputs were produced. Ensure analysis created issues/detections and that brand images were uploaded.",
            "checks": {
                "detections_json": str((w["job"] / "work" / "detections.json")),
                "page_map": str((w["job"] / "pdf" / "page_map.json")),
            },
        }

    return {
        "message": "Apply fix complete",
        "outputs": [str(p) for p in outputs],
        "final_pdf_url": final_pdf_url,
        "hints": hints,
    }


# PUBLIC_INTERFACE
@router.get(
    "/jobs/{job_id}/pages",
    summary="List PDF Pages",
    description="List per-page previews and detection statuses for document pages.",
    tags=["assets", "jobs"],
)
def list_pages(job_id: str, request: Request, state: StateStore = Depends(get_state_store)) -> dict:
    """Return per-page entries with preview URLs and statuses.

    Status logic:
      - fixed if a fixed page raster exists
      - detected if detections.json has >=1 detection with score >= 0.75
      - skipped if no detections above threshold
      - pending if pages exist but analysis has not produced overlays/detections

    Asset mapping:
      - If a page_map.json exists under jobs/{id}/pdf/, use it to associate page indices to asset ids (and local page indices).
      - Else, infer from detections.json keys by matching the base rel_path against assets list.
      - If still ambiguous and a single PDF asset exists, attribute all pages to that asset.
    """
    if state.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    w = get_job_workspace(job_id)
    job_dir = w["job"]
    pdf_root = job_dir / "pdf"
    pages_dir = pdf_root / "pages"
    fixed_pages_dir = pdf_root / "fixed_pages"

    if not pages_dir.exists():
        return {"job_id": job_id, "total": 0, "pages": []}

    # load assets to help map page->asset_id
    assets = {a.rel_path: a for a in state.list_assets(job_id)}
    pdf_assets = [a for a in assets.values() if str(getattr(a, "type", "")) in ("AssetType.pdf", "pdf")]

    # Optional page map
    page_map_path = pdf_root / "page_map.json"
    page_map: dict[str, dict] = {}
    if page_map_path.exists():
        try:
            page_map = json.loads(page_map_path.read_text(encoding="utf-8")) or {}
        except Exception:
            page_map = {}

    # load detections if available
    det_map_path = job_dir / "work" / "detections.json"
    detections_json: dict[str, list[dict]] = {}
    if det_map_path.exists():
        try:
            detections_json = json.loads(det_map_path.read_text(encoding="utf-8"))
        except Exception:
            detections_json = {}
    conf_thr = float(os.getenv("DETECTION_CONFIDENCE_THRESHOLD", "0.70"))

    entries: List[PageEntry] = []
    # sorted by index from filenames {index:04d}.png
    page_paths = sorted([p for p in pages_dir.glob("*.png") if p.is_file()])
    for p in page_paths:
        idx = int(p.stem)
        det_count = 0
        owning_asset_id: Optional[str] = None
        local_index: Optional[int] = None
        rel_path_for_key: Optional[str] = None

        # Determine owning asset via page_map first
        mp = page_map.get(str(idx)) or page_map.get(idx)
        if isinstance(mp, dict):
            maybe_rel = mp.get("asset_rel_path")
            maybe_id = mp.get("asset_id")
            if maybe_id:
                owning_asset_id = str(maybe_id)
            elif maybe_rel and maybe_rel in assets:
                owning_asset_id = assets[maybe_rel].id
            # capture local index mapping for detections lookup
            try:
                local_index = int(mp.get("local_index"))
            except Exception:
                local_index = None
            rel_path_for_key = str(maybe_rel) if maybe_rel else None

        # If not found, infer from detections JSON following legacy pattern
        if owning_asset_id is None:
            for k, arr in detections_json.items():
                if k.endswith(f"::page:{idx}"):  # legacy: when global==local
                    det_count = sum(1 for d in (arr or []) if float(d.get("score", 0.0)) >= conf_thr)
                    base_rel = k.split("::page:")[0]
                    a = assets.get(base_rel)
                    if a:
                        owning_asset_id = a.id
                    break
        else:
            # compute det_count using local_index mapping if available
            if rel_path_for_key is not None and local_index is not None:
                key = f"{rel_path_for_key}::page:{local_index}"
                arr = detections_json.get(key, [])
                det_count = sum(1 for d in (arr or []) if float(d.get("score", 0.0)) >= conf_thr)
            else:
                # fallback to legacy global==local
                for k, arr in detections_json.items():
                    if k.endswith(f"::page:{idx}"):
                        det_count = sum(1 for d in (arr or []) if float(d.get("score", 0.0)) >= conf_thr)
                        break

        # As last resort, if there is exactly one PDF asset, use it
        if owning_asset_id is None and len(pdf_assets) == 1:
            owning_asset_id = pdf_assets[0].id

        fixed_img = fixed_pages_dir / f"{idx:04d}.png"
        if fixed_img.exists():
            status = "fixed"
        else:
            status = "detected" if det_count > 0 else "skipped"
        # If no overlays or detections at all and pages exist, keep "pending"
        overlay = w["previews"] / f"overlay_{p.stem}_p{idx:04d}.png"
        if not overlay.exists() and det_count == 0:
            status = "pending"

        base = f"/api/v1/jobs/{job_id}/pages/{idx}/preview"
        entry = PageEntry(
            index=idx,
            asset_id=owning_asset_id,
            status=status,
            has_detection=bool(det_count > 0),
            original_url=f"{base}?view=original",
            overlay_url=f"{base}?view=overlay",
            fixed_url=f"{base}?view=fixed",
            detections=det_count,
        )
        entries.append(entry)

    return {"job_id": job_id, "total": len(entries), "pages": [e.model_dump(mode='json') for e in entries]}


# PUBLIC_INTERFACE
@router.get(
    "/jobs/{job_id}/pages/{index}/preview",
    summary="Get Page Preview",
    description="Preview for a specific page index with view=original|overlay|fixed.",
    tags=["assets"],
)
def get_page_preview(
    job_id: str,
    index: int,
    view: Literal["original", "overlay", "fixed"] = Query("original", description="Preview type"),
    state: StateStore = Depends(get_state_store),
):
    """Serve a page-level preview file based on requested view.

    Notes:
        - Pages are generated during analysis under jobs/{id}/pdf/pages/{index}.png
        - Overlays are saved under previews/overlay_{index}_p{index}.png
        - Fixed pages under jobs/{id}/pdf/fixed_pages/{index}.png

    Behavior:
        - Returns image/png for existing previews.
        - If a requested view (overlay/fixed) is missing, returns 404 JSON with a clear reason so the frontend can fallback to 'original' gracefully.
        - If the original page image itself is missing (not rasterized yet), returns 404 JSON indicating the page is not ready.
    """
    if state.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if index < 0:
        raise HTTPException(status_code=400, detail="Invalid page index")

    log = logging.getLogger("api.page_preview")
    w = get_job_workspace(job_id)
    job_dir = w["job"]
    pages_dir = job_dir / "pdf" / "pages"
    fixed_pages_dir = job_dir / "pdf" / "fixed_pages"

    page_img = pages_dir / f"{index:04d}.png"
    if not page_img.exists():
        log.warning("page_preview:missing_page job_id=%s index=%d path=%s", job_id, index, page_img)
        return JSONResponse(
            status_code=404,
            content=error_response(
                "page_not_rasterized",
                "Page image not rasterized yet",
                {"job_id": job_id, "page_index": index},
                status_code=404,
            ),
        )

    if view == "original":
        path = page_img
        guessed, _ = mimetypes.guess_type(str(path))
        media_type = guessed or "image/png"
        return FileResponse(path, media_type=media_type, filename=path.name)
    elif view == "overlay":
        overlay = w["previews"] / f"overlay_{page_img.stem}_p{index:04d}.png"
        if overlay.exists():
            path = overlay
            guessed, _ = mimetypes.guess_type(str(path))
            media_type = guessed or "image/png"
            return FileResponse(path, media_type=media_type, filename=path.name)
        # overlay not present -> JSON 404 for explicit fallback on frontend
        log.info("page_preview:overlay_missing job_id=%s index=%d", job_id, index)
        return JSONResponse(
            status_code=404,
            content=error_response(
                "view_not_available",
                "Overlay preview not available for this page",
                {"job_id": job_id, "page_index": index, "view": "overlay"},
                status_code=404,
            ),
        )
    else:  # fixed
        fixed = fixed_pages_dir / f"{index:04d}.png"
        if fixed.exists():
            path = fixed
            guessed, _ = mimetypes.guess_type(str(path))
            media_type = guessed or "image/png"
            return FileResponse(path, media_type=media_type, filename=path.name)
        # fixed not present -> JSON 404 so frontend can show badge/fallback
        log.info("page_preview:fixed_missing job_id=%s index=%d", job_id, index)
        return JSONResponse(
            status_code=404,
            content=error_response(
                "view_not_available",
                "Fixed preview not available for this page",
                {"job_id": job_id, "page_index": index, "view": "fixed"},
                status_code=404,
            ),
        )


# PUBLIC_INTERFACE
@router.get(
    "/jobs/{job_id}/download",
    summary="Download Artifacts",
    description="Download outputs/report for a job. Query param type=zip|report|both|pdf.",
    tags=["report"],
)
def download_artifacts(
    job_id: str,
    type: Literal["zip", "report", "both", "pdf"] = Query("zip", description="Artifact type to download"),
    state: StateStore = Depends(get_state_store),
):
    """Create and return the requested artifact.

    pdf type:
        Returns the final assembled fixed.pdf if available (after apply-fix); 404 if not present.
    """
    if state.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if type == "pdf":
        w = get_job_workspace(job_id)
        fixed_pdf = w["job"] / "pdf" / "final" / "fixed.pdf"
        if not fixed_pdf.exists():
            raise HTTPException(status_code=404, detail="Final PDF not available")
        return FileResponse(fixed_pdf, media_type="application/pdf", filename=fixed_pdf.name)

    report = ReportService(state)
    try:
        target = report.prepare_downloads(job_id, type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    filename = target.name
    media_type = "application/zip" if filename.endswith(".zip") else "application/json"
    return FileResponse(target, media_type=media_type, filename=filename)
