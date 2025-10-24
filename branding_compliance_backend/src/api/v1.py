from __future__ import annotations

import re
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse
import mimetypes
import logging
import os
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
    payload = {
        "assets": [a.model_dump(mode="json") for a in state.list_assets(job_id)],
        "issues": [i.model_dump(mode="json") for i in state.list_issues(job_id)],
        "summary": state.get_summary(job_id).model_dump(mode="json") if state.get_summary(job_id) else None,
    }
    # Final guard to ensure JSON-compatibility
    return jsonable_encoder(payload)


# PUBLIC_INTERFACE
@router.get(
    "/jobs/{job_id}/assets/{asset_id}/preview",
    summary="Get Asset Preview",
    description="Returns a preview file for the given asset. Query param view=original|overlay|fixed.",
    tags=["assets"],
)
def get_asset_preview(
    job_id: str,
    asset_id: str,
    view: Literal["original", "overlay", "fixed"] = Query("original", description="Preview type"),
    state: StateStore = Depends(get_state_store),
):
    """Serve a preview file based on requested view."""
    if state.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")

    assets = state.list_assets(job_id)
    asset = next((a for a in assets if a.id == asset_id), None)
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    w = get_job_workspace(job_id)

    if view == "original":
        path = w["job"] / asset.rel_path
    elif view == "overlay":
        # Stub overlay: if exists, return it, else return original
        overlay = w["previews"] / f"overlay_{Path(asset.original_filename).stem}.png"
        path = overlay if overlay.exists() else (w["job"] / asset.rel_path)
    else:  # fixed
        fixed_path = w["outputs"] / f"fixed_{asset.original_filename}"
        path = fixed_path if fixed_path.exists() else (w["job"] / asset.rel_path)

    if not path.exists():
        raise HTTPException(status_code=404, detail="Preview not available")
    # Infer media type for correct rendering in UI
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
@router.get(
    "/jobs/{job_id}/download",
    summary="Download Artifacts",
    description="Download outputs/report for a job. Query param type=zip|report|both.",
    tags=["report"],
)
def download_artifacts(
    job_id: str,
    type: Literal["zip", "report", "both"] = Query("zip", description="Artifact type to download"),
    state: StateStore = Depends(get_state_store),
):
    """Create and return the requested artifact."""
    if state.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    report = ReportService(state)
    try:
        target = report.prepare_downloads(job_id, type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    filename = target.name
    media_type = "application/zip" if filename.endswith(".zip") else "application/json"
    return FileResponse(target, media_type=media_type, filename=filename)
