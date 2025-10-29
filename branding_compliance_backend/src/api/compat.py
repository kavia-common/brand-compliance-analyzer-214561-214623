from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, Request, BackgroundTasks

# Reuse models and services from v1 to ensure identical behavior/serialization
from src.api.v1 import (
    CreateJobRequest,
    CreateJobResponse,
    StatusResponse,
    FixAssetRequest,
    BatchFixRequest,
    create_job as v1_create_job,
    delete_job as v1_delete_job,
    upload_assets_zip as v1_upload_assets_zip,
    upload_old_brand as v1_upload_old_brand,
    upload_new_brand as v1_upload_new_brand,
    analyze_job as v1_analyze_job,
    get_status as v1_get_status,
    get_results as v1_get_results,
    get_asset_preview as v1_get_asset_preview,
    fix_single_asset as v1_fix_single_asset,
    batch_fix as v1_batch_fix,
    download_artifacts as v1_download_artifacts,
    get_state_store,
)
from src.services.state_store import StateStore


# Router without a prefix to expose root-level endpoints for frontend compatibility.
router = APIRouter(tags=["jobs"])


def _error_payload(status_code: int, message: str):
    # Provide a standardized error JSON shape for the frontend
    # so handleJson can consume it safely when non-2xx.
    return {"error": True, "message": message, "status": status_code}


# PUBLIC_INTERFACE
@router.post(
    "/jobs",
    summary="Create Job (compat)",
    description="Compatibility alias for POST /api/v1/jobs.",
    response_model=CreateJobResponse,
    tags=["jobs"],
    status_code=201,
)
def compat_create_job(data: CreateJobRequest, state: StateStore = Depends(get_state_store)):
    """Compatibility endpoint that forwards to v1 create_job."""
    return v1_create_job(data, state)  # returns CreateJobResponse


# PUBLIC_INTERFACE
@router.delete(
    "/jobs/{job_id}",
    summary="Delete Job (compat)",
    description="Compatibility alias for DELETE /api/v1/jobs/{job_id}.",
    tags=["jobs"],
)
def compat_delete_job(job_id: str, state: StateStore = Depends(get_state_store)):
    """Compatibility endpoint that forwards to v1 delete_job."""
    try:
        return v1_delete_job(job_id, state)
    except HTTPException as e:
        # Standardize error payload
        raise HTTPException(status_code=e.status_code, detail=_error_payload(e.status_code, str(e.detail)))


# PUBLIC_INTERFACE
@router.post(
    "/jobs/{job_id}/upload/assets",
    summary="Upload Assets Zip (compat)",
    description="Compatibility alias for POST /api/v1/jobs/{job_id}/upload/assets.",
    tags=["assets"],
)
async def compat_upload_assets_zip(
    job_id: str,
    file: UploadFile = File(...),
    state: StateStore = Depends(get_state_store),
):
    """Compatibility endpoint for uploading zip to v1 handler."""
    try:
        return await v1_upload_assets_zip(job_id, file, state)
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=_error_payload(e.status_code, str(e.detail)))


# PUBLIC_INTERFACE
@router.post(
    "/jobs/{job_id}/upload/old-brand",
    summary="Upload Old Brand Image (compat)",
    description="Compatibility alias for POST /api/v1/jobs/{job_id}/upload/old-brand.",
    tags=["assets"],
)
async def compat_upload_old_brand(
    job_id: str,
    file: UploadFile = File(...),
    state: StateStore = Depends(get_state_store),
):
    try:
        return await v1_upload_old_brand(job_id, file, state)
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=_error_payload(e.status_code, str(e.detail)))


# PUBLIC_INTERFACE
@router.post(
    "/jobs/{job_id}/upload/new-brand",
    summary="Upload New Brand Image (compat)",
    description="Compatibility alias for POST /api/v1/jobs/{job_id}/upload/new-brand.",
    tags=["assets"],
)
async def compat_upload_new_brand(
    job_id: str,
    file: UploadFile = File(...),
    state: StateStore = Depends(get_state_store),
):
    try:
        return await v1_upload_new_brand(job_id, file, state)
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=_error_payload(e.status_code, str(e.detail)))


# PUBLIC_INTERFACE
@router.post(
    "/jobs/{job_id}/analyze",
    summary="Trigger Analysis (compat)",
    description="Compatibility alias for POST /api/v1/jobs/{job_id}/analyze.",
    tags=["jobs"],
)
def compat_analyze_job(job_id: str, background: BackgroundTasks, request: Request, state: StateStore = Depends(get_state_store)):
    try:
        # Forward background and request context so v1 can schedule work and log origin/headers.
        return v1_analyze_job(job_id, background=background, request=request, state=state)  # type: ignore[arg-type]
    except HTTPException as e:
        # If v1 provided dict detail (structured), pass through; else wrap
        detail = e.detail if isinstance(e.detail, dict) else _error_payload(e.status_code, str(e.detail))
        raise HTTPException(status_code=e.status_code, detail=detail)


# PUBLIC_INTERFACE
@router.get(
    "/jobs/{job_id}/status",
    summary="Get Job Status (compat)",
    description="Compatibility alias for GET /api/v1/jobs/{job_id}/status.",
    response_model=StatusResponse,
    tags=["jobs"],
)
def compat_get_status(job_id: str, state: StateStore = Depends(get_state_store)):
    try:
        return v1_get_status(job_id, state)
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=_error_payload(e.status_code, str(e.detail)))


# PUBLIC_INTERFACE
@router.get(
    "/jobs/{job_id}/results",
    summary="Get Results (compat)",
    description="Compatibility alias for GET /api/v1/jobs/{job_id}/results.",
    tags=["jobs", "assets"],
)
def compat_get_results(job_id: str, state: StateStore = Depends(get_state_store)):
    try:
        return v1_get_results(job_id, state)
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=_error_payload(e.status_code, str(e.detail)))


# PUBLIC_INTERFACE
@router.get(
    "/jobs/{job_id}/assets/{asset_id}/preview",
    summary="Get Asset Preview (compat)",
    description="Compatibility alias for GET /api/v1/jobs/{job_id}/assets/{asset_id}/preview.",
    tags=["assets"],
)
def compat_get_asset_preview(
    job_id: str,
    asset_id: str,
    view: Literal["original", "overlay", "fixed"] = Query("original", description="Preview type"),
    page: int | None = Query(default=None, description="0-based page index (PDF only)"),
    state: StateStore = Depends(get_state_store),
):
    try:
        return v1_get_asset_preview(job_id, asset_id, view, page, state)
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=_error_payload(e.status_code, str(e.detail)))


# PUBLIC_INTERFACE
@router.post(
    "/jobs/{job_id}/assets/{asset_id}/fix",
    summary="Fix Single Asset (compat)",
    description="Compatibility alias for POST /api/v1/jobs/{job_id}/assets/{asset_id}/fix.",
    tags=["assets"],
)
def compat_fix_single_asset(job_id: str, asset_id: str, req: FixAssetRequest, request: Request, state: StateStore = Depends(get_state_store)):
    try:
        # v1 function will now include public_url in response; passthrough
        return v1_fix_single_asset(job_id, asset_id, req, request, state)  # type: ignore[arg-type]
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=_error_payload(e.status_code, str(e.detail)))


# PUBLIC_INTERFACE
@router.post(
    "/jobs/{job_id}/fix/batch",
    summary="Batch Fix (compat)",
    description="Compatibility alias for POST /api/v1/jobs/{job_id}/fix/batch.",
    tags=["assets"],
)
def compat_batch_fix(job_id: str, req: BatchFixRequest, request: Request, state: StateStore = Depends(get_state_store)):
    try:
        # v1 now returns public_urls; passthrough unchanged
        return v1_batch_fix(job_id, req, request, state)  # type: ignore[arg-type]
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=_error_payload(e.status_code, str(e.detail)))


# PUBLIC_INTERFACE
@router.get(
    "/jobs/{job_id}/download",
    summary="Download Artifacts (compat)",
    description="Compatibility alias for GET /api/v1/jobs/{job_id}/download.",
    tags=["report"],
)
def compat_download_artifacts(
    job_id: str,
    type: Literal["zip", "report", "both"] = Query("zip", description="Artifact type to download"),
    state: StateStore = Depends(get_state_store),
):
    try:
        return v1_download_artifacts(job_id, type, state)
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=_error_payload(e.status_code, str(e.detail)))
