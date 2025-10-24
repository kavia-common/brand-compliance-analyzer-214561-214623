from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import logging
from pathlib import Path

from src.api.v1 import router as v1_router
from src.api.compat import router as compat_router
from src.services.state_store import StateStore
from src.storage.workspace import get_root_workspace

# Basic logging setup
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Branding Compliance Backend",
    description="Processes images/documents for branding compliance and exposes RESTful APIs.",
    version="0.1.0",
    openapi_tags=[
        {"name": "health", "description": "Health and diagnostics"},
        {"name": "jobs", "description": "Job lifecycle and status"},
        {"name": "assets", "description": "Asset upload and processing"},
        {"name": "issues", "description": "Detected issues and annotations"},
        {"name": "report", "description": "Report summaries and downloads"},
    ],
)

# Global state store instance; in larger apps this would be managed via DI container
state_store = StateStore()

# Configure CORS: allow local frontend and optional preview origin
default_origins = [
    "https://vscode-internal-27606-beta.beta01.cloud.kavia.ai:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
]
preview_origin = os.getenv("PREVIEW_FRONTEND_ORIGIN")
if preview_origin:
    default_origins.append(preview_origin)

extra_origins = os.getenv("CORS_EXTRA_ORIGINS", "https://vscode-internal-27606-beta.beta01.cloud.kavia.ai:3000")
if extra_origins:
    # comma-separated list
    default_origins.extend([o.strip() for o in extra_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=default_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount a static files route to expose job outputs via HTTP
# This maps to: {workspace_root}/jobs which contains <job_id>/outputs/...
# Public URL shape: /outputs/{job_id}/outputs/<file_name>
try:
    ws_root: Path = get_root_workspace()
    jobs_root = ws_root / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/outputs",
        StaticFiles(directory=str(jobs_root), html=False),
        name="outputs",
    )
except Exception as e:
    logging.getLogger("api.static").exception("Failed to mount static outputs: %s", e)

# PUBLIC_INTERFACE
@app.get("/", tags=["health"], summary="Health Check")
def health_check():
    """Health check endpoint to verify service is responsive.

    Returns:
        JSON payload with a status message and a simple state store readiness flag.
    """
    ready = state_store is not None
    return {"message": "Healthy", "state_store_ready": ready}


# PUBLIC_INTERFACE
@app.get(
    "/api/v1/docs/notes",
    tags=["health"],
    summary="API Notes",
)
def api_notes():
    """Provide usage notes for API clients.

    Includes info about background tasks and preview endpoints.
    """
    return {
        "notes": "Analysis runs asynchronously via BackgroundTasks. Poll /api/v1/jobs/{job_id}/status and /results.",
        "previews": "Use /api/v1/jobs/{job_id}/assets/{asset_id}/preview?view=original|overlay|fixed",
        "downloads": "Use /api/v1/jobs/{job_id}/download?type=zip|report|both; server sets Content-Disposition and proper Content-Type.",
        "public_files": "Outputs are served under /outputs/{job_id}/outputs/<file_name> for direct browser access.",
        "cors": {
            "allow_origins": default_origins,
            "note": "Configure PREVIEW_FRONTEND_ORIGIN and CORS_EXTRA_ORIGINS env vars to add more origins.",
        },
    }


# Mount API v1 routes
app.include_router(v1_router)

# Mount root-level compatibility routes so frontend calling "/jobs" works
# Note: OpenAPI docs focus on /api/v1; these are provided for backward/compat use.
app.include_router(compat_router)

# PUBLIC_INTERFACE
@app.get("/api/v1/health", tags=["health"], summary="Service Health")
def api_health():
    """Service-level health check for API v1.

    Returns:
        JSON with status and readiness including workspace root and state store readiness.
    """
    import logging

    logger = logging.getLogger("api.health")
    ws_ok = False
    root_str = None
    disk_write_ok = False
    try:
        from src.storage.workspace import get_root_workspace  # local import to avoid circulars at import time
        root = get_root_workspace()
        root_str = str(root)
        # Disk write check: touch a temp file under workspace root
        probe = root / ".health_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        disk_write_ok = True
        ws_ok = True
        logger.info("Health workspace root resolved: %s", root_str)
    except Exception as e:
        logger.exception("Health check workspace failure: %s", e)
        ws_ok = False

    status = "ok" if (ws_ok and disk_write_ok) else "degraded"
    return {
        "status": status,
        "state_store_ready": state_store is not None,
        "workspace_root": root_str,
        "disk_write_ok": disk_write_ok,
    }
