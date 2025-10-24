from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging

from src.api.v1 import router as v1_router
from src.services.state_store import StateStore

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
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
preview_origin = os.getenv("PREVIEW_FRONTEND_ORIGIN")
if preview_origin:
    default_origins.append(preview_origin)

extra_origins = os.getenv("CORS_EXTRA_ORIGINS", "")
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
        "cors": {
            "allow_origins": default_origins,
            "note": "Configure PREVIEW_FRONTEND_ORIGIN and CORS_EXTRA_ORIGINS env vars to add more origins.",
        },
    }


# Mount API v1 routes
app.include_router(v1_router)

# PUBLIC_INTERFACE
@app.get("/api/v1/health", tags=["health"], summary="Service Health")
def api_health():
    """Service-level health check for API v1.

    Returns:
        JSON with status and readiness including workspace root and state store readiness.
    """
    try:
        from src.storage.workspace import get_root_workspace  # local import to avoid circulars at import time
        root = str(get_root_workspace())
        ws_ok = True
    except Exception:
        root = None
        ws_ok = False
    return {"status": "ok" if ws_ok else "degraded", "state_store_ready": state_store is not None, "workspace_root": root}
