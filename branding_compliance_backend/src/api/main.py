from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.services.state_store import StateStore

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    # Minimal wiring: ensure state store can be instantiated and workspace is available on demand.
    # We avoid creating any files here; creation occurs when jobs are created.
    ready = state_store is not None
    return {"message": "Healthy", "state_store_ready": ready}
