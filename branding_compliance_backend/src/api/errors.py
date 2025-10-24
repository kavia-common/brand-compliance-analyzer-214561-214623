from __future__ import annotations

import logging
from typing import Any, Dict, Optional


def _safe_str(err: Any) -> str:
    try:
        return str(err)
    except Exception:
        return "Unknown error"


# PUBLIC_INTERFACE
def error_response(code: str, message: str, details: Optional[Dict[str, Any]] = None, status_code: int = 400) -> Dict[str, Any]:
    """Build a standardized error payload used across endpoints.

    Returns a dict to be used as JSON content; the FastAPI route should set the status_code.
    """
    payload: Dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        }
    }
    # Optional: lightweight log here; prefer route-level logs for context like job_id/phase
    logging.getLogger("api.error").debug("Composed error payload code=%s status=%s", code, status_code)
    return payload
