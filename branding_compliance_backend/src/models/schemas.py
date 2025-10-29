from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Detection bounding box in page pixel space (rasterized)."""
    x: int = Field(..., description="Top-left X in pixels")
    y: int = Field(..., description="Top-left Y in pixels")
    w: int = Field(..., description="Width in pixels")
    h: int = Field(..., description="Height in pixels")
    score: float = Field(..., description="Confidence score (0-1)")
    template_id: int = Field(..., description="Index of which old_logo template matched")


class PageFindings(BaseModel):
    """Findings for a single page."""
    page_index: int = Field(..., description="Zero-based page index")
    detections: List[BoundingBox] = Field(default_factory=list, description="Detected old logos on this page")


class Findings(BaseModel):
    """Aggregate findings across pages."""
    total_pages: int = Field(..., description="Number of pages processed")
    pages: List[PageFindings] = Field(default_factory=list, description="Findings by page")


class StartRequestParams(BaseModel):
    """Parameters for job start (form values only; files handled separately)."""
    dpi: int = Field(250, description="Rasterization DPI")
    max_pages: Optional[int] = Field(None, description="Optional cap on number of pages")
    match_threshold: float = Field(0.8, description="Template matching threshold")


class StatusResponse(BaseModel):
    """Status for a job."""
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="one of: queued, processing, completed, failed")
    progress: float = Field(0.0, description="0.0-1.0 fractional progress")
    message: Optional[str] = Field(None, description="Optional status message or error detail")
    findings: Optional[Findings] = Field(None, description="Findings if available (after detection phase)")
