from __future__ import annotations

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, StrictStr, StrictFloat, ConfigDict

from .asset import BoundingBox


class IssueSeverity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class IssueType(str, Enum):
    old_logo = "old_logo"
    wrong_logo = "wrong_logo"
    old_text = "old_text"
    color_mismatch = "color_mismatch"
    font_mismatch = "font_mismatch"
    layout_violation = "layout_violation"
    other = "other"


class IssueStatus(str, Enum):
    open = "open"
    fixed = "fixed"
    wont_fix = "wont_fix"


class Issue(BaseModel):
    """Represents a detected compliance issue for an asset."""
    model_config = ConfigDict(extra="forbid")

    id: StrictStr = Field(..., description="Unique issue identifier.")
    job_id: StrictStr = Field(..., description="The job this issue belongs to.")
    asset_id: StrictStr = Field(..., description="Asset the issue is associated with.")
    type: IssueType = Field(..., description="Issue category.")
    severity: IssueSeverity = Field(..., description="Severity of the issue.")
    message: StrictStr = Field(..., description="Human-readable description.")
    bbox: Optional[BoundingBox] = Field(None, description="Location of the issue within the asset, when applicable.")
    score: Optional[StrictFloat] = Field(None, description="Detector confidence score or severity score.")
    suggestions: List[StrictStr] = Field(default_factory=list, description="Suggested fixes or notes.")
    status: IssueStatus = Field(IssueStatus.open, description="Current status of the issue.")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional structured data for this issue.")
