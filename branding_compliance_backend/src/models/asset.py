from __future__ import annotations

from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, StrictStr, StrictInt, StrictFloat, ConfigDict


class AssetType(str, Enum):
    image = "image"
    pdf = "pdf"
    document = "document"
    other = "other"


class AssetStatus(str, Enum):
    pending = "pending"
    analyzing = "analyzing"
    analyzed = "analyzed"
    failed = "failed"
    fixed = "fixed"  # after auto-fix


class BoundingBox(BaseModel):
    """Axis-aligned bounding box for highlighting issues within assets."""
    model_config = ConfigDict(extra="forbid")
    x: StrictFloat = Field(..., description="Top-left x coordinate in pixels or normalized.")
    y: StrictFloat = Field(..., description="Top-left y coordinate in pixels or normalized.")
    width: StrictFloat = Field(..., description="Box width.")
    height: StrictFloat = Field(..., description="Box height.")
    normalized: bool = Field(False, description="If True, values are in [0,1] relative to asset size.")


class Asset(BaseModel):
    """Represents an input or derived asset belonging to a Job."""
    model_config = ConfigDict(extra="forbid")

    id: StrictStr = Field(..., description="Unique identifier for the asset (UUID or hash).")
    job_id: StrictStr = Field(..., description="The job this asset belongs to.")
    type: AssetType = Field(..., description="Type of asset.")
    original_filename: StrictStr = Field(..., description="Original filename as uploaded.")
    rel_path: StrictStr = Field(..., description="Relative path under the job workspace (e.g., uploads/img1.png).")
    status: AssetStatus = Field(AssetStatus.pending, description="Processing status of the asset.")
    width: Optional[StrictInt] = Field(None, description="Width of asset if applicable (e.g., pixels).")
    height: Optional[StrictInt] = Field(None, description="Height of asset if applicable (e.g., pixels).")
    page_count: Optional[StrictInt] = Field(None, description="Page count for documents if applicable.")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata for the asset (MIME, checksum, etc.).")
