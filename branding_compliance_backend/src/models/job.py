from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, StrictStr, StrictInt, ConfigDict


class JobStatus(str, Enum):
    created = "created"
    uploading = "uploading"
    queued = "queued"
    analyzing = "analyzing"
    generating_previews = "generating_previews"
    fixing = "fixing"
    summarizing = "summarizing"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class Job(BaseModel):
    """Represents a processing job tracking lifecycle and related data."""
    model_config = ConfigDict(extra="forbid")

    id: StrictStr = Field(..., description="Unique job identifier (UUID).")
    status: JobStatus = Field(JobStatus.created, description="Lifecycle status for the job.")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp for creation.")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp for last update.")
    owner: Optional[StrictStr] = Field(None, description="Optional owner/user id.")
    title: Optional[StrictStr] = Field(None, description="Optional human-friendly job title.")
    description: Optional[StrictStr] = Field(None, description="Optional job description.")
    total_assets: StrictInt = Field(0, description="Total number of assets expected or discovered.")
    analyzed_assets: StrictInt = Field(0, description="Number of assets already analyzed.")
    failed_assets: StrictInt = Field(0, description="Number of assets that failed processing.")
    tags: List[StrictStr] = Field(default_factory=list, description="Labels for filtering or grouping.")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata and settings.")
