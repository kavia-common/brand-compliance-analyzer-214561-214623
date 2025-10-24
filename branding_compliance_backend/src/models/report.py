from __future__ import annotations

from typing import Dict, Any
from pydantic import BaseModel, Field, StrictInt, StrictStr, ConfigDict


class ReportSummary(BaseModel):
    """Aggregated report-level metrics and derived info."""
    model_config = ConfigDict(extra="forbid")

    job_id: StrictStr = Field(..., description="The job this report is for.")
    total_assets: StrictInt = Field(..., description="Total assets processed.")
    analyzed_assets: StrictInt = Field(..., description="Number of assets analyzed.")
    failed_assets: StrictInt = Field(..., description="Number of assets failed.")
    total_issues: StrictInt = Field(..., description="Count of all detected issues.")
    high_or_above_issues: StrictInt = Field(..., description="Issues of severity high or critical.")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional aggregate metrics or notes.")
