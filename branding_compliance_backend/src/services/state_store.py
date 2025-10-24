from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from pydantic import TypeAdapter

from src.models.job import Job, JobStatus
from src.models.asset import Asset, AssetStatus
from src.models.issue import Issue, IssueSeverity
from src.models.report import ReportSummary
from src.storage.workspace import get_job_workspace


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON to a file atomically using a temp file and replace.

    Ensures datetime and other non-JSON-native types are encoded as JSON-compatible
    using FastAPI/Pydantic encoders.
    """
    from fastapi.encoders import jsonable_encoder

    path.parent.mkdir(parents=True, exist_ok=True)
    # Normalize data through jsonable_encoder to convert datetimes/enums to JSON-compatible types
    encoded = jsonable_encoder(data)
    # Use text mode to ensure UTF-8 and pretty formatting
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        json.dump(encoded, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_name = tmp.name
    os.replace(temp_name, path)


def _safe_read_json(path: Path) -> Any:
    """Safely read a JSON file and handle empty or partial writes."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # Attempt to read again in case of concurrent write; if still bad, treat as missing
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    except Exception:
        # Any other IO error should be treated as missing to avoid 500
        return None


class StateStore:
    """Filesystem JSON state store for Jobs, Assets, Issues, and ReportSummary."""

    def __init__(self) -> None:
        # No in-memory state; filesystem-based.
        # Ensures workspace root exists early.
        from src.storage.workspace import get_root_workspace
        try:
            _ = get_root_workspace()
        except Exception:
            # Defer errors; health endpoint will reflect degraded state
            pass

    def _paths(self, job_id: str) -> Dict[str, Path]:
        w = get_job_workspace(job_id)
        return {
            "job_json": w["job"] / "job.json",
            "assets_json": w["job"] / "assets.json",
            "issues_json": w["job"] / "issues.json",
            "summary_json": w["job"] / "summary.json",
        }

    # PUBLIC_INTERFACE
    def create_job(self, job: Job) -> Job:
        """Create a new job and persist its initial JSON state."""
        # Ensure workspace root and job directories exist
        w = get_job_workspace(job.id)
        paths = self._paths(job.id)
        if paths["job_json"].exists():
            raise ValueError(f"Job {job.id} already exists")

        # Normalize and set timestamps
        job.created_at = datetime.utcnow()
        job.updated_at = job.created_at

        try:
            # initialize companion files atomically
            _atomic_write_json(paths["job_json"], job.model_dump(mode="json"))
            _atomic_write_json(paths["assets_json"], [])
            _atomic_write_json(paths["issues_json"], [])
            _atomic_write_json(paths["summary_json"], None)
        except Exception as e:
            # Attempt cleanup of a partially created job folder to avoid corrupt state
            try:
                # Do not delete entire root, only specific job dir
                import shutil
                shutil.rmtree(w["job"], ignore_errors=True)
            except Exception:
                # Ignore cleanup errors
                pass
            raise e
        return job

    # PUBLIC_INTERFACE
    def get_job(self, job_id: str) -> Optional[Job]:
        """Load a job by id, or None if not found."""
        paths = self._paths(job_id)
        raw = _safe_read_json(paths["job_json"])
        if raw is None:
            return None
        return Job.model_validate(raw)

    # PUBLIC_INTERFACE
    def update_job_status(self, job_id: str, status: JobStatus) -> Job:
        """Update job status and updated_at timestamp."""
        job = self.get_job(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")
        job.status = status
        job.updated_at = datetime.utcnow()
        _atomic_write_json(self._paths(job_id)["job_json"], job.model_dump(mode="json"))
        return job

    def _load_assets(self, job_id: str) -> List[Asset]:
        raw = _safe_read_json(self._paths(job_id)["assets_json"]) or []
        adapter = TypeAdapter(List[Asset])
        return adapter.validate_python(raw)

    def _save_assets(self, job_id: str, assets: List[Asset]) -> None:
        _atomic_write_json(self._paths(job_id)["assets_json"], [a.model_dump(mode="json") for a in assets])

    # PUBLIC_INTERFACE
    def add_assets(self, job_id: str, assets: List[Asset]) -> List[Asset]:
        """Append new assets to the job's asset list and update job counters."""
        if not assets:
            return self._load_assets(job_id)

        existing = self._load_assets(job_id)
        existing_ids = {a.id for a in existing}
        to_add = [a for a in assets if a.id not in existing_ids]

        if not to_add:
            return existing

        # Persist
        combined = existing + to_add
        self._save_assets(job_id, combined)

        # Update job counters
        job = self.get_job(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")
        job.total_assets = len(combined)
        job.updated_at = datetime.utcnow()
        _atomic_write_json(self._paths(job_id)["job_json"], job.model_dump())

        return combined

    # PUBLIC_INTERFACE
    def update_asset_status(self, job_id: str, asset_id: str, status: AssetStatus) -> Asset:
        """Update a specific asset's status and adjust job counters."""
        assets = self._load_assets(job_id)
        found = False
        for a in assets:
            if a.id == asset_id:
                a.status = status
                found = True
                break

        if not found:
            raise ValueError(f"Asset {asset_id} not found for job {job_id}")

        self._save_assets(job_id, assets)

        # Recompute counters
        job = self.get_job(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        analyzed = sum(1 for a in assets if a.status in (AssetStatus.analyzed, AssetStatus.fixed, AssetStatus.failed))
        failed = sum(1 for a in assets if a.status == AssetStatus.failed)

        job.analyzed_assets = analyzed
        job.failed_assets = failed
        job.total_assets = len(assets)
        job.updated_at = datetime.utcnow()
        _atomic_write_json(self._paths(job_id)["job_json"], job.model_dump())

        return next(a for a in assets if a.id == asset_id)

    def _load_issues(self, job_id: str) -> List[Issue]:
        raw = _safe_read_json(self._paths(job_id)["issues_json"]) or []
        adapter = TypeAdapter(List[Issue])
        return adapter.validate_python(raw)

    def _save_issues(self, job_id: str, issues: List[Issue]) -> None:
        _atomic_write_json(self._paths(job_id)["issues_json"], [i.model_dump(mode="json") for i in issues])

    # PUBLIC_INTERFACE
    def append_issues(self, job_id: str, issues: List[Issue]) -> List[Issue]:
        """Append issues to the job and return the updated list."""
        if not issues:
            return self._load_issues(job_id)

        existing = self._load_issues(job_id)
        existing_ids = {i.id for i in existing}
        to_add = [i for i in issues if i.id not in existing_ids]

        updated = existing + to_add
        self._save_issues(job_id, updated)
        return updated

    def _load_summary(self, job_id: str) -> Optional[ReportSummary]:
        raw = _safe_read_json(self._paths(job_id)["summary_json"])
        if raw in (None, "null"):
            return None
        return ReportSummary.model_validate(raw)

    def _save_summary(self, job_id: str, summary: Optional[ReportSummary]) -> None:
        if summary is None:
            _atomic_write_json(self._paths(job_id)["summary_json"], None)
        else:
            _atomic_write_json(self._paths(job_id)["summary_json"], summary.model_dump(mode="json"))

    # PUBLIC_INTERFACE
    def save_summary(self, job_id: str, summary: ReportSummary) -> ReportSummary:
        """Persist a summary for the job."""
        if summary.job_id != job_id:
            raise ValueError("Summary.job_id must match job_id")
        self._save_summary(job_id, summary)
        return summary

    # PUBLIC_INTERFACE
    def compute_progress(self, job_id: str) -> Dict[str, Any]:
        """Compute current progress metrics for a job."""
        job = self.get_job(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        assets = self._load_assets(job_id)
        issues = self._load_issues(job_id)

        total = len(assets)
        analyzed = sum(1 for a in assets if a.status in (AssetStatus.analyzed, AssetStatus.fixed, AssetStatus.failed))
        failed = sum(1 for a in assets if a.status == AssetStatus.failed)
        open_high_or_above = sum(
            1
            for i in issues
            if (i.severity in (IssueSeverity.high, IssueSeverity.critical))
        )

        pct = (analyzed / total * 100.0) if total > 0 else 0.0

        return {
            "job_id": job_id,
            "status": job.status.value,
            "total_assets": total,
            "analyzed_assets": analyzed,
            "failed_assets": failed,
            "progress_percent": round(pct, 2),
            "issues_total": len(issues),
            "issues_high_or_above": open_high_or_above,
            "updated_at": job.updated_at.isoformat(),
        }

    # Convenience helpers (not marked as public interface)
    def list_assets(self, job_id: str) -> List[Asset]:
        return self._load_assets(job_id)

    def list_issues(self, job_id: str) -> List[Issue]:
        return self._load_issues(job_id)

    def get_summary(self, job_id: str) -> Optional[ReportSummary]:
        return self._load_summary(job_id)
