from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from src.models.asset import Asset, AssetStatus, AssetType
from src.models.issue import Issue, IssueSeverity, IssueType
from src.models.job import JobStatus
from src.services.state_store import StateStore
from src.storage.workspace import get_job_workspace


class AnalyzerService:
    """Performs analysis for a job's uploaded assets.

    This is a stubbed implementation that simulates compliance checks.
    Replace with CV/OCR logic as needed.
    """

    def __init__(self, state_store: StateStore) -> None:
        self.state = state_store

    def _detect_asset_type(self, path: Path) -> AssetType:
        ext = path.suffix.lower()
        if ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}:
            return AssetType.image
        if ext in {".pdf", ".docx", ".doc", ".ppt", ".pptx", ".xlsx"}:
            return AssetType.document
        return AssetType.other

    def _list_uploaded_files(self, job_id: str) -> List[Path]:
        w = get_job_workspace(job_id)
        uploads = w["uploads"]
        if not uploads.exists():
            return []
        return [p for p in uploads.rglob("*") if p.is_file()]

    def _ensure_assets_list(self, job_id: str) -> List[Asset]:
        # Build assets pydantic models from uploaded files if they are not already registered.
        existing = {a.rel_path: a for a in self.state.list_assets(job_id)}
        files = self._list_uploaded_files(job_id)
        new_assets: List[Asset] = []
        for f in files:
            # Compute relative path under job dir
            w = get_job_workspace(job_id)
            rel_path = str(f.relative_to(w["job"]))
            if rel_path in existing:
                continue
            aid = str(uuid.uuid4())
            asset = Asset(
                id=aid,
                job_id=job_id,
                type=self._detect_asset_type(f),
                original_filename=f.name,
                rel_path=rel_path,
                status=AssetStatus.pending,
                meta={"size_bytes": f.stat().st_size},
            )
            new_assets.append(asset)
        if new_assets:
            self.state.add_assets(job_id, new_assets)
        return self.state.list_assets(job_id)

    def analyze_job(self, job_id: str) -> None:
        """Run analysis for all assets of a job.

        This mutates state: updates job status, marks assets as analyzed or failed,
        and appends stubbed issues.
        """
        job = self.state.get_job(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        self.state.update_job_status(job_id, JobStatus.analyzing)
        assets = self._ensure_assets_list(job_id)

        issues_to_add: List[Issue] = []
        for a in assets:
            try:
                # Simple heuristic: flag potential "old brand" if filename contains "old" or "legacy"
                if any(k in a.original_filename.lower() for k in ["old", "legacy"]):
                    iid = str(uuid.uuid4())
                    issues_to_add.append(
                        Issue(
                            id=iid,
                            job_id=job_id,
                            asset_id=a.id,
                            type=IssueType.old_logo,
                            severity=IssueSeverity.high,
                            message=f"Potential old brand detected in {a.original_filename}",
                            bbox=None,
                            score=0.95,
                            suggestions=["Replace with new brand asset."],
                            meta={"rule": "filename_contains_old_or_legacy"},
                        )
                    )
                # Another simple heuristic: if the file hash ends with '0', create a low severity color mismatch
                w = get_job_workspace(job_id)
                fpath = w["job"] / a.rel_path
                if fpath.exists():
                    h = hashlib.md5(fpath.read_bytes()).hexdigest()
                    if h.endswith("0"):
                        iid = str(uuid.uuid4())
                        issues_to_add.append(
                            Issue(
                                id=iid,
                                job_id=job_id,
                                asset_id=a.id,
                                type=IssueType.color_mismatch,
                                severity=IssueSeverity.low,
                                message="Potential color mismatch detected.",
                                bbox=None,
                                score=0.55,
                                suggestions=["Check color palette against brand guide."],
                                meta={"hash_tail": h[-4:]},
                            )
                        )
                self.state.update_asset_status(job_id, a.id, AssetStatus.analyzed)
            except Exception:
                # mark asset failure
                self.state.update_asset_status(job_id, a.id, AssetStatus.failed)

        if issues_to_add:
            self.state.append_issues(job_id, issues_to_add)

        # After analysis, move to preview generation (placeholder step)
        self.state.update_job_status(job_id, JobStatus.generating_previews)

    def generate_previews(self, job_id: str) -> None:
        """Stub preview generation by touching placeholder preview files."""
        w = get_job_workspace(job_id)
        previews: Path = w["previews"]
        previews.mkdir(parents=True, exist_ok=True)
        # Create a simple updated timestamp marker
        marker = previews / "PREVIEWS_READY.txt"
        marker.write_text(f"Previews generated at {datetime.utcnow().isoformat()}\n", encoding="utf-8")
