from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path

from src.models.asset import AssetStatus
from src.models.issue import IssueSeverity
from src.models.report import ReportSummary
from src.services.state_store import StateStore
from src.storage.workspace import get_job_workspace


class ReportService:
    """Creates summaries and downloadable artifacts for a job."""

    def __init__(self, state_store: StateStore) -> None:
        self.state = state_store

    def build_summary(self, job_id: str) -> ReportSummary:
        assets = self.state.list_assets(job_id)
        issues = self.state.list_issues(job_id)

        total = len(assets)
        analyzed = sum(1 for a in assets if a.status in (AssetStatus.analyzed, AssetStatus.fixed))
        failed = sum(1 for a in assets if a.status == AssetStatus.failed)
        total_issues = len(issues)
        high_or_above = sum(1 for i in issues if i.severity in (IssueSeverity.high, IssueSeverity.critical))

        summary = ReportSummary(
            job_id=job_id,
            total_assets=total,
            analyzed_assets=analyzed,
            failed_assets=failed,
            total_issues=total_issues,
            high_or_above_issues=high_or_above,
            meta={"generated_at": datetime.utcnow().isoformat()},
        )
        self.state.save_summary(job_id, summary)
        return summary

    def create_report_json(self, job_id: str) -> Path:
        """Create a JSON report file in report/ folder and return its path."""
        w = get_job_workspace(job_id)
        report_dir = w["report"]
        report_dir.mkdir(parents=True, exist_ok=True)

        summary = self.state.get_summary(job_id) or self.build_summary(job_id)
        details = {
            "summary": summary.model_dump(mode="json"),
            "assets": [a.model_dump(mode="json") for a in self.state.list_assets(job_id)],
            "issues": [i.model_dump(mode="json") for i in self.state.list_issues(job_id)],
        }

        report_path = report_dir / "report.json"
        report_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
        return report_path

    def create_outputs_zip(self, job_id: str) -> Path:
        """Zip the outputs folder content for download."""
        w = get_job_workspace(job_id)
        outputs = w["outputs"]
        outputs.mkdir(parents=True, exist_ok=True)
        zip_path = w["job"] / "outputs.zip"

        # Build zip
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # include outputs
            for p in outputs.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=str(p.relative_to(w["job"])))
            # include report if present
            report_json = w["report"] / "report.json"
            if report_json.exists():
                zf.write(report_json, arcname=str(report_json.relative_to(w["job"])))
        return zip_path

    def prepare_downloads(self, job_id: str, types: str) -> Path:
        """Prepare download artifacts based on requested type: zip|report|both.

        Returns a path to the primary artifact to be served.
        """
        types = types.lower()
        if types not in {"zip", "report", "both"}:
            raise ValueError("Invalid download type. Must be zip|report|both")

        # Always ensure summary/report exists
        report_json = self.create_report_json(job_id)

        if types == "report":
            return report_json

        # Create outputs.zip which also includes report.json for both/zip
        zip_path = self.create_outputs_zip(job_id)
        return zip_path
