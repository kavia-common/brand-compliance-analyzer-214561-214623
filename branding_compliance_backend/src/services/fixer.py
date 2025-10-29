from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List

from src.models.asset import AssetStatus
from src.models.issue import Issue, IssueStatus
from src.models.job import JobStatus
from src.services.state_store import StateStore
from src.storage.workspace import get_job_workspace
from src.services.vision import VisionUtils, DetectorConfig, Detection
from src.services.pdf_raster import rasterize_pdf
from src.services.pdf_assembler import assemble_pdf
import logging


class FixerService:
    """Applies automatic fixes to assets by replacing detected old logos with the new logo."""

    def __init__(self, state_store: StateStore) -> None:
        self.state = state_store

    def _load_detections_for_asset(self, job_id: str, asset_rel_path: str) -> List[Detection]:
        """Read detections from jobs/{id}/work/detections.json for a single asset path key."""
        w = get_job_workspace(job_id)
        det_path = w["job"] / "work" / "detections.json"
        if not det_path.exists():
            return []
        try:
            raw = json.loads(det_path.read_text(encoding="utf-8"))
            entries = raw.get(asset_rel_path, [])
            out: List[Detection] = []
            for e in entries:
                points = e.get("points") or []
                out.append(Detection(
                    x=float(e.get("x", 0.0)),
                    y=float(e.get("y", 0.0)),
                    width=float(e.get("width", 0.0)),
                    height=float(e.get("height", 0.0)),
                    score=float(e.get("score", 0.0)),
                    method=str(e.get("method", "template")),
                    points=[(float(p[0]), float(p[1])) for p in points] if points else None
                ))
            return out
        except Exception:
            return []

    def _choose_new_logo(self, job_id: str) -> Path | None:
        w = get_job_workspace(job_id)
        new_dir = w["analysis"] / "new_brand"
        if not new_dir.exists():
            return None
        files = [p for p in new_dir.iterdir() if p.is_file()]
        return files[0] if files else None

    def fix_asset(self, job_id: str, asset_id: str) -> Path:
        """Fix a single asset and place result in outputs/ using detections + new logo.

        For PDFs:
          - Rasterize pages (if not already), apply replacements per page,
            and reassemble a fixed.pdf under outputs/.
        """
        w = get_job_workspace(job_id)
        job = self.state.get_job(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        assets = self.state.list_assets(job_id)
        asset = next((a for a in assets if a.id == asset_id), None)
        if asset is None:
            raise ValueError(f"Asset {asset_id} not found")

        src = w["job"] / asset.rel_path
        outputs = w["outputs"]
        outputs.mkdir(parents=True, exist_ok=True)

        # Select new logo
        new_logo = self._choose_new_logo(job_id)
        cfg = DetectorConfig()

        # Handle PDFs
        if src.suffix.lower() == ".pdf":
            log = logging.getLogger("fixer.pdf")
            # Directories for pdf workflow
            pdf_root = w["job"] / "pdf"
            pages_dir = pdf_root / "pages"
            fixed_pages_dir = pdf_root / "fixed_pages"
            final_dir = pdf_root / "final"
            fixed_pages_dir.mkdir(parents=True, exist_ok=True)
            final_dir.mkdir(parents=True, exist_ok=True)

            # Rasterize original pages
            pages, page_sizes = rasterize_pdf(src, pages_dir, dpi=300)
            log.info("fix:pdf_rasterized asset=%s pages=%d", asset.id, len(pages))

            # Apply replacements per page using detections keyed as "<rel_path>::page:<idx>"
            page_image_paths: List[Path] = []
            for page in pages:
                key = f"{asset.rel_path}::page:{page.index}"
                dets = self._load_detections_for_asset(job_id, key)
                src_img = page.image_path
                out_img = fixed_pages_dir / f"{page.index:04d}.png"
                if dets and new_logo and new_logo.exists():
                    log.info("fix:apply_page asset=%s page=%d dets=%d", asset.id, page.index, len(dets))
                    VisionUtils.replace_logo(src_img, new_logo, dets, out_img, feather=6, quality_mode=cfg.quality)
                else:
                    log.info("fix:copy_page asset=%s page=%d dets=0", asset.id, page.index)
                    # copy the page image through to maintain pipeline
                    try:
                        shutil.copy2(src_img, out_img)
                    except Exception:
                        out_img = src_img
                page_image_paths.append(out_img if out_img.exists() else src_img)

            # Reassemble into a fixed PDF
            fixed_pdf = final_dir / "fixed.pdf"
            assemble_pdf(page_image_paths, fixed_pdf, page_sizes)
            log.info("fix:assembled_pdf asset=%s pages=%d out=%s", asset.id, len(page_image_paths), fixed_pdf)

            # Update issues status for this asset
            issues = self.state.list_issues(job_id)
            updated: List[Issue] = []
            for i in issues:
                if i.asset_id == asset_id and i.status != IssueStatus.fixed:
                    i.status = IssueStatus.fixed
                updated.append(i)
            self.state._save_issues(job_id, updated)

            self.state.update_asset_status(job_id, asset_id, AssetStatus.fixed)
            if job.status not in (JobStatus.fixing, JobStatus.summarizing, JobStatus.completed):
                self.state.update_job_status(job_id, JobStatus.fixing)

            # Also place a copy in outputs for convenience
            dst = outputs / f"fixed_{Path(asset.original_filename).stem}.pdf"
            try:
                shutil.copy2(fixed_pdf, dst)
            except Exception:
                dst = fixed_pdf
            return dst

        # Default image behavior
        dst = outputs / f"fixed_{asset.original_filename}"
        dets = self._load_detections_for_asset(job_id, asset.rel_path)
        if src.exists() and dets and new_logo and new_logo.exists():
            VisionUtils.replace_logo(src, new_logo, dets, dst, feather=6, quality_mode=cfg.quality)
        else:
            if src.exists():
                shutil.copy2(src, dst)

        # mark issues for this asset as fixed
        issues = self.state.list_issues(job_id)
        updated: List[Issue] = []
        for i in issues:
            if i.asset_id == asset_id and i.status != IssueStatus.fixed:
                i.status = IssueStatus.fixed
            updated.append(i)
        self.state._save_issues(job_id, updated)

        self.state.update_asset_status(job_id, asset_id, AssetStatus.fixed)
        if job.status not in (JobStatus.fixing, JobStatus.summarizing, JobStatus.completed):
            self.state.update_job_status(job_id, JobStatus.fixing)

        return dst

    def fix_all(self, job_id: str) -> List[Path]:
        """Fix all assets that have open issues using stored detections."""
        w = get_job_workspace(job_id)
        outputs = w["outputs"]
        outputs.mkdir(parents=True, exist_ok=True)

        paths: List[Path] = []
        issues = self.state.list_issues(job_id)
        assets = self.state.list_assets(job_id)
        by_asset = {a.id: a for a in assets}
        affected_assets = {i.asset_id for i in issues if i.status != IssueStatus.fixed}

        for aid in affected_assets:
            a = by_asset.get(aid)
            if not a:
                continue
            p = self.fix_asset(job_id, aid)
            paths.append(p)

        # finalize status progression
        self.state.update_job_status(job_id, JobStatus.summarizing)
        return paths
