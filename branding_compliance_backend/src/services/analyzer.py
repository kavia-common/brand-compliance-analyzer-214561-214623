from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from src.models.asset import Asset, AssetStatus, AssetType, BoundingBox
from src.models.issue import Issue, IssueSeverity, IssueType
from src.services.pdf_raster import rasterize_pdf
from src.models.job import JobStatus
from src.services.state_store import StateStore
from src.storage.workspace import get_job_workspace
from src.services.vision import VisionUtils, DetectorConfig, Detection
import logging


class AnalyzerService:
    """Performs analysis for a job's uploaded assets.

    Uses OpenCV/Pillow-based detection utilities to find old logos, persists detections,
    and generates lightweight overlay previews.
    """

    def __init__(self, state_store: StateStore) -> None:
        self.state = state_store

    def _detect_asset_type(self, path: Path) -> AssetType:
        ext = path.suffix.lower()
        if ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}:
            return AssetType.image
        if ext in {".pdf"}:
            return AssetType.pdf
        if ext in {".docx", ".doc", ".ppt", ".pptx", ".xlsx"}:
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

    def _choose_old_logo_template(self, job_id: str) -> Path | None:
        """Pick the first uploaded old brand image as template."""
        w = get_job_workspace(job_id)
        old_dir = w["analysis"] / "old_brand"
        if not old_dir.exists():
            return None
        cands = [p for p in old_dir.iterdir() if p.is_file()]
        return cands[0] if cands else None

    def _save_detections(self, job_id: str, det_map: Dict[str, List[Detection]]) -> Path:
        w = get_job_workspace(job_id)
        work_dir = w["job"] / "work"
        out = work_dir / "detections.json"
        VisionUtils.save_detections_json(det_map, out)
        return out

    def _overlay_preview(self, job_id: str, asset_path: Path, detections: List[Detection], page_index: int | None = None) -> None:
        """Create a simple overlay PNG drawing rectangles on detections.

        If page_index is provided, include it in the overlay filename to disambiguate per-page overlays.
        """
        try:
            from PIL import Image, ImageDraw
        except Exception:
            return
        try:
            with Image.open(asset_path) as im:
                im = im.convert("RGBA")
                overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                for d in detections:
                    x1, y1 = d.x, d.y
                    x2, y2 = d.x + d.width, d.y + d.height
                    # red rectangle with alpha
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=4)
                    # confidence text
                    draw.text((x1 + 2, max(0, y1 - 12)), f"{d.method}:{d.score:.2f}", fill=(255, 0, 0, 255))
                out = Image.alpha_composite(im, overlay)
                w = get_job_workspace(job_id)
                previews: Path = w["previews"]
                previews.mkdir(parents=True, exist_ok=True)
                stem = Path(asset_path.name).stem
                suffix = f"_p{page_index:04d}" if page_index is not None else ""
                name = f"overlay_{stem}{suffix}.png"
                out.save(previews / name, format="PNG")
        except Exception:
            # best-effort only
            return

    def analyze_job(self, job_id: str) -> None:
        """Run analysis for all assets of a job using CV-based detection and persist results."""
        job = self.state.get_job(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        self.state.update_job_status(job_id, JobStatus.analyzing)
        assets = self._ensure_assets_list(job_id)

        old_logo_template = self._choose_old_logo_template(job_id)
        cfg = DetectorConfig()
        log = logging.getLogger("analyzer.run")
        log.info("analyze:job_start job_id=%s old_logo=%s detector=%s quality=%s", job_id, old_logo_template, cfg.detector, cfg.quality)

        detections_map: Dict[str, List[Detection]] = {}
        issues_to_add: List[Issue] = []

        for a in assets:
            try:
                w = get_job_workspace(job_id)
                fpath = w["job"] / a.rel_path

                if a.type == AssetType.image and old_logo_template and fpath.exists():
                    dets = VisionUtils.detect_old_logo(fpath, old_logo_template, cfg)
                    detections_map[a.rel_path] = dets
                    log.info("analyze:image_detections rel_path=%s count=%d", a.rel_path, len(dets))
                    self._overlay_preview(job_id, fpath, dets)
                    for d in dets:
                        iid = str(uuid.uuid4())
                        bbox = BoundingBox(x=d.x, y=d.y, width=d.width, height=d.height, normalized=False)
                        issues_to_add.append(
                            Issue(
                                id=iid,
                                job_id=job_id,
                                asset_id=a.id,
                                type=IssueType.old_logo,
                                severity=IssueSeverity.high if d.score >= 0.8 else IssueSeverity.medium,
                                message=f"Old logo detected ({d.method})",
                                bbox=bbox,
                                score=d.score,
                                suggestions=["Replace with new brand asset."],
                                meta={"method": d.method, "points": d.points or []},
                            )
                        )

                elif a.type == AssetType.pdf and fpath.exists() and old_logo_template:
                    # Rasterize PDF pages to images and run detection per page
                    pdf_pages_dir = w["job"] / "pdf" / "pages"
                    pdf_overlays_dir = w["job"] / "pdf" / "overlays"
                    pdf_pages_dir.mkdir(parents=True, exist_ok=True)
                    pdf_overlays_dir.mkdir(parents=True, exist_ok=True)
                    pages, _sizes = rasterize_pdf(fpath, pdf_pages_dir, dpi=300)
                    log.info("analyze:pdf_rasterized rel_path=%s page_count=%d out_dir=%s", a.rel_path, len(pages), pdf_pages_dir)

                    # update asset page_count and persist immediately
                    try:
                        current_assets = self.state.list_assets(job_id)
                        for asset in current_assets:
                            if asset.id == a.id:
                                asset.page_count = len(pages)
                        self.state._save_assets(job_id, current_assets)
                        log.info("analyze:asset_page_count_set asset_id=%s pages=%d", a.id, len(pages))
                    except Exception as e:
                        log.warning("analyze:page_count_save_failed asset_id=%s err=%s", a.id, e)

                    for page in pages:
                        dets = VisionUtils.detect_old_logo(page.image_path, old_logo_template, cfg)
                        # detections map key per-page for downstream fixer
                        key = f"{a.rel_path}::page:{page.index}"
                        detections_map[key] = dets
                        log.info("analyze:pdf_page_detections rel_path=%s page=%d count=%d", a.rel_path, page.index, len(dets))
                        # Save per-page overlay to previews with page index in filename
                        self._overlay_preview(job_id, page.image_path, dets, page_index=page.index)
                        for d in dets:
                            iid = str(uuid.uuid4())
                            bbox = BoundingBox(x=d.x, y=d.y, width=d.width, height=d.height, normalized=False)
                            issues_to_add.append(
                                Issue(
                                    id=iid,
                                    job_id=job_id,
                                    asset_id=a.id,
                                    type=IssueType.old_logo,
                                    severity=IssueSeverity.high if d.score >= 0.8 else IssueSeverity.medium,
                                    message=f"Old logo detected on page {page.index} ({d.method})",
                                    bbox=bbox,
                                    score=d.score,
                                    suggestions=["Replace with new brand asset."],
                                    page_number=page.index,
                                    meta={"method": d.method, "points": d.points or [], "page_index": page.index},
                                )
                            )
                else:
                    # Keep heuristic to still demonstrate non-logo issues
                    h = None
                    if fpath.exists():
                        h = hashlib.md5(fpath.read_bytes()).hexdigest()
                    if h and h.endswith("0"):
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
                # mark analyzed
                self.state.update_asset_status(job_id, a.id, AssetStatus.analyzed)
            except Exception:
                # mark asset failure
                self.state.update_asset_status(job_id, a.id, AssetStatus.failed)

        # persist detections JSON
        if detections_map:
            self._save_detections(job_id, detections_map)

        if issues_to_add:
            self.state.append_issues(job_id, issues_to_add)

        # After analysis, move to preview generation (marker)
        self.state.update_job_status(job_id, JobStatus.generating_previews)

    def generate_previews(self, job_id: str) -> None:
        """Mark previews ready by writing a timestamp marker."""
        w = get_job_workspace(job_id)
        previews: Path = w["previews"]
        previews.mkdir(parents=True, exist_ok=True)
        marker = previews / "PREVIEWS_READY.txt"
        marker.write_text(f"Previews generated at {datetime.utcnow().isoformat()}\n", encoding="utf-8")
