from __future__ import annotations

import hashlib
import uuid
import json
import shutil
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

    def _overlay_preview(
        self,
        job_id: str,
        asset_path: Path,
        detections: List[Detection],
        page_index: int | None = None,
        out_stem: str | None = None,
    ) -> None:
        """Create a simple overlay PNG drawing rectangles on detections.

        If page_index is provided, include it in the overlay filename to disambiguate per-page overlays.
        out_stem optionally overrides the base stem used in the overlay filename; this allows generating
        overlays for job-level page indices even if the source image resides elsewhere.
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
                stem = out_stem if out_stem else Path(asset_path.name).stem
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

                elif a.type == AssetType.pdf and fpath.exists():
                    # Rasterize PDF pages under a per-asset src directory, then copy to a global pages dir with unique global indices.
                    pdf_root = w["job"] / "pdf"
                    pages_global_dir = pdf_root / "pages"
                    pages_global_dir.mkdir(parents=True, exist_ok=True)
                    src_pages_dir = pdf_root / "src_pages" / a.id
                    src_pages_dir.mkdir(parents=True, exist_ok=True)
                    overlays_dir = pdf_root / "overlays"
                    overlays_dir.mkdir(parents=True, exist_ok=True)

                    pages, _sizes = rasterize_pdf(fpath, src_pages_dir, dpi=300)
                    log.info("analyze:pdf_rasterized rel_path=%s page_count=%d src_dir=%s", a.rel_path, len(pages), src_pages_dir)
                    if not pages:
                        log.warning("analyze:pdf_no_pages rel_path=%s", a.rel_path)

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

                    # Load or initialize page_map dict
                    page_map_path = pdf_root / "page_map.json"
                    try:
                        page_map: dict = json.loads(page_map_path.read_text(encoding="utf-8")) if page_map_path.exists() else {}
                        if not isinstance(page_map, dict):
                            page_map = {}
                    except Exception:
                        page_map = {}
                    # Compute next global index
                    try:
                        keys = [int(str(k)) for k in page_map.keys() if str(k).isdigit()]
                        next_global = max(keys) + 1 if keys else 0
                    except Exception:
                        next_global = 0

                    # Process each local page, detect, copy to global, overlay with global naming, and record page_map
                    for page in pages:
                        global_idx = next_global
                        next_global += 1

                        # Copy local page image to global pages directory with global index filename
                        global_img_path = pages_global_dir / f"{global_idx:04d}.png"
                        try:
                            shutil.copy2(page.image_path, global_img_path)
                        except Exception:
                            # fallback to rename if copy fails
                            try:
                                page.image_path.replace(global_img_path)
                            except Exception:
                                global_img_path = page.image_path  # last resort

                        dets: List[Detection] = []
                        if old_logo_template:
                            # Run detection on the source (identical size to global copy)
                            dets = VisionUtils.detect_old_logo(page.image_path, old_logo_template, cfg)
                            # Key detections by local page index so fixer can map via page_map later
                            key = f"{a.rel_path}::page:{page.index}"
                            detections_map[key] = dets
                            log.info("analyze:pdf_page_detections asset=%s local_page=%d global_page=%d count=%d",
                                     a.id, page.index, global_idx, len(dets))
                            # Save overlay using the global index naming, composited on the global image path
                            self._overlay_preview(job_id, global_img_path, dets, page_index=global_idx, out_stem=f"{global_idx:04d}")
                            # Create issues with local page info
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

                        # Record page_map for global -> asset/local mapping
                        page_map[str(global_idx)] = {
                            "asset_id": a.id,
                            "asset_rel_path": a.rel_path,
                            "local_index": int(page.index),
                        }

                    # Save page_map back
                    try:
                        page_map_path.parent.mkdir(parents=True, exist_ok=True)
                        page_map_path.write_text(json.dumps(page_map, indent=2), encoding="utf-8")
                    except Exception:
                        # non-fatal
                        pass

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
