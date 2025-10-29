from __future__ import annotations

import io
import os
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any

import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image

from src.models.schemas import BoundingBox, Findings, PageFindings, StartRequestParams, StatusResponse
from src.services.image_utils import (
    pil_to_cv,
    cv_to_png_bytes,
    multi_scale_template_match,
    draw_boxes,
    Detection,
    feature_match_fallback,
)
from src.services.detect_config import default_detection_config, DetectionConfig
from src.services.json_utils import to_native_jsonable
from src.storage.paths import (
    ensure_job_dirs,
    input_pdfs_dir,
    input_logos_old_dir,
    input_logos_new_dir,
    work_pages_dir,
    work_overlays_dir,
    output_replaced_dir,
    save_metadata,
    load_metadata_safely,
)

# In-memory registry of job statuses (metadata persisted on disk)
_JOBS_LOCK = threading.Lock()
_JOBS: Dict[str, StatusResponse] = {}


def _update_status(job_id: str, **kwargs) -> None:
    with _JOBS_LOCK:
        s = _JOBS.get(job_id)
        if not s:
            s = StatusResponse(job_id=job_id, status="queued", progress=0.0)
        for k, v in kwargs.items():
            setattr(s, k, v)
        _JOBS[job_id] = s


def _save_uploaded_file(upload, dest_path: str) -> None:
    with open(dest_path, "wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    upload.file.seek(0)


def _encode_pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _rasterize_pdf_to_images(pdf_path: str, dpi: int, max_pages: Optional[int]) -> List[Image.Image]:
    """
    Rasterize PDF pages into PIL images at the given DPI.
    Ensures consistent RGB colorspace, sufficient DPI (enforced minimum 200),
    and enables antialiasing for vector content.
    """
    dpi = max(200, min(600, int(dpi or 250)))
    images: List[Image.Image] = []
    with fitz.open(pdf_path) as doc:
        page_count = len(doc)
        to_process = page_count if max_pages is None else min(page_count, max_pages)
        for i in range(to_process):
            page = doc[i]
            zoom = dpi / 72.0
            # Use matrix that ensures antialias for vector drawing (default with get_pixmap)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    return images


def _overlay_logo_on_page(page: fitz.Page, new_logo_img: Image.Image, boxes: List[Detection], dpi: int):
    """
    Overlay new_logo onto page within detected boxes:
    - Draw white rectangle
    - Draw image scaled to box size
    """
    if not boxes:
        return
    # Convert PIL to bytes for PyMuPDF insertion
    logo_bytes = _encode_pil_to_png_bytes(new_logo_img.convert("RGBA"))

    inv_scale = 72.0 / dpi  # convert raster pixels back to PDF points
    for d in boxes:
        x0 = d.x * inv_scale
        y0 = d.y * inv_scale
        x1 = (d.x + d.w) * inv_scale
        y1 = (d.y + d.h) * inv_scale

        rect = fitz.Rect(x0, y0, x1, y1)
        page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)
        page.insert_image(rect, stream=logo_bytes, keep_proportion=False, overlay=True)


def _save_page_preview_images(job_id: str, page_idx: int, page_img: Image.Image, detections: List[Detection], new_logo: Image.Image):
    """Generate and save detected and replaced preview PNGs and the raw page image for debugging."""
    pages_dir = work_pages_dir(job_id)
    overlays_dir = work_overlays_dir(job_id)
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)

    # Save raw page render for diagnostics
    raw_path = os.path.join(pages_dir, f"page_{page_idx:04d}_raw.png")
    page_img.save(raw_path, format="PNG")

    # Save detected overlay preview
    bgr = pil_to_cv(page_img)
    det_vis = draw_boxes(bgr, detections, color=(0, 0, 255), thickness=2)
    det_bytes = cv_to_png_bytes(det_vis)
    with open(os.path.join(pages_dir, f"page_{page_idx:04d}_detected.png"), "wb") as f:
        f.write(det_bytes)

    # Save replaced preview by compositing new_logo onto image (visual only)
    replaced_bgr = bgr.copy()
    logo_bgr = pil_to_cv(new_logo)
    for d in detections:
        if d.w <= 0 or d.h <= 0:
            continue
        resized = cv2.resize(logo_bgr, (d.w, d.h), interpolation=cv2.INTER_AREA)
        x0, y0, w, h = d.x, d.y, d.w, d.h
        H, W = replaced_bgr.shape[:2]
        x0 = max(0, min(x0, W - 1))
        y0 = max(0, min(y0, H - 1))
        w = max(1, min(w, W - x0))
        h = max(1, min(h, H - y0))
        replaced_bgr[y0 : y0 + h, x0 : x0 + w] = 255
        replaced_bgr[y0 : y0 + h, x0 : x0 + w] = resized

    rep_bytes = cv_to_png_bytes(replaced_bgr)
    with open(os.path.join(overlays_dir, f"page_{page_idx:04d}_replaced.png"), "wb") as f:
        f.write(rep_bytes)


def _compute_simple_metrics(detections_count: int, prev_detections_count: Optional[int]) -> Dict[str, Any]:
    """Compute approximate recall/precision deltas using counts only (no GT available)."""
    # Without ground truth, we report before/after counts as a proxy.
    if prev_detections_count is None:
        return {"before": None, "after": detections_count, "delta": None}
    return {
        "before": prev_detections_count,
        "after": detections_count,
        "delta": detections_count - prev_detections_count,
    }


def _write_output_pdf(job_id: str, pdf_in_path: str, detections_by_page: Dict[int, List[Detection]], new_logo_img: Image.Image, dpi: int) -> str:
    """Open original PDF, overlay logos on matching boxes, and write replaced PDF."""
    out_dir = output_replaced_dir(job_id)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_in_path))[0]
    out_path = os.path.join(out_dir, f"{base}_replaced.pdf")

    with fitz.open(pdf_in_path) as doc:
        for i, page in enumerate(doc):
            boxes = detections_by_page.get(i, [])
            if boxes:
                _overlay_logo_on_page(page, new_logo_img, boxes, dpi)
        doc.save(out_path)

    return out_path


def _process_job(job_id: str, pdf_path: str, old_logo_paths: List[str], new_logo_path: str, params: StartRequestParams):
    """Worker function that executes in a separate thread."""
    try:
        _update_status(job_id, status="processing", progress=0.01, message="Starting")

        # Load new logo
        new_logo_img = Image.open(new_logo_path).convert("RGBA")

        # Rasterize PDF
        _update_status(job_id, message="Rasterizing PDF")
        raster_pages = _rasterize_pdf_to_images(pdf_path, dpi=params.dpi, max_pages=params.max_pages)
        total_pages = len(raster_pages)
        if total_pages == 0:
            raise RuntimeError("No pages to process.")

        # Prepare templates (normalize and save debug copies)
        templates_bgr: List[np.ndarray] = []
        loaded_templates_info: List[Dict[str, str]] = []
        for idx, p in enumerate(old_logo_paths):
            try:
                tpl_img = Image.open(p).convert("RGBA")
                tpl_bgr = pil_to_cv(tpl_img)
                templates_bgr.append(tpl_bgr)
                # write debug copy after normalization
                dbg_tpl_dir = work_pages_dir(job_id)
                os.makedirs(dbg_tpl_dir, exist_ok=True)
                with open(os.path.join(dbg_tpl_dir, f"template_{idx:02d}.png"), "wb") as f:
                    f.write(cv_to_png_bytes(tpl_bgr))
                loaded_templates_info.append({"path": p, "status": "loaded"})
            except Exception as ex:
                loaded_templates_info.append({"path": p, "status": f"failed: {ex}"})
                continue
        if not templates_bgr:
            raise RuntimeError("No valid old_logo templates could be loaded.")

        # Setup detection configuration (tunable)
        cfg: DetectionConfig = default_detection_config()

        findings_pages: List[PageFindings] = []
        detections_by_page: Dict[int, List[Detection]] = {}

        # Process each page
        per_page_counts: List[int] = []
        all_detections_dump: Dict[int, List[Dict[str, float]]] = {}
        page_diagnostics: Dict[int, Dict[str, Any]] = {}
        for idx, page_img in enumerate(raster_pages):
            _update_status(job_id, message=f"Detecting logos on page {idx + 1}/{total_pages}")
            page_bgr = pil_to_cv(page_img)

            diag: Dict[str, Any] = {}
            # Primary: multi-scale template matching with rotation search and scale prior
            dets = multi_scale_template_match(
                page_bgr,
                templates_bgr,
                match_threshold=float(params.match_threshold or cfg.match_threshold),
                scales=cfg.scales,
                method=cv2.TM_CCOEFF_NORMED,
                cfg=cfg,
                page_diag=diag,
            )

            # Fallback if zero detections: feature matching with small rotations
            if len(dets) == 0:
                fm_dets = feature_match_fallback(
                    page_bgr,
                    templates_bgr,
                    rotations_deg=cfg.feature_rotations,
                    min_inliers=cfg.feature_min_inliers,
                    ransac_reproj_thresh=cfg.feature_ransac_reproj_thresh,
                    cfg=cfg,
                    page_diag=diag,
                )
                if fm_dets:
                    dets = fm_dets
                else:
                    # Add failure reasons if clearly problematic (heuristics)
                    if "reasons" not in diag:
                        diag["reasons"] = []
                    # Low-contrast heuristic: compute contrast
                    g = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
                    if float(g.std()) < 18.0:
                        diag["reasons"].append("low_contrast_page")
                    # Rotation heuristic (no matches but strong edges): suggest increasing rotation
                    edges = cv2.Canny(g, 50, 150)
                    if edges.mean() > 20:
                        diag["reasons"].append("possible_rotation>15deg")

            detections_by_page[idx] = dets
            per_page_counts.append(len(dets))

            # Save previews and raw page
            _save_page_preview_images(job_id, idx, page_img, dets, new_logo_img)

            # Collect findings and dump per-box metadata
            page_boxes = []
            all_detections_dump[idx] = []
            for d in dets:
                page_boxes.append(BoundingBox(x=d.x, y=d.y, w=d.w, h=d.h, score=d.score, template_id=d.template_id))
                all_detections_dump[idx].append(
                    {"x": d.x, "y": d.y, "w": d.w, "h": d.h, "score": float(d.score), "template_id": d.template_id}
                )
            findings_pages.append(PageFindings(page_index=idx, detections=page_boxes))
            if cfg.collect_page_diagnostics:
                page_diagnostics[idx] = diag

            # Update progress
            _update_status(job_id, progress=(idx + 1) / max(1, total_pages))

        # Write replaced PDF
        _update_status(job_id, message="Writing replaced PDF")
        out_pdf = _write_output_pdf(job_id, pdf_path, detections_by_page, new_logo_img, dpi=max(200, params.dpi))

        # Try to compute an approximate before/after improvement if prior metadata exists for same input path
        prev_total = None
        try:
            # scan sibling job directories for same input file path to compare counts (best-effort)
            jobs_root = os.path.join(os.getcwd(), "storage", "jobs")
            if os.path.isdir(jobs_root):
                for d in os.listdir(jobs_root):
                    md = load_metadata_safely(d)
                    if not md or d == job_id:
                        continue
                    if md.get("input_pdf") == pdf_path and isinstance(md.get("findings"), dict):
                        prev_total = int(md["findings"].get("total_detections"))  # type: ignore
        except Exception:
            prev_total = None

        # Save metadata
        total_now = int(sum(per_page_counts))
        meta = {
            "job_id": job_id,
            "status": "completed",
            "progress": 1.0,
            "message": "Completed",
            "input_pdf": pdf_path,
            "output_pdf": out_pdf,
            "outputs": [
                {"type": "pdf", "path": out_pdf, "label": "replaced"}
            ],
            "params": params.model_dump(),
            "detection_config": default_detection_config().to_dict(),  # persisted for transparency
            "findings": {
                "total_pages": total_pages,
                "pages": [fp.model_dump() for fp in findings_pages],
                "per_page_detection_counts": per_page_counts,
                "total_detections": total_now,
                "detections_dump": all_detections_dump,  # per-page detailed detections
                "page_diagnostics": page_diagnostics,    # per-page failure reasons and search settings
                "approx_improvement": _compute_simple_metrics(total_now, prev_total),
            },
            "debug": {
                "templates": loaded_templates_info,
                "notes": "Raw page PNGs stored as page_XXXX_raw.png; normalized templates stored as template_XX.png in work/pages.",
            },
            "timestamps": {"completed": time.time()},
        }
        # Normalize to native types prior to saving
        meta = to_native_jsonable(meta)
        save_metadata(job_id, meta)

        # Persist a small before/after report in job root for convenience
        try:
            from src.services.report_utils import persist_job_report
            persist_job_report(job_id)
        except Exception:
            pass

        # Update in-memory status
        _update_status(
            job_id,
            status="completed",
            progress=1.0,
            message="Completed",
        )

    except Exception as e:
        # Persist failure to metadata
        meta = load_metadata_safely(job_id) or {}
        meta.update(
            {
                "job_id": job_id,
                "status": "failed",
                "message": str(e),
                "timestamps": {**meta.get("timestamps", {}), "failed": time.time()},
            }
        )
        meta = to_native_jsonable(meta)
        save_metadata(job_id, meta)
        _update_status(job_id, status="failed", message=str(e))


class PdfLogoReplaceManager:
    """Coordinator for job creation, status retrieval, previews, and download."""

    # PUBLIC_INTERFACE
    @staticmethod
    def create_job_and_start(pdf_file, old_logo_files, new_logo_file, params: StartRequestParams) -> str:
        """Create job, persist inputs, and spawn background processing thread."""
        job_id = str(uuid.uuid4())
        ensure_job_dirs(job_id)

        # Save inputs
        in_pdf_dir = input_pdfs_dir(job_id)
        in_old_dir = input_logos_old_dir(job_id)
        in_new_dir = input_logos_new_dir(job_id)

        os.makedirs(in_pdf_dir, exist_ok=True)
        os.makedirs(in_old_dir, exist_ok=True)
        os.makedirs(in_new_dir, exist_ok=True)

        pdf_path = os.path.join(in_pdf_dir, pdf_file.filename)
        _save_uploaded_file(pdf_file, pdf_path)

        old_logo_paths: List[str] = []
        for i, f in enumerate(old_logo_files):
            ext = os.path.splitext(f.filename or f"old_{i}.png")[1] or ".png"
            dest = os.path.join(in_old_dir, f"old_{i}{ext}")
            _save_uploaded_file(f, dest)
            old_logo_paths.append(dest)

        new_ext = os.path.splitext(new_logo_file.filename or "new_logo.png")[1] or ".png"
        new_logo_path = os.path.join(in_new_dir, f"new{new_ext}")
        _save_uploaded_file(new_logo_file, new_logo_path)

        # Initialize metadata
        meta = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "message": "Queued",
            "input_pdf": pdf_path,
            "old_logos": old_logo_paths,
            "new_logo": new_logo_path,
            "params": params.model_dump(),
            "timestamps": {"created": time.time()},
            "outputs": [],
        }
        meta = to_native_jsonable(meta)
        save_metadata(job_id, meta)

        # Register and spawn worker thread
        _update_status(job_id, status="queued", progress=0.0, message="Queued")
        t = threading.Thread(
            target=_process_job,
            kwargs={
                "job_id": job_id,
                "pdf_path": pdf_path,
                "old_logo_paths": old_logo_paths,
                "new_logo_path": new_logo_path,
                "params": params,
            },
            daemon=True,
        )
        t.start()
        return job_id

    # PUBLIC_INTERFACE
    @staticmethod
    def get_status(job_id: str) -> Optional[StatusResponse]:
        """Return current job status, merging with on-disk metadata if needed."""
        with _JOBS_LOCK:
            s = _JOBS.get(job_id)
        md = load_metadata_safely(job_id)
        if md:
            findings = None
            if "findings" in md:
                try:
                    pages = [
                        PageFindings(
                            page_index=p["page_index"],
                            detections=[BoundingBox(**bb) for bb in p.get("detections", [])],
                        )
                        for p in md["findings"].get("pages", [])
                    ]
                    findings = Findings(total_pages=md["findings"]["total_pages"], pages=pages)
                except Exception:
                    findings = None
            status = md.get("status", "queued")
            progress = md.get("progress", s.progress if s else 0.0)
            message = md.get("message", None)
            return StatusResponse(job_id=job_id, status=status, progress=progress, message=message, findings=findings)
        return s

    # PUBLIC_INTERFACE
    @staticmethod
    def get_preview(job_id: str, page_index: int, preview_type: str) -> Optional[bytes]:
        """Return PNG bytes for the requested preview."""
        if preview_type == "detected":
            path = os.path.join(work_pages_dir(job_id), f"page_{page_index:04d}_detected.png")
        else:
            path = os.path.join(work_overlays_dir(job_id), f"page_{page_index:04d}_replaced.png")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()

    # PUBLIC_INTERFACE
    @staticmethod
    def confirm(job_id: str) -> bool:
        """Stub confirmation: mark metadata as confirmed if job completed."""
        md = load_metadata_safely(job_id)
        if not md:
            return False
        if md.get("status") != "completed":
            return False
        md["confirmed"] = True
        md = to_native_jsonable(md)
        save_metadata(job_id, md)
        _update_status(job_id, message="Confirmed")
        return True

    # PUBLIC_INTERFACE
    @staticmethod
    def get_download(job_id: str) -> Optional[Tuple[str, bytes]]:
        """Return filename and bytes for the replaced PDF."""
        md = load_metadata_safely(job_id)
        if not md:
            return None
        outputs = md.get("outputs") or []
        out_pdf = md.get("output_pdf")
        path: Optional[str] = None
        if outputs and isinstance(outputs, list):
            for item in outputs:
                if isinstance(item, dict) and item.get("type") == "pdf" and item.get("path"):
                    cand = item["path"]
                    if os.path.exists(cand):
                        path = cand
                        break
        if path is None:
            path = out_pdf
        if not path or not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            data = f.read()
        filename = os.path.basename(path)
        return filename, data

    # PUBLIC_INTERFACE
    @staticmethod
    def get_all_output_pdfs(job_id: str) -> list[str]:
        """Return a list of file paths to all generated replaced PDFs for the job."""
        md = load_metadata_safely(job_id)
        if not md:
            return []
        paths: list[str] = []
        outputs = md.get("outputs") or []
        for item in outputs:
            if isinstance(item, dict) and item.get("type") == "pdf" and item.get("path"):
                p = item["path"]
                if os.path.exists(p):
                    paths.append(p)
        out_pdf = md.get("output_pdf")
        if out_pdf and os.path.exists(out_pdf):
            if out_pdf not in paths:
                paths.append(out_pdf)
        return paths
