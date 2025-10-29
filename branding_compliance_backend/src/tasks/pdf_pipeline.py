from __future__ import annotations

import io
import os
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image

from src.models.schemas import BoundingBox, Findings, PageFindings, StartRequestParams, StatusResponse
from src.services.image_utils import pil_to_cv, cv_to_png_bytes, multi_scale_template_match, draw_boxes, Detection
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
        # UploadFile exposes .file for SpooledTemporaryFile
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
    """Rasterize PDF pages into PIL images at the given DPI."""
    images: List[Image.Image] = []
    with fitz.open(pdf_path) as doc:
        page_count = len(doc)
        to_process = page_count if max_pages is None else min(page_count, max_pages)
        for i in range(to_process):
            page = doc[i]
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    return images


def _overlay_logo_on_page(page: fitz.Page, new_logo_img: Image.Image, boxes: List[Detection], dpi: int):
    """
    Overlay new_logo onto page within detected boxes. For MVP:
    - Draw white rectangle
    - Draw image scaled to box size
    """
    if not boxes:
        return
    # Convert PIL to bytes for PyMuPDF insertion
    logo_bytes = _encode_pil_to_png_bytes(new_logo_img.convert("RGBA"))

    inv_scale = 72.0 / dpi  # convert raster pixels back to PDF points
    for d in boxes:
        # Bounding box in PDF points
        x0 = d.x * inv_scale
        y0 = d.y * inv_scale
        x1 = (d.x + d.w) * inv_scale
        y1 = (d.y + d.h) * inv_scale

        rect = fitz.Rect(x0, y0, x1, y1)
        # Paint white rectangle as background to avoid bleed-through
        page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)
        # Draw image in the rect
        page.insert_image(rect, stream=logo_bytes, keep_proportion=False, overlay=True)


def _save_page_preview_images(job_id: str, page_idx: int, page_img: Image.Image, detections: List[Detection], new_logo: Image.Image):
    """Generate and save detected and replaced preview PNGs."""
    pages_dir = work_pages_dir(job_id)
    overlays_dir = work_overlays_dir(job_id)
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)

    # Save detected overlay preview
    bgr = pil_to_cv(page_img)
    det_vis = draw_boxes(bgr, detections, color=(0, 0, 255), thickness=2)
    det_bytes = cv_to_png_bytes(det_vis)
    with open(os.path.join(pages_dir, f"page_{page_idx:04d}_detected.png"), "wb") as f:
        f.write(det_bytes)

    # Save replaced preview by compositing new_logo onto image (visual only)
    replaced_bgr = bgr.copy()
    logo_bgr = pil_to_cv(new_logo)
    lh, lw = logo_bgr.shape[:2]
    for d in detections:
        # Resize logo to box
        if d.w <= 0 or d.h <= 0:
            continue
        resized = cv2.resize(logo_bgr, (d.w, d.h), interpolation=cv2.INTER_AREA)
        x0, y0, w, h = d.x, d.y, d.w, d.h
        H, W = replaced_bgr.shape[:2]
        x0 = max(0, min(x0, W - 1))
        y0 = max(0, min(y0, H - 1))
        w = max(1, min(w, W - x0))
        h = max(1, min(h, H - y0))
        # paint white then paste
        replaced_bgr[y0 : y0 + h, x0 : x0 + w] = 255
        replaced_bgr[y0 : y0 + h, x0 : x0 + w] = resized

    rep_bytes = cv_to_png_bytes(replaced_bgr)
    with open(os.path.join(overlays_dir, f"page_{page_idx:04d}_replaced.png"), "wb") as f:
        f.write(rep_bytes)


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

        # Prepare templates
        templates_bgr: List[np.ndarray] = []
        for p in old_logo_paths:
            try:
                tpl_img = Image.open(p).convert("RGBA")
                templates_bgr.append(pil_to_cv(tpl_img))
            except Exception:
                continue
        if not templates_bgr:
            raise RuntimeError("No valid old_logo templates could be loaded.")

        findings_pages: List[PageFindings] = []
        detections_by_page: Dict[int, List[Detection]] = {}

        # Process each page
        for idx, page_img in enumerate(raster_pages):
            _update_status(job_id, message=f"Detecting logos on page {idx + 1}/{total_pages}")
            page_bgr = pil_to_cv(page_img)
            dets = multi_scale_template_match(page_bgr, templates_bgr, match_threshold=params.match_threshold)
            detections_by_page[idx] = dets

            # Save previews
            _save_page_preview_images(job_id, idx, page_img, dets, new_logo_img)

            # Collect findings
            page_boxes = [
                BoundingBox(x=d.x, y=d.y, w=d.w, h=d.h, score=d.score, template_id=d.template_id) for d in dets
            ]
            findings_pages.append(PageFindings(page_index=idx, detections=page_boxes))

            # Update progress
            _update_status(job_id, progress=(idx + 1) / max(1, total_pages))

        # Write replaced PDF
        _update_status(job_id, message="Writing replaced PDF")
        out_pdf = _write_output_pdf(job_id, pdf_path, detections_by_page, new_logo_img, dpi=params.dpi)

        # Save metadata
        meta = {
            "job_id": job_id,
            "status": "completed",
            "progress": 1.0,
            "message": "Completed",
            "input_pdf": pdf_path,
            "output_pdf": out_pdf,
            "params": params.model_dump(),
            "findings": {
                "total_pages": total_pages,
                "pages": [fp.model_dump() for fp in findings_pages],
            },
            "timestamps": {"completed": time.time()},
        }
        save_metadata(job_id, meta)

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
        }
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
                    # Rehydrate Findings
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
        out_pdf = md.get("output_pdf")
        if not out_pdf or not os.path.exists(out_pdf):
            return None
        with open(out_pdf, "rb") as f:
            data = f.read()
        filename = os.path.basename(out_pdf)
        return filename, data
