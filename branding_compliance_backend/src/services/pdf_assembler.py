from __future__ import annotations

"""
PDF assembly utilities using PyMuPDF (fitz).

Assembles a list of per-page image paths into a PDF, preserving page sizes from original points.
"""

from pathlib import Path
from typing import List, Dict, Tuple
import fitz  # PyMuPDF


# PUBLIC_INTERFACE
def assemble_pdf(page_images: List[Path], out_pdf_path: Path, page_sizes_points: Dict[int, Tuple[float, float]]) -> Path:
    """Assemble a PDF from page images using provided original page sizes (points)."""
    out_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    try:
        for idx, img_path in enumerate(page_images):
            w_pt, h_pt = page_sizes_points.get(idx, (595.0, 842.0))  # default A4-ish points
            page = doc.new_page(width=w_pt, height=h_pt)
            # Place image full-bleed
            rect = fitz.Rect(0, 0, w_pt, h_pt)
            page.insert_image(rect, filename=str(img_path), keep_proportion=False)
        doc.save(str(out_pdf_path))
    finally:
        doc.close()
    return out_pdf_path
