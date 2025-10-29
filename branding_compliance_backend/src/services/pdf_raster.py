from __future__ import annotations

"""
PDF rasterization utilities using PyMuPDF (fitz).

Rasterizes PDFs into per-page PNGs at a given DPI and records page sizes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import fitz  # PyMuPDF


@dataclass
class RasterPageInfo:
    index: int
    image_path: Path
    width_px: int
    height_px: int
    dpi: int


# PUBLIC_INTERFACE
def rasterize_pdf(pdf_path: Path, out_dir: Path, dpi: int = 300, max_pages: Optional[int] = None) -> Tuple[List[RasterPageInfo], Dict[int, Tuple[float, float]]]:
    """Rasterize a PDF to per-page PNGs.

    Returns:
        (pages, page_sizes_points)
        pages: list of RasterPageInfo describing raster outputs
        page_sizes_points: mapping page_index -> (width_points, height_points) for later reassembly
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pages: List[RasterPageInfo] = []
    page_sizes: Dict[int, Tuple[float, float]] = {}
    with fitz.open(str(pdf_path)) as doc:
        count = doc.page_count
        limit = min(count, max_pages) if max_pages else count
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i in range(limit):
            page = doc.load_page(i)
            page_sizes[i] = (float(page.rect.width), float(page.rect.height))  # points
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_path = out_dir / f"{i:04d}.png"
            pix.save(str(img_path))
            pages.append(RasterPageInfo(index=i, image_path=img_path, width_px=pix.width, height_px=pix.height, dpi=dpi))
    return pages, page_sizes
