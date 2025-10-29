# Detection Pipeline Notes

This backend uses PyMuPDF to rasterize PDFs and OpenCV to detect old logos using multi-scale template matching.

Key stages:
- PDF rasterization: PyMuPDF with DPI from request (default 250). Vector PDFs are rasterized consistently.
- Preprocessing: Convert page and template images to grayscale with CLAHE and a light Gaussian blur to equalize contrast and reduce noise. Matching occurs on grayscale for robustness.
- Matching: OpenCV `TM_CCOEFF_NORMED` over multiple scales: [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5]. A small adaptive relaxation of threshold is applied for small templates.
- NMS: Suppresses overlapping matches using IoU to reduce duplicates.
- Overlay: New logo is inserted into the PDF using detected boxes converted back to PDF points.

Parameters:
- dpi (default 250): PDF rasterization resolution.
- match_threshold (default 0.8 exposed by API; internally a minimum of 0.75 is applied): Lower if false negatives occur; raise if false positives occur.
- max_pages: Optional cap on pages.

Debugging:
- Per-page detection counts are saved in job metadata at `findings.per_page_detection_counts` with `findings.total_detections`.
- Preview images saved under `storage/jobs/<job_id>/work/pages` (detected) and `/work/overlays` (replaced).
- If detections are zero, consider:
  - Lower `match_threshold` to 0.7.
  - Increase DPI to 300 for small logos.
  - Ensure the old logo template closely matches the target (same orientation and color-on-white). Transparent backgrounds are supported.

Notes:
- Different color spaces across PDFs/templates can reduce correlation. Working in grayscale with contrast normalization reduces these failures.
- For very small rendered logos, signal is weak; the code relaxes the threshold slightly for tiny scales.
