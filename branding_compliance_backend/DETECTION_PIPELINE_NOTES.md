# Detection Pipeline Notes

This backend uses PyMuPDF to rasterize PDFs and OpenCV to detect old logos using multi-scale template matching, with a robust feature-matching fallback and extensive debug instrumentation.

Key stages:
- PDF rasterization: PyMuPDF with enforced minimum DPI 200 (recommended 250–300). Rasterization uses RGB colorspace and antialias for vector content to ensure consistent inputs across PDFs.
- Preprocessing: Convert page and template images to grayscale with CLAHE and a light Gaussian blur to equalize contrast and reduce noise. Alpha is normalized to white for templates.
- Matching (primary): OpenCV `TM_CCOEFF_NORMED` over multiple scales: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.45, 1.6]. A small adaptive relaxation of threshold is applied for tiny templates.
- Matching (fallback): ORB/AKAZE feature matching with BFMatcher and RANSAC homography; supports small rotations (−10..+10°). Converts homography-projected template corners to page-space bounding boxes. Used only when template matching finds zero detections.
- NMS: Tuned suppression using IoU plus a minimum center distance to avoid over-suppression of nearby small detections.
- Overlay: New logo is inserted into the PDF using detected boxes converted back to PDF points.

Parameters:
- dpi (default 250): PDF rasterization resolution (clamped to [200, 600]).
- match_threshold (default 0.8 exposed by API; internally adaptive down to ~0.58 for tiny templates): Lower if false negatives occur; raise if false positives occur.
- max_pages: Optional cap on pages.

Debugging and Diagnostics:
- Per-page detections are persisted in metadata at `findings.detections_dump[page_index]` including x, y, w, h, score, template_id.
- Per-page detection counts at `findings.per_page_detection_counts` and aggregate `findings.total_detections`.
- Raw rendered pages saved as `work/pages/page_XXXX_raw.png` alongside `page_XXXX_detected.png` and overlays in `work/overlays/page_XXXX_replaced.png`.
- Loaded template debug copies saved as `work/pages/template_XX.png` with load status recorded at `debug.templates`.
- Temporary debug endpoints:
  - GET `/debug/job/{job_id}`: returns metadata and lists debug artifacts.
  - POST `/debug/verify-last-job`: re-runs the latest job with safe defaults to verify non-zero detections after pipeline changes.

Notes:
- Working in grayscale with contrast normalization reduces failures from color space differences.
- For very small rendered logos, correlation can be weak; the code adapts thresholds for tiny scales and provides a feature-matching fallback resilient to small rotations.
