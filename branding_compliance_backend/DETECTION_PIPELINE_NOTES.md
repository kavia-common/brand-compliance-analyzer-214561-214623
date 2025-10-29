# Detection Pipeline Notes

This backend uses PyMuPDF to rasterize PDFs and OpenCV to detect old logos using multi-scale template matching with rotation search, a scale prior, robust preprocessing, and a feature-matching fallback. Extensive diagnostics and tunable configuration are included.

Key stages:
- PDF rasterization: PyMuPDF with enforced minimum DPI 200 (recommended 250–300). Rasterization uses RGB colorspace and antialias for vector content to ensure consistent inputs across PDFs.
- Preprocessing: Convert page and template images to grayscale with CLAHE (tunable) and a light Gaussian blur to equalize contrast and reduce noise. Optional morphological white top-hat suppresses background shading. Alpha is normalized to white for templates.
- Matching (primary): Color-invariant matching via grayscale + OpenCV `TM_CCOEFF_NORMED` (normalized cross-correlation) over multiple scales with a small-angle rotation search (−15..+15°). A scale prior (based on page/template size statistics) nudges scales toward more likely sizes. A small adaptive relaxation of threshold is applied for tiny templates.
- Matching (fallback): ORB/AKAZE feature matching with BFMatcher and RANSAC homography; supports small rotations (configurable). Converts homography-projected template corners to page-space bounding boxes. Used only when template matching finds zero detections.
- NMS: Size-aware suppression using IoU plus a minimum center distance tuned by template size to avoid over-suppression of nearby small detections.
- Overlay: New logo is inserted into the PDF using detected boxes converted back to PDF points.

Parameters (DetectionConfig):
- dpi_min/dpi_max: bounds for rasterization DPI.
- match_threshold (default 0.8; adaptively relaxed to ~0.58 for tiny templates).
- scales: base scales list (default [0.5..1.6]); multiplied by a per-page scale prior.
- rotation_degrees: angles for small-angle rotation search (default −15..+15 step 3).
- clahe_clip_limit (default 2.0) and clahe_tile_grid (default 8); gaussian_blur kernel size (default 3).
- Background suppression: use_tophat (default True) with tophat_kernel (default 9).
- Fallback feature matching: feature_rotations, feature_min_inliers (8), feature_ransac_reproj_thresh (3.0).
- NMS: nms_iou (0.4) and nms_min_dist (6) further adjusted dynamically using detection width statistics.
- collect_page_diagnostics: includes reasons and search params per page.

Debugging and Diagnostics:
- Per-page detections are persisted in metadata at `findings.detections_dump[page_index]` including x, y, w, h, score, template_id.
- Per-page detection counts at `findings.per_page_detection_counts` and aggregate `findings.total_detections`.
- Per-page diagnostics (reasons and search params) are saved under `findings.page_diagnostics[page_index]`, including scale_prior, used_scales, rotations, and failure reasons like `low_contrast_page`, `no_template_match_above_threshold`, or `possible_rotation>15deg`.
- Approximate before/after improvement is reported at `findings.approx_improvement` when prior jobs for the same input exist.
- Raw rendered pages saved as `work/pages/page_XXXX_raw.png` alongside `page_XXXX_detected.png` and overlays in `work/overlays/page_XXXX_replaced.png`.
- Loaded template debug copies saved as `work/pages/template_XX.png` with load status recorded at `debug.templates`.
- Temporary debug endpoints:
  - GET `/debug/job/{job_id}`: returns metadata and lists debug artifacts.
  - POST `/debug/verify-last-job`: re-runs the latest job with DPI max(250, existing) and slightly relaxed threshold if the last run had zero detections. Returns a new verification job_id.

Notes:
- Working in grayscale with contrast normalization and top-hat reduces failures from color space and background shading differences.
- For very small rendered logos, correlation can be weak; the code adapts thresholds for tiny scales and provides a feature-matching fallback resilient to small rotations.
- The rotation and scale prior typically improve recall without materially impacting precision when combined with size-aware NMS.
