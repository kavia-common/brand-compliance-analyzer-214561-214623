# brand-compliance-analyzer-214561-214623

Developer notes

- JSON normalization:
  - All metadata persisted to storage and API responses are normalized to native Python JSON-serializable types using src/services/json_utils.to_native_jsonable.
  - Numpy scalars (np.integer, np.floating, np.bool_) are converted to int/float/bool.
  - Numpy arrays are converted to lists.
  - Sets and tuples are converted to lists.
  - Dataclasses and Pydantic models are converted to dicts and normalized recursively.
  - Storage.save_metadata applies normalization automatically; API endpoints normalize dict responses and StatusResponse models before returning.
- Minimal logging:
  - json_utils emits an INFO log "Applied JSON normalization." when normalization occurs. Integrators may configure logging at the app level to manage handlers/levels.

Detection configuration and diagnostics

- Tunable detection parameters are centralized in src/services/detect_config.DetectionConfig with sane defaults:
  - Small-angle rotation search (−15..+15°), scale prior from page/template sizes, CLAHE tuning, optional white top-hat for background suppression, and size-aware NMS.
  - Color-invariant matching (grayscale + TM_CCOEFF_NORMED) with ORB/AKAZE fallback (RANSAC).
- Per-page diagnostics and failure reasons are stored in metadata under findings.page_diagnostics[page]:
  - Includes used scales, rotation list, scale_prior, and reasons such as low_contrast_page, no_template_match_above_threshold, possible_rotation>15deg.
- A small before/after report is written to report_summary.json in the job root comparing total detections with previous jobs for the same input PDF (best-effort proxy for recall).

Testing

- Run pytest within the backend container to validate JSON normalization:
  - tests/test_json_utils.py includes tests verifying np scalars/arrays and save_metadata robustness.
