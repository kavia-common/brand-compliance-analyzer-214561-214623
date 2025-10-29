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

Testing

- Run pytest within the backend container to validate JSON normalization:
  - tests/test_json_utils.py includes tests verifying np scalars/arrays and save_metadata robustness.
