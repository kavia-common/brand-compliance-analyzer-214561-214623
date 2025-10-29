import json
import os
from typing import Any

import numpy as np

from src.services.json_utils import to_native_jsonable
from src.storage.paths import ensure_storage_root, ensure_job_dirs, save_metadata, metadata_path


def test_to_native_jsonable_numpy_scalars_and_arrays():
    data = {
        "i": np.int64(5),
        "f": np.float32(3.14),
        "b": np.bool_(True),
        "arr": np.array([np.int32(1), np.float64(2.5), np.bool_(False)]),
        "set_tuple": (1, 2, 3),
        "set_val": {4, 5},
    }
    norm = to_native_jsonable(data)
    assert isinstance(norm["i"], int)
    assert isinstance(norm["f"], float)
    assert isinstance(norm["b"], bool)
    assert isinstance(norm["arr"], list)
    assert isinstance(norm["set_tuple"], list)
    assert isinstance(norm["set_val"], list)
    # Ensure JSON can dump
    json.dumps(norm)


def test_save_metadata_handles_numpy(tmp_path: Any, monkeypatch: Any):
    # Redirect storage root to tmp for test isolation
    monkeypatch.setenv("PWD", str(tmp_path))  # not strictly used; functions use os.getcwd()
    # Patch CWD via chdir
    os.chdir(tmp_path)

    ensure_storage_root()
    job_id = "test_job"
    ensure_job_dirs(job_id)
    meta = {
        "job_id": job_id,
        "status": "completed",
        "progress": np.float64(1.0),
        "findings": {
            "total_pages": np.int64(2),
            "per_page_detection_counts": np.array([np.int32(0), np.int32(1)]),
            "total_detections": np.int64(1),
        },
    }
    # Should not raise
    save_metadata(job_id, meta)
    # And file should be valid JSON
    with open(metadata_path(job_id), "r") as f:
        loaded = json.load(f)
    assert loaded["progress"] == 1.0
    assert loaded["findings"]["total_pages"] == 2
    assert loaded["findings"]["total_detections"] == 1
    assert isinstance(loaded["findings"]["per_page_detection_counts"], list)
