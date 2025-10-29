from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

STORAGE_ROOT = os.path.join(os.getcwd(), "storage")


def ensure_storage_root() -> None:
    """Ensure the root storage directory exists."""
    os.makedirs(STORAGE_ROOT, exist_ok=True)


def job_root(job_id: str) -> str:
    return os.path.join(STORAGE_ROOT, "jobs", job_id)


def input_dir(job_id: str) -> str:
    return os.path.join(job_root(job_id), "input")


def input_pdfs_dir(job_id: str) -> str:
    return os.path.join(input_dir(job_id), "pdfs")


def input_logos_old_dir(job_id: str) -> str:
    return os.path.join(input_dir(job_id), "logos", "old")


def input_logos_new_dir(job_id: str) -> str:
    return os.path.join(input_dir(job_id), "logos", "new")


def work_dir(job_id: str) -> str:
    return os.path.join(job_root(job_id), "work")


def work_pages_dir(job_id: str) -> str:
    return os.path.join(work_dir(job_id), "pages")


def work_masks_dir(job_id: str) -> str:
    return os.path.join(work_dir(job_id), "masks")


def work_overlays_dir(job_id: str) -> str:
    return os.path.join(work_dir(job_id), "overlays")


def output_dir(job_id: str) -> str:
    return os.path.join(job_root(job_id), "output")


def output_replaced_dir(job_id: str) -> str:
    return os.path.join(output_dir(job_id), "replaced")


def metadata_path(job_id: str) -> str:
    return os.path.join(job_root(job_id), "metadata.json")


def ensure_job_dirs(job_id: str) -> None:
    """Create the full directory structure for a job."""
    for p in [
        job_root(job_id),
        input_dir(job_id),
        input_pdfs_dir(job_id),
        input_logos_old_dir(job_id),
        input_logos_new_dir(job_id),
        work_dir(job_id),
        work_pages_dir(job_id),
        work_masks_dir(job_id),
        work_overlays_dir(job_id),
        output_dir(job_id),
        output_replaced_dir(job_id),
    ]:
        os.makedirs(p, exist_ok=True)


def save_metadata(job_id: str, data: Dict[str, Any]) -> None:
    """Persist job metadata to disk."""
    path = metadata_path(job_id)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_metadata(job_id: str) -> Dict[str, Any]:
    """Load job metadata. Raises FileNotFoundError if missing."""
    path = metadata_path(job_id)
    with open(path, "r") as f:
        return json.load(f)


def load_metadata_safely(job_id: str) -> Optional[Dict[str, Any]]:
    try:
        return load_metadata(job_id)
    except Exception:
        return None
