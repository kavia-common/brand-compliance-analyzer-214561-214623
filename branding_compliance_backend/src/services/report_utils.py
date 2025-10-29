from __future__ import annotations

import os
import json
from typing import Dict, Any, List, Optional

from src.storage.paths import load_metadata_safely, job_root


# PUBLIC_INTERFACE
def compute_before_after_for_same_input(current_job_id: str) -> Dict[str, Any]:
    """Find previous jobs with identical input_pdf and compute before/after detection count summary.

    Returns a dict with:
      - input_pdf
      - current_total
      - previous_best_total
      - delta
      - previous_job_ids
    """
    md = load_metadata_safely(current_job_id)
    if not md:
        return {}

    input_pdf = md.get("input_pdf")
    if not input_pdf:
        return {}

    current_total = None
    try:
        current_total = int(md.get("findings", {}).get("total_detections"))
    except Exception:
        current_total = None

    jobs_root = os.path.join(os.getcwd(), "storage", "jobs")
    previous_totals: List[int] = []
    previous_job_ids: List[str] = []
    if os.path.isdir(jobs_root):
        for d in os.listdir(jobs_root):
            if d == current_job_id:
                continue
            prev_md = load_metadata_safely(d)
            if prev_md and prev_md.get("input_pdf") == input_pdf:
                try:
                    t = int(prev_md.get("findings", {}).get("total_detections"))
                    previous_totals.append(t)
                    previous_job_ids.append(d)
                except Exception:
                    pass

    prev_best = max(previous_totals) if previous_totals else None
    delta = (current_total - prev_best) if (current_total is not None and prev_best is not None) else None

    return {
        "input_pdf": input_pdf,
        "current_total": current_total,
        "previous_best_total": prev_best,
        "delta": delta,
        "previous_job_ids": previous_job_ids,
    }


# PUBLIC_INTERFACE
def persist_job_report(job_id: str) -> Optional[str]:
    """Compute and persist a small before/after report JSON into the job root for quick access."""
    report = compute_before_after_for_same_input(job_id)
    if not report:
        return None
    root = job_root(job_id)
    out_path = os.path.join(root, "report_summary.json")
    try:
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
    except Exception:
        return None
    return out_path
