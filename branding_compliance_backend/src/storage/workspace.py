from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


# PUBLIC_INTERFACE
def get_root_workspace() -> Path:
    """Return the root workspace path for job storage.

    This reads BRANDING_WS_ROOT from environment variables if set; otherwise
    defaults to a 'workspace' folder under the project root.
    """
    env_root = os.getenv("BRANDING_WS_ROOT")
    if env_root:
        root = Path(env_root)
    else:
        # Default relative path within container working directory
        root = Path("workspace")
    root.mkdir(parents=True, exist_ok=True)
    return root


# PUBLIC_INTERFACE
def get_job_workspace(job_id: str) -> Dict[str, Path]:
    """Return workspace paths for a job and ensure required directories exist.

    Layout:
      /{root}/jobs/{job_id}/
        uploads/
        analysis/
        previews/
        outputs/
        report/

    Args:
        job_id: Unique job identifier.

    Returns:
        Dict mapping section name to Path objects for convenience, including 'root' and 'job'.
    """
    root = get_root_workspace()
    jobs_root = root / "jobs"
    job_dir = jobs_root / job_id

    uploads = job_dir / "uploads"
    analysis = job_dir / "analysis"
    previews = job_dir / "previews"
    outputs = job_dir / "outputs"
    report = job_dir / "report"

    for p in [jobs_root, job_dir, uploads, analysis, previews, outputs, report]:
        p.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "jobs_root": jobs_root,
        "job": job_dir,
        "uploads": uploads,
        "analysis": analysis,
        "previews": previews,
        "outputs": outputs,
        "report": report,
    }
