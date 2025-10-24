from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger("storage.workspace")


# PUBLIC_INTERFACE
def get_root_workspace() -> Path:
    """Return the root workspace path for job storage.

    This reads BRANDING_WS_ROOT from environment variables if set; otherwise
    defaults to a container-writable path.

    Resolution order:
      1) BRANDING_WS_ROOT if set and creatable
      2) /tmp/branding_workspace (container-writable)
      3) ./workspace (as last resort)

    Adds verbose logs to help diagnose permission/path issues.
    """
    env_root = os.getenv("BRANDING_WS_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    # Prefer container-writable temp dir
    candidates.append(Path("/tmp/branding_workspace"))
    # Fallback to relative workspace
    candidates.append(Path("./workspace"))

    for cand in candidates:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            # extra: test we can write a temp file
            probe = cand / ".ws_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            logger.info("Workspace root resolved OK: %s", cand)
            return cand
        except Exception as e:
            logger.exception("Workspace root candidate failed: %s error=%s", cand, e)

    # If all candidates failed, raise to surface in health and logs
    raise RuntimeError("Unable to create or access any workspace root directory")


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
