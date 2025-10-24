from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from src.models.asset import AssetStatus
from src.models.issue import Issue, IssueStatus
from src.models.job import JobStatus
from src.services.state_store import StateStore
from src.storage.workspace import get_job_workspace


class FixerService:
    """Applies automatic fixes to assets.

    This is a stubbed implementation that "fixes" by copying assets to outputs/.
    """

    def __init__(self, state_store: StateStore) -> None:
        self.state = state_store

    def fix_asset(self, job_id: str, asset_id: str) -> Path:
        """Fix a single asset and place result in outputs/."""
        w = get_job_workspace(job_id)
        job = self.state.get_job(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        assets = self.state.list_assets(job_id)
        asset = next((a for a in assets if a.id == asset_id), None)
        if asset is None:
            raise ValueError(f"Asset {asset_id} not found")

        src = w["job"] / asset.rel_path
        outputs = w["outputs"]
        outputs.mkdir(parents=True, exist_ok=True)
        dst = outputs / f"fixed_{asset.original_filename}"
        if src.exists():
            shutil.copy2(src, dst)

        # mark issues for this asset as fixed
        issues = self.state.list_issues(job_id)
        updated: List[Issue] = []
        for i in issues:
            if i.asset_id == asset_id and i.status != IssueStatus.fixed:
                i.status = IssueStatus.fixed
            updated.append(i)
        self.state._save_issues(job_id, updated)

        # mark asset as fixed
        self.state.update_asset_status(job_id, asset_id, AssetStatus.fixed)
        # update job status to fixing if not already
        if job.status not in (JobStatus.fixing, JobStatus.summarizing, JobStatus.completed):
            self.state.update_job_status(job_id, JobStatus.fixing)

        return dst

    def fix_all(self, job_id: str) -> List[Path]:
        """Fix all assets that have open issues."""
        w = get_job_workspace(job_id)
        outputs = w["outputs"]
        outputs.mkdir(parents=True, exist_ok=True)

        paths: List[Path] = []
        issues = self.state.list_issues(job_id)
        assets = self.state.list_assets(job_id)
        by_asset = {a.id: a for a in assets}
        affected_assets = {i.asset_id for i in issues}

        for aid in affected_assets:
            a = by_asset.get(aid)
            if not a:
                continue
            p = self.fix_asset(job_id, aid)
            paths.append(p)

        # finalize status progression
        self.state.update_job_status(job_id, JobStatus.summarizing)
        return paths
