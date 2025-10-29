# Branding Compliance Backend (FastAPI)

Processes images/documents for branding compliance and exposes REST APIs for upload, analysis, previews, fixing, and downloads.

## Run locally

- Python 3.10+
- Install deps:
  - `pip install -r requirements.txt`
- Start server:
  - `uvicorn src.api.main:app --host 0.0.0.0 --port 3001 --reload`

OpenAPI docs: http://localhost:3001/docs

## Configuration (.env)

Do not commit secrets. Place environment variables in a local `.env` read by uvicorn/container.

- BRANDING_WS_ROOT: Optional path for workspace storage (default: ./workspace)
- PREVIEW_FRONTEND_ORIGIN: Optional additional allowed CORS origin (e.g. preview URL)
- CORS_EXTRA_ORIGINS: Optional comma-separated list of extra CORS origins
- DETECTOR: template|orb|hybrid (default: hybrid)
- QUALITY: fast|balanced|best (default: balanced)

## Download/Preview headers

- `/api/v1/jobs/{job_id}/assets/{asset_id}/preview?view=original|overlay|fixed` returns the correct Content-Type for images/documents and includes a filename.
- `/api/v1/jobs/{job_id}/download?type=zip|report|both` returns:
  - `application/zip` for zip, `application/json` for report
  - `Content-Disposition: attachment; filename="..."`

## Status/Results shapes

- Status (`GET /api/v1/jobs/{job_id}/status`) returns:
  - `job_id, status, total_assets, analyzed_assets, failed_assets, progress_percent, issues_total, issues_high_or_above, updated_at`
- Results (`GET /api/v1/jobs/{job_id}/results`) returns:
  - `{ assets: Asset[], issues: Issue[], summary: ReportSummary | null }`

## Notes

- Analysis runs asynchronously via background tasks.
- Use the status/results endpoints to poll progress.

### Troubleshooting CORS / Failed to fetch

If the frontend shows "Failed to fetch" on createJob:

1) Verify the frontend API base points to this backend on port 3001 and includes `/api/v1`.
2) Check backend CORS: call `GET /cors-check` from a browser tab; response includes `allowed_origins` and a `suggested_frontend_origin`.
3) Set the environment variable `PREVIEW_FRONTEND_ORIGIN` to your frontend origin (e.g., `https://<host>:3000`) or provide a comma-separated list in `CORS_EXTRA_ORIGINS`.
4) Restart the backend after changing env vars.

See `.env.example` for variables.
