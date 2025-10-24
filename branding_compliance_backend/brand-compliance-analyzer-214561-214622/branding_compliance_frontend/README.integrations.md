# Frontend Integration Notes

- API base URL:
  - Uses `REACT_APP_API_BASE` when defined.
  - Defaults to `http://localhost:3001/api/v1` if not set.
- Start frontend (port 3000) and backend (port 3001). Ensure backend CORS allows:
  - http://localhost:3000
  - your cloud preview origin (set `PREVIEW_FRONTEND_ORIGIN` in backend)

Blob downloads/previews:
- Image previews are served with appropriate `Content-Type` and filename.
- For downloads (`/download?type=zip|report|both`), fetch as `blob`, then create an object URL and trigger a file save with `Content-Disposition` filename if needed.

Example fetch for blob:
```js
const res = await fetch(`${API_BASE}/jobs/${jobId}/download?type=zip`);
const blob = await res.blob();
const url = URL.createObjectURL(blob);
const a = document.createElement('a');
a.href = url;
a.download = (res.headers.get('Content-Disposition')?.match(/filename="?(.*)"?/)?.[1]) || 'outputs.zip';
document.body.appendChild(a);
a.click();
a.remove();
URL.revokeObjectURL(url);
```

Environment:
- `.env.example`
  - `REACT_APP_API_BASE=http://localhost:3001/api/v1`
