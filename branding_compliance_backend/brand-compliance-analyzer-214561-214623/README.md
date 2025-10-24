# brand-compliance-analyzer-214561-214623

Monorepo with:
- branding_compliance_backend (FastAPI on 3001)
- branding_compliance_frontend (React on 3000)

Quick start:
- Backend
  - cd branding_compliance_backend
  - pip install -r requirements.txt
  - uvicorn src.api.main:app --host 0.0.0.0 --port 3001 --reload
- Frontend
  - cd ../branding_compliance_frontend
  - Copy .env.example to .env and adjust REACT_APP_API_BASE
  - npm install && npm start

Configuration:
- Backend CORS allows http://localhost:3000 by default; add preview origin via PREVIEW_FRONTEND_ORIGIN or extras via CORS_EXTRA_ORIGINS.
- Frontend uses REACT_APP_API_BASE; defaults to http://localhost:3001/api/v1 if not set.

Docs:
- Backend OpenAPI: http://localhost:3001/docs
- Frontend integration notes: branding_compliance_frontend/README.integrations.md
