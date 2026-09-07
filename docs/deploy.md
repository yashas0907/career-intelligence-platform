# Deploy guide (single service)

The app ships as ONE container: FastAPI serves both the API (`/api/*`) and the
built React frontend (`/`). No separate web server, no CORS setup, one port.

## Option A — Render / Railway / Fly.io (easiest)

1. Push the repo to GitHub.
2. Create a **Web Service** pointing at the repo.
   - **Docker directory / dockerfile path:** `deploy/Dockerfile`
     (set build context to the repo root — Render: "Root directory" empty,
     Dockerfile path `deploy/Dockerfile`)
   - **Health check path:** `/api/health`
3. Environment variables (all optional — app runs without any):
   - `OPENAI_API_KEY` — enables LLM narration + OpenAI embeddings
   - `EMBEDDING_BACKEND=auto` (default; falls back gracefully)
4. That's it. The platform gives you HTTPS on a public URL.

## Option B — Any VPS with Docker

```bash
git clone <your-repo>
cd <your-repo>
docker compose -f deploy/compose.yml up --build -d
# app: http://your-server:8000  (put nginx/caddy TLS in front for prod)
```

Data persists in the `app_data` volume (SQLite + vector store + logs).

## Option C — Render.yaml blueprint

`.render/render.yaml` in the repo defines the service; on Render choose
"New → Blueprint" and it auto-configures.

## Post-deploy verification checklist

- `GET /api/health` → `{"status": "ok"}`
- Open the site → hero loads → click **"Try the live demo"** → demo resume seeds
- Paste the sample ML JD → results stream in
- Assistant answers stream and cite sources
- Session survives a page refresh (localStorage + DB)

## Production notes

- SQLite is fine for a portfolio deployment; for multi-instance scaling set
  `DATABASE_URL` to a managed Postgres connection string (SQLAlchemy handles it).
- Rate limiting is per-instance in-memory (30 req/min default,
  `RATE_LIMIT_PER_MINUTE`). For multi-instance, move to Redis-based limiting.
- Resumes are personal data. On managed platforms the volume is your retention
  boundary — add a deletion cron if needed (`backend_data/career_intel.db`).
- Never commit `.env`; set secrets in the platform dashboard instead.
