# 100% FREE deployment guide (verified 2026)

> **Reality check (verified against provider docs):**
> - ❌ Hugging Face **Docker Spaces are now PRO-only** ($9/mo) — free accounts get
>   static spaces + 2 ZeroGPU **Gradio** spaces only
> - ❌ Koyeb removed its free tier
> - ❌ Heroku free tier is long gone
> - ✅ **Render free web services still exist**: 750 hrs/month, GitHub-integrated,
>   no credit card required — **this is the recommended free option**

## Recommended: Render free tier (~10 min, $0)

The repo is already wired for it (`.render/render.yaml` + `deploy/render.Dockerfile`).

1. Go to **https://dashboard.render.com/register** → sign up (free, no card)
2. Click **New → Web Service**
3. Connect your GitHub account → select `career-intelligence-platform`
4. It auto-detects the Blueprint from `.render/render.yaml` — or configure manually:
   - **Runtime:** Docker · **Dockerfile path:** `deploy/render.Dockerfile` · **Context:** repo root
   - **Plan:** **Free**
   - **Health check path:** `/api/health`
5. Environment variables (all optional):
   - `GEMINI_API_KEY` = free key from https://aistudio.google.com/apikey
     (enables LLM chat streaming, extraction merge, bullet rewriting — $0)
6. Click **Create Web Service** → first build ~5 min → live at
   `https://career-intelligence-xxxx.onrender.com`

**Free-tier behavior to know (honest):**
- Sleeps after 15 min without traffic; first request after sleep takes ~1 min
  (cold start). The frontend is static and served from the same service, so the
  app wakes on any click.
- 512 MB RAM — that's why the Render image uses the zero-dependency hashing
  embedding backend (no torch). With a Gemini key you still get full LLM features.
- SQLite is ephemeral on the free tier (resumes reset on redeploys/sleeps) —
  fine for demos. Render free Postgres exists but expires after 30 days, so we
  don't wire it by default.

## Keep-alive (optional trick)

Free instances sleep after 15 min idle. If you want it always awake for a
demo/interview, ping it every 10 minutes from any free cron service (e.g.
cron-job.org) hitting `/api/health`. That stays within the 750 free hours
(750 h > 744 h max month) — zero cost.

## Alternative free paths

| Option | How | Trade-offs |
|---|---|---|
| **GitHub Codespaces** | Repo → Code → Codespaces → `cd backend && pip install -r requirements-ci.txt && uvicorn app.main:app` (forward port 8000) | Runs on demand in a browser/VS Code; free 120 core-hrs/mo; great for live interviews, not a public URL |
| **Local demo** | `uvicorn app.main:app` after copying `frontend/dist` → `backend/static` | Perfect for showing recruiters in person |

## Verifying your deployment

1. `https://<your-app>.onrender.com/api/health` → `{"status":"ok","llm_provider":"gemini"...}` 
2. Landing page → **Try the live demo** → analyses stream in
3. Assistant → answers stream with citations
4. Results → **Fix my resume** → improved text + DOCX download

## Post-push updates

Render auto-deploys on every push to `main` (auto-sync is on by default).
CI on GitHub runs the full 135-test suite before anything reaches Render.
