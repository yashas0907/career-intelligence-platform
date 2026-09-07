# 100% FREE deployment guide

No credit cards, no paid tiers, ever. Two paths below — both driven from GitHub.

## Stack (all free)

| Piece | Service | Cost |
|---|---|---|
| Hosting | **Hugging Face Spaces** (Docker, 2 vCPU / 16 GB) | $0 forever |
| Deploy automation | **GitHub Actions** (already wired: `.github/workflows/deploy-hf.yml`) | $0 (public repos) |
| LLM (optional) | **Google Gemini free tier** | $0, no card — https://aistudio.google.com |
| Embeddings | **sentence-transformers** (runs inside the container, CPU) | $0 |
| Database | SQLite (inside the free 20 GB volume) | $0 |

---

## A. Deploy to Hugging Face Spaces (~10 minutes, one time)

1. **Create the Space**
   - Go to https://huggingface.co → sign up (free)
   - https://huggingface.co/new-space
   - Name: `career-intelligence` · SDK: **Docker** · License: MIT → Create
2. **Create a write token**
   - https://huggingface.co/settings/tokens → *New token* → role **write**
   - Copy it (`hf_...`)
3. **Add secrets to the GitHub repo**
   - Repo → Settings → Secrets and variables → Actions → *New repository secret*
   - `HF_TOKEN` = your `hf_...` token
   - `HF_SPACE` = `your-username/career-intelligence`
4. **Add the Gemini key to the Space** (optional but recommended)
   - Space → Settings → *Variables and secrets* → New secret
   - `GEMINI_API_KEY` = your key from https://aistudio.google.com/apikey (free, no card)
5. **Push to `main`** — the GitHub Action runs tests, builds, and pushes to the
   Space automatically. First build takes ~4–6 min; then your app is live at:

```
https://your-username-career-intelligence.hf.space
```

Every later `git push` redeploys automatically.

### Why HF Spaces over Render's free tier?
Render free spins down after 15 idle minutes (cold start ~40 s) and expires
after 90 days. HF Spaces free stays **always-on** with no expiry.

## B. Alternative: GitHub Codespaces demo (zero deploy)

Repo → green *Code* button → *Codespaces* → create. Terminal:

```bash
cd backend && pip install -r requirements.txt && uvicorn app.main:app --port 8000
```

Forwarded port 8000 gives you the single-service app (SPA + API). Good for
showing the project in an interview from any browser.

---

## Verifying the live deployment

- `https://<you>-career-intelligence.hf.space/api/health` → `{"status":"ok","llm_provider":"gemini",...}`
- Landing page → **Try the live demo** → analyses stream in
- Assistant → token-by-token answers with citations (Gemini if key set, extractive if not)
- **Fix my resume** → improved text + DOCX download works

## Notes

- HF free containers restart weekly — SQLite lives in an ephemeral FS, so
  uploaded resumes reset periodically. Fine for a portfolio demo; the volume
  persists on the paid tier or via the Postgres swap documented in README.
- The Dockerfile auto-detects `$PORT` (HF injects 7860).
- Never put the Gemini key in the repo — only HF Space secrets / GitHub secrets.
