# AI Career Intelligence Platform

A full-stack, explainable resume â†” job matching system. Upload a resume, paste
job descriptions, and get a **deterministically computed, fully explained**
compatibility breakdown â€” skill gaps, transferable skills, ATS-oriented review,
ranked job fit, and a RAG career assistant that cites your own documents.

> The core engineering principle: **an LLM is never asked to produce a number.**
> Every score is computed by a documented, tested, deterministic engine. LLMs are
> used only where they genuinely add value (hybrid extraction, narration of
> already-computed facts, RAG answers) â€” and the app runs fully without one.

---

## Why this exists

Most "resume analyzer" projects are a thin wrapper: `resume â†’ LLM â†’ paragraph`.
This project demonstrates an actual AI/ML **system**:

- a document ingestion pipeline with real validation and failure modes
- a skill taxonomy with alias normalization and a relatedness graph
- a hybrid extractor (deterministic heuristics + validated LLM merge)
- a multi-signal, weighted scoring engine with published methodology
- embedding-backed semantic similarity with a pluggable backend chain
- a grounded RAG assistant with citations and an extractive offline fallback
- a relational database, typed API, structured logging, Docker, and 135 tests

It was inspired by the patterns in [KalyanMurapaka45/DocGenius](https://github.com/KalyanMurapaka45/DocGenius-Revolutionizing-PDFs-with-AI)
(PDF + chunking + embeddings + LLM Q&A) and the chatbot projects in the
[AI-Project-Gallery](https://github.com/KalyanM45/AI-Project-Gallery) reference
repository â€” then substantially redesigned around explainability and
deterministic scoring rather than a single LLM call.

## Features

| Feature | How it works |
|---|---|
| **Resume ingestion** | PDF (pdfminer.six) / DOCX (python-docx, incl. tables) / TXT; rejects corrupted, fake, empty, oversized files with actionable errors |
| **Structured profile** | Hybrid extraction: section heuristics + regex contacts/dates + taxonomy scan (each skill keeps its evidence snippet) + optional validated LLM merge |
| **JD analysis** | Required vs preferred split from bullet cues, seniority, min-years, education requirements, responsibilities |
| **Skill normalization** | ~100 canonical skills, ~250 aliases (`sklearnâ†’scikit-learn`, `k8sâ†’kubernetes`), relatedness graph (TensorFlowâ†”PyTorch) for transferable credit |
| **Match scoring** | Deterministic weighted engine: Skills 35% / Experience 20% / Projects 15% / Education 10% / Semantic 20% â€” see [docs/scoring.md](docs/scoring.md) |
| **Missing skills analysis** | Strong / transferable / missing-required / missing-preferred, each with evidence |
| **ATS-oriented review** | Keyword coverage, section completeness, contact discoverability, formatting risk, readability â€” explicitly *not* a proprietary-ATS replica |
| **Job ranking** | Multi-JD analysis with computed ranking + reasons |
| **Improvement plan** | Learn/build/improve recommendations derived from actual gaps â€” never suggests fabricating experience |
| **RAG assistant** | Chunks + embeds resume & JDs; **answers stream token-by-token (SSE)** and cite retrieved sources; works offline via extractive fallback |
| **Resume Fixer** | **Fixes the uploaded resume directly**: rebuilt contact header, standardized sections, ATS-safe formatting, JD-aligned skill ordering, stronger bullets (LLM, guarded) â€” with a change log and `.docx`/`.txt` download. **No-fabrication guard**: only your own stated facts are ever used |
| **Real-time UX** | **Job analyses stream in one-by-one as each completes** â€” live progress bar + results appearing in sequence, chat with typing animation |
| **Live demo mode** | One click on the landing page seeds a demo resume + 3 sample JDs â€” visitors can try everything instantly |
| **Session persistence** | Survives page refresh (localStorage + server-side restore of resume, analyses and history) |
| **Abuse guards** | Per-IP rate limiting (30/min default) on expensive endpoints, payload caps, file validation |
| **Explainability** | Every number traces to evidence strings; stored per analysis for later audit |
| **Deployable** | Single Docker service serves API + frontend on one port; Render/Railway/Fly-ready ([docs/deploy.md](docs/deploy.md)) |

## Tech stack

**Backend** â€” Python 3.11+ Â· FastAPI Â· Pydantic v2 Â· SQLAlchemy 2 (SQLite) Â·
pdfminer.six Â· python-docx Â· NumPy Â· OpenAI client (optional) Â·
sentence-transformers (optional)

**Frontend** â€” React 18 Â· TypeScript Â· Vite (no UI framework â€” hand-rolled design system)

**Infra** â€” Docker + docker-compose (backend + nginx-served frontend) Â· pytest

## Screenshots

*(placeholder â€” add `docs/screenshots/` with: dashboard, upload with extracted
skills, match breakdown with evidence, skill comparison grid, ATS panel,
ranking, assistant with citations)*

## Quick start (local dev)

```bash
# 1. Backend (http://localhost:8000, docs at /api/docs)
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# 2. Frontend (http://localhost:5173, proxies /api -> :8000)
cd frontend
npm install
npm run dev
```

No API key needed â€” the app runs in fully offline heuristic mode.
**Free AI option:** get a Gemini key at https://aistudio.google.com (no credit
card) and set `GEMINI_API_KEY` in `.env` (copy from `.env.example`) to enable
LLM streaming chat, extraction and bullet rewriting at $0 cost.

*Optional:* `pip install sentence-transformers` for free local semantic
embeddings (not installed by default — it's a large torch download; the
hashing backend covers offline mode without it).

## Free deployment

**Hosted free forever on Hugging Face Spaces, auto-deployed from GitHub** â€”
full step-by-step in [docs/deploy-free.md](docs/deploy-free.md). Summary:

1. Create a free Docker Space at huggingface.co
2. Add `HF_TOKEN` + `HF_SPACE` secrets to the GitHub repo
3. Push to `main` â€” GitHub Action runs the full test suite, then deploys
4. Live at `https://<you>-career-intelligence.hf.space` (always-on, no cold starts, no expiry)

## Quick start (Docker)

```bash
cp .env.example .env       # optionally set OPENAI_API_KEY
docker compose up --build
# frontend: http://localhost:5173 Â· API docs: http://localhost:8000/api/docs
```

## API overview

```
GET  /api/health                 liveness + backend capabilities
POST /api/resume/upload          multipart file â†’ structured profile
GET  /api/resume/{id}            stored profile
GET  /api/resume/{id}/history    past analyses for the resume
POST /api/jobs/analyze           {resume_id, jobs[]} â†’ full analyses (â‰¤10 jobs)
POST /api/jobs/analyze/stream    SSE: each analysis emitted the moment it's ready
POST /api/match                  single-job convenience alias
GET  /api/analysis/{id}          stored analysis with full breakdown
GET  /api/recommendations        ?resume_id= â†’ ranked jobs with reasons
POST /api/chat                   {question, analysis_id?} â†’ grounded answer + sources
POST /api/chat/stream            SSE: sources â†’ token stream â†’ done(method)
GET  /api/chat/history           ?analysis_id= â†’ conversation log
GET  /api/demo/sample            demo resume + 3 JDs (no state)
POST /api/demo/seed              seed the demo resume â€” instant full product tour
POST /api/fix/{resume_id}        FIX the uploaded resume directly (JD-aware) + change log
GET  /api/fix/{resume_id}/download  ?format=docx|txt â€” download the fixed resume
```

Chat works with **just a resume** (no analysis needed) â€” the assistant grounds
in whatever documents exist. Rate limited: 30 req/min/IP on expensive endpoints
(`RATE_LIMIT_PER_MINUTE`).

All errors return a consistent `{detail, error_code}` envelope; validation
failures tell you which field failed.

## Environment variables

See [.env.example](.env.example) â€” highlights:

| Var | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | sqlite (auto) | any SQLAlchemy URL |
| `EMBEDDING_BACKEND` | `auto` | `openai` / `st` / `hashing` / `auto` (tries in that order) |
| `OPENAI_API_KEY` | â€” | enables LLM + OpenAI embeddings (optional) |
| `MAX_UPLOAD_MB` | 10 | upload size cap |
| `LLM_ENABLED` | true | set false to force pure-deterministic mode |

## Testing

```bash
pip install -r backend/requirements.txt
python -m pytest tests -v              # 135 tests
python tests/eval_extraction.py        # labeled-corpus P/R/F1 report
```

- **135 tests**: scoring methodology contract, skill normalization, parsing (real
  generated PDF/DOCX bytes, malformed files), extraction, ATS, ranking, RAG
  retrieval, API lifecycle + edge cases, embedding stability, SSE streaming
  (chat + analysis), demo seeding, rate limiting.
- **Extraction evaluation**: 3 labeled resumes with per-field ground truth;
  measured macro-F1 â€” name/email 1.00, skills 0.86, projects 1.00, experience
  titles 1.00, education 1.00 (details + caveats in docs/evaluation.md). CI
  enforces these as regression thresholds.
- **CI**: GitHub Actions runs backend tests + the extraction report + frontend
  build on every push/PR (`.github/workflows/ci.yml`).

## Documentation

- [docs/scoring.md](docs/scoring.md) â€” the exact match-score methodology with worked examples
- [docs/architecture.md](docs/architecture.md) â€” system + AI pipeline diagrams, deterministic-vs-LLM table, security notes
- [docs/evaluation.md](docs/evaluation.md) â€” evaluation methodology and honest limitations

## Limitations

Summarized from [docs/evaluation.md](docs/evaluation.md):

1. Heuristic extraction favors conventional resume layouts; exotic ones degrade
   gracefully to whole-text skill scanning.
2. Taxonomy targets tech/AI/data roles (~100 skills); unlisted skills are
   compared verbatim.
3. Default offline embeddings are lexical (hashing) â€” for true semantic
   scoring configure OpenAI or install sentence-transformers. The active backend
   is always visible in the UI and `/api/health`.
4. Experience years are approximated from date ranges and explicit statements.
5. ATS analysis is heuristic, based on publicly-known parsing signals.
6. Ranking orders provided JDs; it is not a job-search engine.
7. The evaluation corpus is small (3 labeled resumes) â€” good for regression
   gating, not a generalization claim.

## Future improvements

- Per-field precision/recall eval on an annotated resume corpus (see docs/evaluation.md)
- LLM-judge scoring of RAG faithfulness in CI
- Auth + multi-user support (schema already isolates by user_id)
- Postgres migration path (one-line `DATABASE_URL` swap)
- PDF layout detection (columns/graphics heuristics) for richer ATS signals
- Streaming chat, resume versioning and diff-based coaching

## License

MIT â€” see [LICENSE](LICENSE).
