# Architecture

## System overview

```
                        ┌─────────────────────────────────────────────┐
                        │                FRONTEND (React)             │
                        │  Vite + TS · Dashboard · Upload · Jobs ·     │
                        │  Results · Ranking · Assistant · History    │
                        └──────────────────┬──────────────────────────┘
                                           │ /api (proxy)
                        ┌──────────────────▼──────────────────────────┐
                        │            BACKEND (FastAPI)                │
                        │  routes → validation (Pydantic) → services  │
                        │  middleware: request-id, latency logs,     │
                        │  slow-request warnings, error envelopes     │
                        └───┬──────────┬──────────┬──────────┬────────┘
                            │          │          │          │
             ┌──────────────▼─┐ ┌──────▼──────┐ ┌─▼────────┐ ┌▼───────────┐
             │ DocumentParser │ │ Extractor   │ │ Matching │ │ RAG svc    │
             │ pdfminer/docx/ │ │ hybrid:     │ │ engine   │ │            │
             │ txt + validate │ │ heuristic + │ │ (pure    │ │ vector     │
             │                │ │ LLM merge   │ │ determ.) │ │ store +    │
             └───────┬────────┘ └──────┬──────┘ └────┬─────┘ │ LLM/quote  │
                     │                 │             │       │ fallback   │
                     │          ┌──────▼──────┐      │       └─────┬──────┘
                     │          │ SkillTaxonomy│     │             │
                     │          │ normalize + │      │             │
                     │          │ relatedness  │     │             │
                     │          └──────────────┘     │             │
                     │                               │             │
             ┌───────▼───────────────────────────────▼─────┐ ┌────▼────┐
             │ SQLite (SQLAlchemy)                         │ │ Chunks  │
             │ resumes · jobs · analyses · chat messages   │ │ + vectors│
             └─────────────────────────────────────────────┘ └─────────┘
```

## AI pipeline (per request)

```
Upload (PDF/DOCX/TXT)
   ↓ validate: extension, size, decodability, non-empty text
Text cleaning (control chars, whitespace)
   ↓
Structured extraction ──────────────┐
   heuristic pass (sections, regex   │  (optional) LLM pass
   dates, contacts, projects)        │  strict JSON, merged under
   + taxonomy skill scan w/ evidence │  type validation; heuristic
   ↓                                │  result is the fallback
Merged profile (validated JSON)  ◄──┘
   ↓
Skill normalization (canonical ids, aliases, relatedness graph)
   ↓
Embedding generation (OpenAI → sentence-transformers → hashing fallback)
   ↓
Semantic matching (cosine: doc-level, experience↔responsibilities, projects↔requirements)
   ↓
Scoring engine (deterministic weighted components — see docs/scoring.md)
   ↓
Explanation layer (deterministic template always; optional LLM narration of computed facts)
   ↓
ATS analysis + recommendations (pure heuristics from the same evidence)
   ↓
Persistence (full breakdown stored) + vector indexing (resume + JD chunks)
```

## Deterministic vs probabilistic components

| Component | Approach | Why |
|---|---|---|
| File parsing | pdfminer.six / python-docx | deterministic |
| Skill detection | compiled taxonomy patterns | deterministic, auditable |
| Skill normalization + relatedness | curated alias map + graph | deterministic |
| Resume/JD structure extraction | section heuristics **+ optional LLM** | LLM adds recall on messy docs; heuristics guarantee a result |
| Match scoring (all numbers) | **deterministic only** | explainability, testability, no hallucinated scores |
| Semantic similarity | embeddings (pluggable backend) | probabilistic model, but deterministic given a backend |
| Explanation narration | template **+ optional LLM** | LLM phrasing on computed facts only |
| RAG answers | retrieval (deterministic top-k) + LLM or extractive fallback | grounding via citations either way |
| ATS checks | heuristics | deterministic |
| Recommendations | rules over computed gaps | deterministic |

## Database schema (SQLite via SQLAlchemy)

- `resumes` — raw text, parse method, structured profile JSON, warnings
- `jobs` — JD text + parsed requirements JSON, linked to a resume
- `analyses` — overall score, full component breakdown, skill comparison,
  ATS results, recommendations, explanation + method (heuristic/hybrid)
- `chat_messages` — conversation history + retrieved chunk citations per analysis

Why SQLite: zero-config local setup and single-container Docker, while remaining a
real relational schema. The ORM keeps swapping to Postgres a one-line change
(`DATABASE_URL`).

## Vector store

In-process, per-resume namespace:

- Paragraph-aware chunking (~650 chars) of resume + all its JDs
- Embeddings via the active backend; cached to disk (`backend_data/vector_store/`)
- Backend label stored with each index — auto-rebuild when the embedding space changes
- Top-k cosine retrieval with source labels (RESUME/JOB) and similarity scores

A dedicated vector-DB server was deliberately not used: at this document scale it
would add infrastructure without adding engineering value. The retrieval concepts
(chunking, embedding, top-k, citation) are identical.

## Security notes

- No keys in code; `.env.example` documents every variable
- Upload validation: extension allow-list, size cap, decode checks, empty-file and
  fake-file rejection (e.g. renamed .doc)
- LLM prompt-injection defense: documents are passed as data, system prompts
  explicitly instruct the model to ignore embedded instructions, and payloads are
  length-capped
- LLM output is JSON-validated and merged under type checks — never blindly trusted
- CORS restricted to known dev origins
- Resumes are personal data: stored locally only, never sent anywhere except the
  configured LLM endpoint (opt-in via API key)
