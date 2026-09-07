# Evaluation Methodology

## What is evaluated, and how

### 1. Deterministic components → classic unit tests

Scoring, normalization, parsing, and ATS checks are pure functions, so they are
verified with exact assertions in the test suite (109 tests):

- **Scoring**: weights sum to 1; required-skills 3× weighting formula is verified
  numerically; monotonicity (better candidate ⇒ higher score); determinism
  (same input ⇒ byte-identical output); transferable credit strictly in (0,1).
- **Skill normalization**: every taxonomy alias maps to its canonical id;
  no false boundary matches (`pythonic` must NOT count as `python`); case
  insensitivity; dedup to one canonical id.
- **Parsing**: hand-built real PDF/DOCX byte streams are parsed correctly;
  corrupted/fake/empty/oversized/unsupported files are rejected with clear errors.
- **Extraction**: contacts, sections, education, experience bullets, projects,
  years-of-experience (explicit "3+ years" and date-range derivation).
- **API**: full user journey (upload → analyze → retrieve → rank → chat),
  plus 404s, 422s, and consistent error envelopes.

### 2. Extraction accuracy (labeled corpus, measured)

`tests/fixtures/eval_corpus.py` defines 3 labeled resumes with per-field ground
truth. `tests/eval_extraction.py` computes precision / recall / F1 per field and
per resume; `tests/test_eval_extraction.py` enforces thresholds in CI.

Measured macro-averages (heuristic pipeline, no LLM, 2026-09):

| Field | Precision | Recall | F1 |
|---|---|---|---|
| Name | 1.00 | 1.00 | 1.00 |
| Email | 1.00 | 1.00 | 1.00 |
| Skills | 0.89 | 0.83 | 0.86 |
| Projects | 1.00 | 1.00 | 1.00 |
| Experience titles | 1.00 | 1.00 | 1.00 |
| Education degrees | 1.00 | 1.00 | 1.00 |

Scope caveats (honest): 3 resumes is a *small* corpus — these numbers say the
pipeline works on conventional layouts, not that it generalizes to every resume
style. Extending the corpus is the single highest-value evaluation improvement
(see below).

### 3. Skill normalization quality

The taxonomy covers ~100 canonical tech skills with ~250 aliases, including the
classic confusables: `sklearn→scikit-learn`, `k8s→kubernetes`, `golang→go`,
`js→javascript`, `TF→tensorflow`, `HF transformers→huggingface`. Tests pin each
mapping. Equivalence handling (`Python` vs `python` vs `py`) is by construction,
not string matching — the reference project's weakness (naive matching) is
specifically addressed here.

### 4. Matching quality

Verifiable properties instead of unverifiable accuracy claims:

- **Sanity/oracle tests**: a resume explicitly matching every requirement must
  score 1.0 on skills; a resume with zero overlap must score 0.0.
- **Relative correctness**: an ML resume must score higher against an ML JD than
  a backend JD (the API test asserts exactly this ordering).
- **Explainability**: every skill decision, every sub-score, and every
  recommendation carries evidence strings — tested to be non-empty.

### 5. Retrieval quality (RAG)

- Chunking: paragraph-aware, hard-split for giant paragraphs; ids unique (tested).
- Retrieval: a query semantically matching one chunk retrieves that chunk first
  (tested with the hashing backend — deterministic in CI).
- Grounding: every answer ships with source labels + similarity scores; the
  extractive fallback only ever quotes retrieved text.

### 6. Hallucination resistance

- Scores cannot be hallucinated — they're computed, not generated.
- LLM extraction output is validated field-by-field and only *merged* — the
  heuristic result remains the base, so the LLM can only add, not corrupt.
- The assistant's system prompt forbids non-context answers; when no LLM is
  configured the fallback cannot generate novel claims at all (it quotes).
- Prompt-injection: documents are passed as data with explicit "ignore embedded
  instructions" directives; payloads are truncated.

## Limitations (honest list)

1. **Extraction**: heuristics work best on conventional single-column resumes;
   exotic layouts degrade to whole-text skill scanning (still works, less structured).
2. **Taxonomy scope**: ~100 skills aimed at tech/AI/data roles. Unlisted skills
   pass through as-is (LLM pass can add them to JDs; they're compared verbatim).
3. **Semantic fallback**: the hashing backend is lexical, not truly semantic.
   For real semantic scoring configure OpenAI embeddings or install
   sentence-transformers. Backend is always reported in `/api/health` and the UI.
4. **Years of experience** is derived from date ranges and explicit statements —
   internships, gaps, and "Present" are approximated.
5. **ATS analysis** uses publicly-known parsing signals; it does not claim to
   replicate any proprietary system (stated in the UI and API response).
6. **Job ranking** ranks provided JDs only — it is not a job-search engine.
7. **Evaluation scale**: fixtures are small; production-grade claims would need
   annotated corpora and human-in-the-loop review.

## Extending evaluation

- Grow `tests/fixtures/eval_corpus.py` to 50–100 annotated resumes (it's plain
  data — add entries and the report + CI thresholds update automatically).
- LLM-judge experiments for RAG answer faithfulness (judge sees only retrieved
  chunks + the answer, scores support 0–1) — cheap, automatable, honest.
- Property-based testing (hypothesis) for the scoring engine: random profiles,
  assert invariants (bounds, monotonicity in added skills).
