# Scoring Methodology

The match score is **fully deterministic** — an LLM is never asked to produce or
modify a number. Every sub-score returns both a value and the evidence that
produced it, which is what the UI displays under "Why this score".

## Overall formula

```
Overall = Σ( effective_weight_i × score_i )

effective_weight_i = base_weight_i,                       if component i is informative
                   = 0 (renormalized across informative), if component i has no signal
```

Base weights (defined in `backend/app/services/matching/scoring.py::WEIGHTS`):

```
Skills 0.35 · Experience 0.20 · Projects 0.15 · Education 0.10 · Semantic 0.20
```

## Dynamic weight redistribution (the "watchman fix")

A component is **informative** only when the job description actually contains
that signal:

| Component | Informative when… | Neutral fallback (non-informative) |
|---|---|---|
| Skills | JD requirement bullets contain non-soft taxonomy skills | 0.5 + full-JD fallback scan; if still nothing, 0.5 |
| Experience | JD states a years requirement OR has responsibilities | 0.5 years-neutral |
| Projects | JD text exists to compare against | — (always informative when projects exist) |
| Education | JD names a recognizable degree level | 0.7 |
| Semantic | always (document-level similarity is always real) | — |

**Why:** the original engine gave a requirement-less JD (e.g. a watchman posting)
free neutral credit everywhere — a tech resume scored **68%** against it. With
redistribution, that JD is scored ONLY on what's real (responsibilities,
projects, semantic relevance) and lands at **~16%**. A fully-specified tech JD
keeps the exact base weights and scores ~60%.

## 1. Skills (35%)

Required vs preferred skills (from JD bullet extraction; **soft skills are
excluded entirely** — "good communication skills" is boilerplate in ~90% of
postings and would let a resume listing "communication" score off an unrelated
JD). Sparse-JD fallback: if no requirement bullets yield skills, the whole JD
text is scanned for non-soft taxonomy skills.

| Situation | Credit |
|---|---|
| Exact taxonomy match | 1.0 |
| Related skill (taxonomy graph, TensorFlow↔PyTorch) | 0.6 × category factor |
| Absent | 0.0 |

Weighted average, **required 3× vs preferred 1×**.

## 2. Experience (20%)

`0.6 × years + 0.4 × relevance`:

- **years**: satisfied → 1.0; partial → have/required; unknown → 0.4;
  no requirement → 0.5 (neutral)
- **relevance**: calibrated similarity between the candidate's experience text
  and the JD's responsibilities (short-text anchors — see below)

## 3. Projects (15%)

`0.4 × best + 0.2 × mean + 0.4 × skill_coverage` (coverage weight dropped and
the rest renormalized when the JD has no recognizable skills):

- **best/mean**: calibrated cosine between project texts and the JD's requirement
  bullets (or the full JD text for sparse JDs)
- **skill_coverage**: fraction of required skills mentioned inside projects —
  deterministic and the most reliable signal at short-text scale

## 4. Education (10%)

Degrees map to levels (high school 1 → diploma 2 → bachelor 3 → master 4 → PhD 5).

- candidate level ≥ required → 1.0
- candidate level below → candidate/required
- requirement names no recognizable degree ("8th pass", "any graduate") → 0.7 neutral
- no requirement → 0.7 neutral

## 5. Semantic (20%)

Raw cosine similarity between the full resume and full JD text, calibrated with
backend-specific linear anchors (measured on labeled pairs):

| Backend | anchors (lo, hi) | unrelated raw | related raw |
|---|---|---|---|
| hashing:768 (offline default) | 0.18, 0.50 | ~0.08–0.21 | ~0.22–0.45 |
| true embedding models (ST/OpenAI) | 0.30, 0.65 | ~0.10–0.30 | ~0.50–0.70 |

Short-text comparisons (projects, experience) use separate anchors
(hashing: 0.04–0.25) because lexical cosine runs much lower on short texts.

**Why raw cosine, not (cos+1)/2:** that normalization is for embeddings that can
go negative; the hashing backend's non-negative vectors would be silently
inflated (0.21 → 0.61) — the exact bug behind the inflated 46% semantic score.

## Calibration methodology

Anchors were measured with a calibration script comparing labeled related pairs
(resume ↔ its matching JD) against unrelated pairs (resume ↔ watchman/security
JD), with and without stopword filtering. Stopword filtering (~120 function
words) is what creates real separation: watchman 0.263 → 0.212, chef
0.192 → 0.083, while related pairs stay ≥ 0.22.

## Properties guaranteed by tests (`tests/test_watchman_regression.py` + others)

- Watchman JD vs tech resume scores **< 40%** (was 68%)
- Watchman vs ML role gap **> 25 points**
- Skills score **never** defaults to 100% on requirement-less JDs
- Sparse JD → weights redistributed (skills/education get 0 weight)
- Fully-specified JD → **exact** base weights, no redistribution
- Three different JDs → three different scores (no collapse)
- Weights always sum to 1.0; same input → same output
- Stopwords filtered from the hashing tokenizer; no false positives
  (`pythonic` ≠ `python`, "Security Guard" ≠ cybersecurity)

## What the LLM does NOT do

- Compute or adjust any score
- Decide skill matches
- Produce percentages

The LLM (when configured) only narrates the already-computed facts. Without an
API key the deterministic template explanation is used — identical numbers.
