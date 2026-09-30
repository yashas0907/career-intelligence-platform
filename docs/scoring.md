# Scoring Methodology

The match score is **fully deterministic** â€” an LLM is never asked to produce or
modify a number. Every sub-score returns both a value and the evidence that
produced it, which is what the UI displays under "Why this score".

## Overall formula

```
Overall = Î£( effective_weight_i Ã— score_i )

effective_weight_i = base_weight_i,                       if component i is informative
                   = 0 (renormalized across informative), if component i has no signal
```

Base weights (defined in `backend/app/services/matching/scoring.py::WEIGHTS`):

```
Skills 0.35 Â· Experience 0.20 Â· Projects 0.15 Â· Education 0.10 Â· Semantic 0.20
```

## Dynamic weight redistribution (the "watchman fix")

A component is **informative** only when the job description actually contains
that signal:

| Component | Informative whenâ€¦ | Neutral fallback (non-informative) |
|---|---|---|
| Skills | JD requirement bullets contain non-soft taxonomy skills | 0.5 + full-JD fallback scan; if still nothing, 0.5 |
| Experience | JD states a years requirement OR has responsibilities | 0.5 years-neutral |
| Projects | JD text exists to compare against | â€” (always informative when projects exist) |
| Education | JD names a recognizable degree level | 0.7 |
| Semantic | always (document-level similarity is always real) | â€” |

**Why:** the original engine gave a requirement-less JD (e.g. a watchman posting)
free neutral credit everywhere â€” a tech resume scored **68%** against it. With
redistribution, that JD is scored ONLY on what's real (responsibilities,
projects, semantic relevance) and lands at **~16%**. A fully-specified tech JD
keeps the exact base weights and scores ~60%.

## 1. Skills (35%)

**Thin JDs (role titles / one-liners):** when a JD has no explicit requirements, the
expected skill set is INFERRED from the role title via a curated role-keyword table
(`_ROLE_SKILL_SETS`: web/frontend/backend/data/ML/cyber/devops/mobile/AI roles). Title-inferred
and full-text-scanned requirements are lower-confidence than stated ones — the skills
score is capped by 0.85 for either. A 'web developer' one-liner vs a web-skilled resume
therefore scores ~64% instead of collapsing to neutral filler.

Required vs preferred skills (from JD bullet extraction; **soft skills are
excluded entirely** â€” "good communication skills" is boilerplate in ~90% of
postings and would let a resume listing "communication" score off an unrelated
JD). Sparse-JD fallback: if no requirement bullets yield skills, the whole JD
text is scanned for non-soft taxonomy skills.

| Situation | Credit |
|---|---|
| Exact taxonomy match | 1.0 |
| Related skill (taxonomy graph, TensorFlowâ†”PyTorch) | 0.6 Ã— category factor |
| Absent | 0.0 |

Weighted average, **required 3Ã— vs preferred 1Ã—**.

## 2. Experience (20%)

`0.6 Ã— years + 0.4 Ã— relevance`:

- **years**: satisfied â†’ 1.0; partial â†’ have/required; unknown â†’ 0.4;
  no requirement â†’ 0.5 (neutral)
- **relevance**: calibrated similarity between the candidate's experience text
  and the JD's responsibilities (short-text anchors â€” see below)

## 3. Projects (15%)

`0.4 Ã— best + 0.2 Ã— mean + 0.4 Ã— skill_coverage` (coverage weight dropped and
the rest renormalized when the JD has no recognizable skills):

- **best/mean**: calibrated cosine between project texts and the JD's requirement
  bullets (or the full JD text for sparse JDs)
- **skill_coverage**: fraction of required skills mentioned inside projects â€”
  deterministic and the most reliable signal at short-text scale

## 4. Education (10%)

Degrees map to levels (high school 1 â†’ diploma 2 â†’ bachelor 3 â†’ master 4 â†’ PhD 5).

- candidate level â‰¥ required â†’ 1.0
- candidate level below â†’ candidate/required
- requirement names no recognizable degree ("8th pass", "any graduate") â†’ 0.7 neutral
- no requirement â†’ 0.7 neutral

## 5. Semantic (20%)

Raw cosine similarity between the full resume and full JD text, calibrated with
backend-specific linear anchors (measured on labeled pairs):

| Backend | anchors (lo, hi) | unrelated raw | related raw |
|---|---|---|---|
| hashing:768 (offline default) | 0.18, 0.50 | ~0.08â€“0.21 | ~0.22â€“0.45 |
| true embedding models (ST/OpenAI) | 0.30, 0.65 | ~0.10â€“0.30 | ~0.50â€“0.70 |

Short-text comparisons (projects, experience) use separate anchors
(hashing: 0.04â€“0.25) because lexical cosine runs much lower on short texts.

**Why raw cosine, not (cos+1)/2:** that normalization is for embeddings that can
go negative; the hashing backend's non-negative vectors would be silently
inflated (0.21 â†’ 0.61) â€” the exact bug behind the inflated 46% semantic score.

## Calibration methodology

Anchors were measured with a calibration script comparing labeled related pairs
(resume â†” its matching JD) against unrelated pairs (resume â†” watchman/security
JD), with and without stopword filtering. Stopword filtering (~120 function
words) is what creates real separation: watchman 0.263 â†’ 0.212, chef
0.192 â†’ 0.083, while related pairs stay â‰¥ 0.22.

## Properties guaranteed by tests (`tests/test_watchman_regression.py` + others)

- Watchman JD vs tech resume scores **< 40%** (was 68%)
- Watchman vs ML role gap **> 25 points**
- Skills score **never** defaults to 100% on requirement-less JDs
- Sparse JD â†’ weights redistributed (skills/education get 0 weight)
- Fully-specified JD â†’ **exact** base weights, no redistribution
- Three different JDs â†’ three different scores (no collapse)
- Weights always sum to 1.0; same input â†’ same output
- Stopwords filtered from the hashing tokenizer; no false positives
  (`pythonic` â‰  `python`, "Security Guard" â‰  cybersecurity)

## What the LLM does NOT do

- Compute or adjust any score
- Decide skill matches
- Produce percentages

The LLM (when configured) only narrates the already-computed facts. Without an
API key the deterministic template explanation is used â€” identical numbers.

