# Scoring Methodology

The match score is **fully deterministic** — an LLM is never asked to produce or
modify a number. Every sub-score returns both a value and the evidence that
produced it, which is what the UI displays under "Why this score".

## Overall formula

```
Overall = 0.35 × Skills + 0.20 × Experience + 0.15 × Projects
        + 0.10 × Education + 0.20 × Semantic
```

Weights are defined in `backend/app/services/matching/scoring.py::WEIGHTS`
and enforced by `tests/test_scoring.py` (they must sum to exactly 1.0).

---

## 1. Skills (35%)

Job skills are split into **required** and **preferred** (from JD bullet
extraction, plus the optional LLM pass). Each skill is compared against the
candidate's extracted skills (taxonomy-canonicalized):

| Situation | Credit |
|---|---|
| Exact taxonomy match (candidate has the skill) | 1.0 |
| Related skill present (taxonomy relatedness graph, e.g. TensorFlow↔PyTorch) | 0.6 × category factor* |
| Absent | 0.0 |

*Category factor: languages/frameworks/libraries/tools/databases 0.9, cloud 0.8,
concepts 0.8, domain 0.5, soft 0.7.

Weighted average with **required skills weighted 3× vs preferred 1×**:

```
skills_score = Σ(weight_i × credit_i) / Σ(weight_i)
```

### Example
Required = {python, pytorch, docker, sql, kubernetes} (5), preferred = {aws} (1).
Candidate has python, pytorch, docker, sql exactly; kubernetes missing; has
docker (related to kubernetes) but docker is already exact-matched to itself —
so kubernetes gets transferable credit via Docker.

```
skills = (3×(1+1+1+1+0.54) + 1×0) / (3×5 + 1) = 15.62/16 ≈ 0.976
```

(With AWS missing: preferred contributes 0.)

## 2. Experience (20%)

Two signals, blended 60/40:

- **Years signal (60%)**: required minimum vs detected total experience.
  - satisfied → 1.0
  - partial (have < required) → have/required
  - unknown candidate years → 0.40 (conservative, not punitive-zero)
  - no requirement stated → 0.75 (neutral)
- **Relevance signal (40%)**: cosine similarity between the candidate's
  experience section text and the JD's responsibility bullets (embedding backend).

## 3. Projects (15%)

For early-career candidates this is often the strongest real signal.

```
projects = 0.5 × best_project_similarity + 0.3 × mean_similarity + 0.2 × skill_coverage
```

- **best_project_similarity**: max cosine similarity between any single resume
  project text and the JD requirement text (the "best evidence" project is cited).
- **mean_similarity**: average across all projects.
- **skill_coverage**: fraction of required skills mentioned *inside projects*.
- No projects section → 0.15 (low default, triggers a "build a project" recommendation).

## 4. Education (10%)

Degrees are mapped to levels (high school 1 → diploma 2 → bachelor 3 →
master 4 → PhD 5).

- candidate level ≥ required level → 1.0
- candidate level below required → candidate_level / required_level
- no education detected but requirement exists → 0.30
- no requirement stated → 0.8 (0.6 if resume also lacks education info)

## 5. Semantic (20%)

Cosine similarity between the *entire* resume text and the *entire* JD text
using the configured embedding backend. Raw document-level cosine tends to
concentrate in 0.5–0.85, so it's linearly stretched:

```
stretched = clamp((cos - 0.35) / 0.55, 0, 1)
```

---

## Why these weights?

- **Skills 35%** — the most decision-relevant, most objectively measurable signal.
- **Experience 20% + Semantic 20%** — experience years are noisy (interns,
  career switchers); semantic similarity captures stack/domain overlap that
  skill lists miss. Together they balance precision and fuzziness.
- **Projects 15%** — critical for students, but only meaningful when present.
- **Education 10%** — usually a binary gate, weakly differentiating beyond that.

## Properties guaranteed by tests

- Weights sum to exactly 1.0 (`test_weights_sum_to_one`)
- A strictly better candidate never scores lower (`test_better_candidate_scores_higher`)
- Same input → same output, byte-identical (`test_same_input_same_output`)
- Exact formula verification for the required-3× weighting
  (`test_required_weights_3x`)
- Transferable credit is strictly between 0 and 1 (`test_transferable_partial_credit`)
- Every skill decision carries an evidence string (`test_evidence_always_present`)

## What the LLM does NOT do

- Compute or adjust any score
- Decide skill matches
- Produce percentages

The LLM (when configured) only narrates the already-computed facts in the
explanation step, and its prompt forbids inventing numbers. Without an API key
the deterministic template explanation is used — the numbers are identical.
