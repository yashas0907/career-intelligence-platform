"""Semantic matching + scoring engine (fully deterministic).

Scoring methodology (documented in docs/scoring.md, verified by tests)
---------------------------------------------------------------------
Overall = weighted mix of Skills (0.35) / Experience (0.20) / Projects (0.15) /
Education (0.10) / Semantic (0.20) — WITH DYNAMIC WEIGHT REDISTRIBUTION:

* Every component reports `informative` — False when it had to fall back to a
  neutral default because the job description lacks that signal entirely
  (e.g. a watchman posting with no recognizable skill requirements).
* Weight from non-informative components is REDISTRIBUTED to the informative
  ones (renormalized). A skill-sparse JD is therefore scored mainly on
  semantic/project relevance instead of collecting free neutral credit —
  a tech resume vs watchman lands ~15-25%, a real ML role stays ~70%.
* When ALL components are informative the weights are exactly the base weights.

Integrity rules (the "watchman bug" fixes, all test-verified):
* No-requirement JDs NEVER get free 100% on skills — neutral 0.5 max, and the
  component is marked non-informative.
* The hashing backend filters ~120 English stopwords so unrelated documents
  don't inherit cosine from generic words.
* Semantic anchors are calibrated per backend (see docs/scoring.md).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.services.ai.embeddings import get_backend, semantic_similarity
from app.services.skills import normalizer, related_skills, skill_category, skill_display

logger = logging.getLogger("app.scoring")

WEIGHTS = {
    "skills": 0.35,
    "experience": 0.20,
    "projects": 0.15,
    "education": 0.10,
    "semantic": 0.20,
}
REQUIRED_WEIGHT = 3.0
PREFERRED_WEIGHT = 1.0
TRANSFERABLE_CREDIT = 0.6

CATEGORY_PENALTY_BONUS = {
    # required skill categories the resume can partially satisfy via related stacks
    "language": 1.0,
    "framework": 0.9,
    "library": 0.9,
    "tool": 0.9,
    "cloud": 0.8,
    "database": 0.9,
    "concept": 0.8,
    "domain": 0.5,
    "soft": 0.7,
}


@dataclass
class SkillMatchDetail:
    skill: str
    display: str
    category: str
    requirement: str  # required | preferred
    status: str  # exact | transferable | missing
    score: float
    via: str | None = None  # related skill that provided transferable credit
    evidence: list[str] = field(default_factory=list)


def _candidate_skills(profile: dict[str, Any]) -> dict[str, dict]:
    return profile.get("skills") or {}


def _job_skills(parsed_jd: dict[str, Any]) -> tuple[list[str], list[str]]:
    required = [s for s in (parsed_jd.get("required_skills") or []) if s]
    preferred = [s for s in (parsed_jd.get("preferred_skills") or []) if s]
    # de-dup, preserving order
    required = list(dict.fromkeys(required))
    preferred = [s for s in dict.fromkeys(preferred) if s not in required]
    return required, preferred


def _project_texts(profile: dict[str, Any]) -> list[str]:
    out = []
    for p in profile.get("projects") or []:
        name = p.get("name") or ""
        desc = " ".join(p.get("description") or []) if isinstance(p.get("description"), list) else str(p.get("description") or "")
        text = f"{name}. {desc}".strip(". ")
        if text:
            out.append(text)
    return out


def _experience_text(profile: dict[str, Any]) -> str:
    parts: list[str] = []
    for e in profile.get("experience") or []:
        title = e.get("title") or ""
        company = e.get("company") or ""
        bullets = " ".join(e.get("bullets") or []) if isinstance(e.get("bullets"), list) else ""
        parts.append(f"{title} at {company}. {bullets}".strip())
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Skill comparison
# ---------------------------------------------------------------------------

def compare_skills(profile: dict[str, Any], parsed_jd: dict[str, Any], jd_text: str | None = None) -> dict[str, Any]:
    cand = _candidate_skills(profile)
    required, preferred = _job_skills(parsed_jd)
    skills_source = "bullets"

    # Sparse-JD fallback: when bullet extraction found NO skill requirements,
    # scan the whole JD text for non-soft taxonomy skills and score against
    # those (honest overlap). Soft skills alone never count — a resume that
    # lists "communication" must not earn a 100% skills score off a watchman JD.
    if not required and not preferred and jd_text:
        found = set(normalizer.extract(jd_text).keys())
        found = {s for s in found if skill_category(s) != "soft"}
        if found:
            required = sorted(found)
            skills_source = "full_text"

    if not required and not preferred:
        # No recognizable skill requirements at all -> NEUTRAL, never free 100%.
        return {
            "score": 0.5,
            "informative": False,
            "skills_source": "none",
            "details": [],
            "strong": [],
            "transferable": [],
            "missing_required": [],
            "missing_preferred": [],
            "note": "No recognizable skill requirements in this job description — neutral skills credit; the match relies on overall relevance.",
        }

    details: list[SkillMatchDetail] = []
    strong: list[SkillMatchDetail] = []
    transferable: list[SkillMatchDetail] = []
    missing_required: list[SkillMatchDetail] = []
    missing_preferred: list[SkillMatchDetail] = []

    for skill, weight_kind in [(s, "required") for s in required] + [(s, "preferred") for s in preferred]:
        detail = SkillMatchDetail(
            skill=skill,
            display=skill_display(skill),
            category=skill_category(skill),
            requirement=weight_kind,
            status="missing",
            score=0.0,
        )
        if skill in cand:
            ev = cand[skill].get("evidence") or []
            detail.status = "exact"
            detail.score = 1.0
            detail.evidence = [f"Found in resume: “{ev[0]}”"] if ev else ["Detected in resume text"]
            strong.append(detail)
        else:
            # transferable via relatedness
            rel = related_skills(skill)
            hits = sorted(rel & set(cand.keys()))
            if hits:
                via = hits[0]
                detail.status = "transferable"
                detail.score = TRANSFERABLE_CREDIT * CATEGORY_PENALTY_BONUS.get(detail.category, 0.8)
                detail.via = via
                detail.evidence = [
                    f"No direct “{skill_display(skill)}”, but has related “{skill_display(via)}” (partial credit {detail.score:.0%})"
                ]
                transferable.append(detail)
            else:
                ev = [f"“{skill_display(skill)}” ({detail.requirement}) not found in resume"]
                detail.evidence = ev
                (missing_required if weight_kind == "required" else missing_preferred).append(detail)
        details.append(detail)

    # weighted skills score
    total_weight = 0.0
    total_score = 0.0
    for d in details:
        w = REQUIRED_WEIGHT if d.requirement == "required" else PREFERRED_WEIGHT
        total_weight += w
        total_score += w * d.score
    skills_score = total_score / total_weight if total_weight else 1.0

    return {
        "score": round(skills_score, 4),
        "informative": True,
        "skills_source": skills_source,
        "details": [d.__dict__ for d in details],
        "strong": [d.__dict__ for d in strong],
        "transferable": [d.__dict__ for d in transferable],
        "missing_required": [d.__dict__ for d in missing_required],
        "missing_preferred": [d.__dict__ for d in missing_preferred],
    }


# ---------------------------------------------------------------------------
# Experience score
# ---------------------------------------------------------------------------

def _score_years(profile: dict[str, Any], parsed_jd: dict[str, Any]) -> tuple[float, list[str]]:
    req = parsed_jd.get("min_years_experience")
    have = profile.get("total_years_experience")
    if not req:
        return 0.5, ["No explicit experience requirement — neutral credit (50%)"]
    if have is None:
        return 0.4, [f"Job requires ~{req:g} yrs; resume experience length could not be determined (40%)"]
    if have >= req:
        return 1.0, [f"Experience: {have:g} yrs detected vs {req:g} required — satisfied"]
    ratio = max(0.0, have / max(req, 0.1))
    return ratio, [f"Experience: {have:g} yrs detected vs {req:g} required — partial ({ratio:.0%})"]


def score_experience(profile: dict[str, Any], parsed_jd: dict[str, Any], resume_text: str) -> dict[str, Any]:
    req = parsed_jd.get("min_years_experience")
    has_resp = bool(parsed_jd.get("responsibilities"))
    years_score, years_ev = _score_years(profile, parsed_jd)
    # Informative when the JD states a years requirement OR has real
    # responsibilities to compare against; fully neutral only when it has neither.
    informative = req is not None or has_resp
    resp_text = " ".join(parsed_jd.get("responsibilities") or []) or parsed_jd.get("title") or ""
    exp_text = _experience_text(profile) or resume_text[:1500]
    # exp↔resp is a short-text comparison — same calibrated anchors as projects
    # (raw cosine on the hashing backend runs low even for related pairs)
    if resp_text:
        raw_sem = semantic_similarity(exp_text, resp_text)
        lo, hi = _short_text_anchors()
        sem = max(0.0, min(1.0, (raw_sem - lo) / (hi - lo)))
    else:
        sem = 0.5
    score = 0.6 * years_score + 0.4 * sem
    return {
        "score": round(score, 4),
        "informative": informative,
        "years": round(years_score, 4),
        "semantic": round(sem, 4),
        "evidence": years_ev + [f"Experience↔responsibilities relevance: {sem:.0%}"],
    }


# ---------------------------------------------------------------------------
# Projects score
# ---------------------------------------------------------------------------

def _short_text_anchors() -> tuple[float, float]:
    """Anchors for SHORT-text cosine (project ↔ JD bullets).

    Calibrated on labeled pairs (docs/scoring.md): related ~0.08-0.12 raw,
    unrelated ~0.04-0.10 raw on the hashing backend. The lexical cosine is
    weakly discriminative at this scale, so the projects formula leans on the
    deterministic skill-coverage signal (weight 0.4) over the noisy cosine.
    """
    backend_name = get_backend().name
    if backend_name.startswith("hashing"):
        return 0.04, 0.25
    return 0.25, 0.55


def score_projects(profile: dict[str, Any], parsed_jd: dict[str, Any], jd_text: str | None = None) -> dict[str, Any]:
    projects = _project_texts(profile)
    if not projects:
        # A real signal for early-career candidates (not a neutral default)
        return {"score": 0.15, "informative": True, "evidence": ["No projects section detected — low default credit"]}
    req_text = " ".join(parsed_jd.get("required_bullets") or [])
    if not req_text:
        req_text = jd_text or ""  # sparse JD: compare against the whole job text
    if not req_text:
        return {"score": 0.6, "informative": False, "evidence": ["No job text to compare projects against"]}

    lo, hi = _short_text_anchors()

    backend = get_backend()
    proj_vecs = backend.embed(projects)
    req_vec = backend.embed([req_text])[0]
    proj_vecs = proj_vecs / (np_norm(proj_vecs, axis=1, keepdims=True) + 1e-9)
    req_vec = req_vec / (np_norm(req_vec) + 1e-9)
    # raw cosine clamped to [0,1] — no (cos+1)/2 remap: the hashing backend's
    # non-negative vectors would be silently inflated by it
    sims = np.clip(proj_vecs @ req_vec, 0.0, 1.0)
    best_idx = int(sims.argmax())
    best = max(0.0, min(1.0, (float(sims[best_idx]) - lo) / (hi - lo)))
    mean = max(0.0, min(1.0, (float(sims.mean()) - lo) / (hi - lo)))

    # project skill coverage: fraction of required skills mentioned in projects
    from app.services.skills import normalizer as _n

    req_skills = parsed_jd.get("required_skills") or []
    if req_skills:
        proj_text_all = " ".join(projects)
        in_proj = _n.extract(proj_text_all)
        cover = len(set(req_skills) & set(in_proj.keys())) / len(req_skills)
        cover_weight = 0.4
    else:
        cover = 0.0
        cover_weight = 0.0  # no recognizable skills -> coverage undefined, skip it

    score = (0.4 * best + 0.2 * mean + cover_weight * cover) / (0.6 + cover_weight)
    return {
        "score": round(score, 4),
        "informative": True,
        "best_project": projects[best_idx][:120],
        "best_similarity": round(float(sims[best_idx]), 4),
        "skill_coverage": round(cover, 4),
        "evidence": [
            f"Best-matching project: “{projects[best_idx][:80]}…” ({best:.0%} calibrated similarity)",
            f"Required skills appearing inside projects: {cover:.0%}" if req_skills else "No recognizable skill requirements — scored on relevance",
        ],
    }


def np_norm(arr, axis=None, keepdims=False):
    import numpy as np

    return np.linalg.norm(arr, axis=axis, keepdims=keepdims)


# ---------------------------------------------------------------------------
# Education score
# ---------------------------------------------------------------------------

_DEGREE_LEVELS = {
    "high school": 1, "diploma": 2, "intermediate": 2,
    "bachelor": 3, "b.tech": 3, "btech": 3, "b.e": 3, "bsc": 3, "b.sc": 3, "bca": 3, "b.com": 3, "ba": 3,
    "master": 4, "m.tech": 4, "mtech": 4, "msc": 4, "m.sc": 4, "mca": 4, "mba": 4,
    "phd": 5, "ph.d": 5, "doctorate": 5,
}


def _max_degree_level(texts: list[str]) -> int:
    best = 0
    joined = " ".join(texts).lower()
    for kw, lvl in _DEGREE_LEVELS.items():
        if kw in joined:
            best = max(best, lvl)
    return best


def score_education(profile: dict[str, Any], parsed_jd: dict[str, Any]) -> dict[str, Any]:
    reqs = parsed_jd.get("education_requirements") or []
    cand_edu = profile.get("education") or []
    cand_texts = [str(e.get("raw") or e.get("degree") or "") for e in cand_edu if isinstance(e, dict)]

    if not reqs:
        # no requirement -> neutral, non-informative
        return {"score": 0.7, "informative": False, "evidence": ["No education requirement stated — neutral credit (70%)"]}

    req_level = max((_DEGREE_LEVELS.get(r.lower(), 0) for r in reqs), default=0)
    cand_level = _max_degree_level(cand_texts)

    if req_level == 0:
        # requirement exists but names no recognizable degree (e.g. "8th pass",
        # "any graduate") — meaningless for differentiation -> neutral
        return {
            "score": 0.7,
            "informative": False,
            "evidence": ["Education requirement doesn't specify a degree level — neutral credit (70%)"],
        }

    if cand_level >= req_level:
        return {
            "score": 1.0,
            "informative": True,
            "evidence": [f"Candidate degree level ({cand_level}) meets requirement ({req_level})"],
        }
    if cand_level > 0:
        ratio = cand_level / max(req_level, 1)
        return {
            "score": round(ratio, 4),
            "informative": True,
            "evidence": [f"Candidate degree level ({cand_level}) below required ({req_level}) — partial"],
        }
    return {
        "score": 0.3,
        "informative": True,
        "evidence": ["No degree information detected in resume vs requirement present"],
    }


# ---------------------------------------------------------------------------
# Semantic score
# ---------------------------------------------------------------------------

def _semantic_anchors() -> tuple[float, float]:
    """Linear-map anchors (lo, hi) for stretching raw doc-level cosine to [0,1].

    Calibrated per backend on labeled resume↔JD pairs (see docs/scoring.md):
    - hashing: stopword-filtered lexical cosine; unrelated ~0.08-0.21,
      related ~0.22-0.45 -> anchors chosen to separate them honestly.
    - true embedding models (ST/OpenAI): different cosine distribution.
    """
    backend_name = get_backend().name
    if backend_name.startswith("hashing"):
        return 0.18, 0.50
    return 0.30, 0.65


def score_semantic(resume_text: str, parsed_jd: dict[str, Any], jd_text: str) -> dict[str, Any]:
    sim = semantic_similarity(resume_text, jd_text)
    lo, hi = _semantic_anchors()
    stretched = max(0.0, min(1.0, (sim - lo) / (hi - lo)))
    return {
        "score": round(stretched, 4),
        "informative": True,
        "raw_similarity": round(sim, 4),
        "evidence": [f"Document-level semantic similarity: {sim:.0%} (calibrated to {stretched:.0%})"],
    }


# ---------------------------------------------------------------------------
# Overall engine
# ---------------------------------------------------------------------------

def analyze_match(profile: dict[str, Any], resume_text: str, parsed_jd: dict[str, Any], jd_text: str) -> dict[str, Any]:
    skills = compare_skills(profile, parsed_jd, jd_text)
    experience = score_experience(profile, parsed_jd, resume_text)
    projects = score_projects(profile, parsed_jd, jd_text)
    education = score_education(profile, parsed_jd)
    semantic = score_semantic(resume_text, parsed_jd, jd_text)

    components = {"skills": skills, "experience": experience, "projects": projects, "education": education, "semantic": semantic}

    # --- Dynamic weight redistribution ---
    # Components that fell back to neutral defaults (JD lacks that signal) carry
    # no information — their weight is redistributed to the informative ones.
    # A skill-sparse JD ("watchman") is thus scored on relevance only, while a
    # fully-specified JD keeps the exact base weights.
    informative = {k: components[k].get("informative", True) for k in WEIGHTS}
    effective = {k: (w if informative[k] else 0.0) for k, w in WEIGHTS.items()}
    total_eff = sum(effective.values())
    if total_eff > 0:
        effective = {k: w / total_eff for k, w in effective.items()}
    else:  # unreachable: semantic is always informative — safety net
        effective = dict(WEIGHTS)

    overall = sum(effective[k] * components[k]["score"] for k in effective)
    redistributed = any(not v for v in informative.values())

    return {
        "overall_score": round(overall, 4),
        "breakdown": {k: round(components[k]["score"], 4) for k in WEIGHTS},
        "weights": {k: round(effective[k], 4) for k in WEIGHTS},
        "base_weights": WEIGHTS,
        "weights_redistributed": redistributed,
        "components": components,
        "methodology": "See docs/scoring.md — weighted deterministic components with dynamic redistribution for sparse JDs; no LLM involvement in scoring.",
    }
