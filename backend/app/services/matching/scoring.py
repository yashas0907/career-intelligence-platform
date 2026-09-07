"""Semantic matching + scoring engine (fully deterministic).

Scoring methodology (documented in docs/scoring.md, verified by tests)
---------------------------------------------------------------------
Overall = 0.35*Skills + 0.20*Experience + 0.15*Projects + 0.10*Education + 0.20*Semantic

* **Skills (35%)** — required skills weighted 3x vs preferred (1x). Exact taxonomy
  match scores 1.0; a *related* skill (taxonomy relatedness graph, e.g.
  tensorflow vs pytorch) scores TRANSFERABLE_CREDIT (0.6); scaled by ratio matched.
* **Experience (20%)** — combines (a) years gap vs required, and (b) semantic
  similarity between resume experience section and JD responsibilities.
* **Projects (15%)** — max semantic similarity between any resume project text and
  the JD requirement bullets (best-project evidence), averaged with skill overlap
  within projects.
* **Education (10%)** — binary compatibility between candidate degrees and required
  degree keywords, with partial credit for any-vs-missing requirements.
* **Semantic (20%)** — cosine similarity (embedding backend) between the whole
  resume text and the whole JD text, clamped and scaled.

Every sub-score returns (score, evidence[]) so the UI can fully explain results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.services.ai.embeddings import get_backend, semantic_similarity
from app.services.skills import related_skills, skill_category, skill_display

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

def compare_skills(profile: dict[str, Any], parsed_jd: dict[str, Any]) -> dict[str, Any]:
    cand = _candidate_skills(profile)
    required, preferred = _job_skills(parsed_jd)

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
        return 0.75, ["No explicit experience requirement — neutral credit (75%)"]
    if have is None:
        return 0.4, [f"Job requires ~{req:g} yrs; resume experience length could not be determined (40%)"]
    if have >= req:
        return 1.0, [f"Experience: {have:g} yrs detected vs {req:g} required — satisfied"]
    ratio = max(0.0, have / max(req, 0.1))
    return ratio, [f"Experience: {have:g} yrs detected vs {req:g} required — partial ({ratio:.0%})"]


def score_experience(profile: dict[str, Any], parsed_jd: dict[str, Any], resume_text: str) -> dict[str, Any]:
    years_score, years_ev = _score_years(profile, parsed_jd)
    resp_text = " ".join(parsed_jd.get("responsibilities") or []) or parsed_jd.get("title") or ""
    exp_text = _experience_text(profile) or resume_text[:1500]
    sem = semantic_similarity(exp_text, resp_text) if resp_text else 0.5
    score = 0.6 * years_score + 0.4 * sem
    return {
        "score": round(score, 4),
        "years": round(years_score, 4),
        "semantic": round(sem, 4),
        "evidence": years_ev + [f"Experience↔responsibilities semantic similarity: {sem:.0%}"],
    }


# ---------------------------------------------------------------------------
# Projects score
# ---------------------------------------------------------------------------

def score_projects(profile: dict[str, Any], parsed_jd: dict[str, Any]) -> dict[str, Any]:
    projects = _project_texts(profile)
    if not projects:
        return {"score": 0.15, "evidence": ["No projects section detected — low default credit"]}
    req_text = " ".join(parsed_jd.get("required_bullets") or []) or " ".join(
        parsed_jd.get("required_skills") or []
    )
    if not req_text:
        return {"score": 0.6, "evidence": ["No specific requirements to compare projects against"]}

    backend = get_backend()
    proj_vecs = backend.embed(projects)
    req_vec = backend.embed([req_text])[0]
    proj_vecs = proj_vecs / (np_norm(proj_vecs, axis=1, keepdims=True) + 1e-9)
    req_vec = req_vec / (np_norm(req_vec) + 1e-9)
    sims = (proj_vecs @ req_vec + 1) / 2  # to [0,1]
    best_idx = int(sims.argmax())
    best = float(sims[best_idx])
    mean = float(sims.mean())

    # project skill coverage: fraction of required skills mentioned in projects
    from app.services.skills import normalizer as _n

    req_skills = parsed_jd.get("required_skills") or []
    if req_skills:
        proj_text_all = " ".join(projects)
        in_proj = _n.extract(proj_text_all)
        cover = len(set(req_skills) & set(in_proj.keys())) / len(req_skills)
    else:
        cover = 0.5

    score = 0.5 * best + 0.3 * mean + 0.2 * cover
    return {
        "score": round(score, 4),
        "best_project": projects[best_idx][:120],
        "best_similarity": round(best, 4),
        "skill_coverage": round(cover, 4),
        "evidence": [
            f"Best-matching project: “{projects[best_idx][:80]}…” ({best:.0%} semantic similarity)",
            f"Required skills appearing inside projects: {cover:.0%}",
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
        # no requirement -> neutral-high
        return {"score": 0.8 if cand_texts else 0.6, "evidence": ["No education requirement stated"]}

    req_level = max((_DEGREE_LEVELS.get(r.lower(), 0) for r in reqs), default=0)
    cand_level = _max_degree_level(cand_texts)

    if cand_level >= req_level and cand_level > 0:
        return {"score": 1.0, "evidence": [f"Candidate degree level ({cand_level}) meets requirement ({req_level})"]}
    if cand_level > 0:
        ratio = cand_level / max(req_level, 1)
        return {"score": round(ratio, 4), "evidence": [f"Candidate degree level ({cand_level}) below required ({req_level}) — partial"]}
    return {"score": 0.3, "evidence": ["No degree information detected in resume vs requirement present"]}


# ---------------------------------------------------------------------------
# Semantic score
# ---------------------------------------------------------------------------

def score_semantic(resume_text: str, parsed_jd: dict[str, Any], jd_text: str) -> dict[str, Any]:
    sim = semantic_similarity(resume_text, jd_text)
    # stretch: raw doc-level cosine tends to concentrate 0.5-0.85; map linearly
    stretched = max(0.0, min(1.0, (sim - 0.35) / 0.55))
    return {
        "score": round(stretched, 4),
        "raw_similarity": round(sim, 4),
        "evidence": [f"Document-level semantic similarity: {sim:.0%} (stretched to {stretched:.0%})"],
    }


# ---------------------------------------------------------------------------
# Overall engine
# ---------------------------------------------------------------------------

def analyze_match(profile: dict[str, Any], resume_text: str, parsed_jd: dict[str, Any], jd_text: str) -> dict[str, Any]:
    skills = compare_skills(profile, parsed_jd)
    experience = score_experience(profile, parsed_jd, resume_text)
    projects = score_projects(profile, parsed_jd)
    education = score_education(profile, parsed_jd)
    semantic = score_semantic(resume_text, parsed_jd, jd_text)

    components = {"skills": skills, "experience": experience, "projects": projects, "education": education, "semantic": semantic}
    overall = sum(WEIGHTS[k] * components[k]["score"] for k in WEIGHTS)

    return {
        "overall_score": round(overall, 4),
        "breakdown": {k: round(components[k]["score"], 4) for k in WEIGHTS},
        "weights": WEIGHTS,
        "components": components,
        "methodology": "See docs/scoring.md — weighted deterministic components, no LLM involvement in scoring.",
    }
