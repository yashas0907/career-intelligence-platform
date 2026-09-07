"""High-level orchestration: ties parsing, extraction, scoring, ATS,
recommendations, explanation, persistence and vector indexing together."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from app.models.entities import Analysis, Job, Resume
from app.services.ai.vector_store import VectorStore, chunk_text
from app.services.ats import analyze_ats
from app.services.explanation import explain
from app.services.extractor import extract_job
from app.services.matching.scoring import analyze_match
from app.services.recommendations import build_recommendations, rank_jobs

logger = logging.getLogger("app.pipeline")


def _chunks_for_resume(resume: Resume, jobs: list[Job]) -> list:
    chunks = chunk_text(resume.raw_text, f"resume:{resume.id}", "resume", "Resume", max_chars=650)
    for job in jobs:
        chunks += chunk_text(job.raw_text, f"job:{job.id}", "job", job.title or "Role", max_chars=650)
    return chunks


def index_resume_context(db: Session, resume: Resume) -> VectorStore:
    """(Re)build the vector index for a resume + all its job descriptions."""
    jobs = db.query(Job).filter(Job.resume_id == resume.id).all()
    store = VectorStore(namespace=resume.id)
    store.build_from_chunks(_chunks_for_resume(resume, jobs))
    return store


def run_match(db: Session, resume: Resume, job: Job) -> Analysis:
    """Full deterministic match pipeline for one resume-job pair."""
    profile = resume.profile
    parsed_jd = job.parsed

    result = analyze_match(profile, resume.raw_text, parsed_jd, job.raw_text)
    ats = analyze_ats(profile, resume.raw_text, parsed_jd, resume.parse_method)
    recs = build_recommendations(result, profile, parsed_jd)
    job_title = job.title or "Role"
    explanation, method = explain(result, job_title)

    analysis = Analysis(
        resume_id=resume.id,
        job_id=job.id,
        overall_score=result["overall_score"],
        breakdown={
            "overall": result["overall_score"],
            **result["breakdown"],
        },
        skill_comparison=result["components"]["skills"],
        ats=ats,
        recommendations=recs,
        explanation=explanation,
        explanation_method=method,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    # keep vector index fresh with the new job chunks
    index_resume_context(db, resume)

    logger.info(
        "Analysis complete resume=%s job=%s score=%.3f",
        resume.id, job.id, result["overall_score"],
    )
    return analysis


def analysis_to_dict(a: Analysis, job: Job) -> dict[str, Any]:
    return {
        "analysis_id": a.id,
        "resume_id": a.resume_id,
        "job_id": a.job_id,
        "job_title": job.title or "Role",
        "overall_score": a.overall_score,
        "breakdown": {k: v for k, v in a.breakdown.items() if k != "overall"},
        "weights": {"skills": 0.35, "experience": 0.20, "projects": 0.15, "education": 0.10, "semantic": 0.20},
        "components": {
            "skills": a.skill_comparison,
            "experience": _experience_component(a.breakdown),
            "projects": _projects_component(a.breakdown),
            "education": _education_component(a.breakdown),
            "semantic": _semantic_component(a.breakdown),
        },
        "skill_comparison": a.skill_comparison,
        "ats": a.ats,
        "recommendations": a.recommendations,
        "explanation": a.explanation,
        "explanation_method": a.explanation_method,
        "created_at": a.created_at.isoformat(),
    }


def _experience_component(breakdown: dict) -> dict:
    return {"score": breakdown.get("experience", 0.0), "evidence": ["See explanation for details."]}


def _projects_component(breakdown: dict) -> dict:
    return {"score": breakdown.get("projects", 0.0), "evidence": ["See explanation for details."]}


def _education_component(breakdown: dict) -> dict:
    return {"score": breakdown.get("education", 0.0), "evidence": ["See explanation for details."]}


def _semantic_component(breakdown: dict) -> dict:
    return {"score": breakdown.get("semantic", 0.0), "evidence": ["See explanation for details."]}


def create_job_for_resume(db: Session, resume: Resume, title: str | None, company: str | None, description: str) -> Job:
    parsed = extract_job(description)
    final_title = (title and title.strip()) or parsed.get("title") or "Untitled role"
    job = Job(
        resume_id=resume.id,
        title=final_title[:300],
        company=(company or parsed.get("company") or "")[:200],
        raw_text=description,
        parsed={k: v for k, v in parsed.items() if not k.startswith("_")},
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def rank_jobs_for_resume(db: Session, resume_id: str) -> list[dict]:
    resume = db.get(Resume, resume_id)
    if not resume:
        return []
    analyses = db.query(Analysis).filter(Analysis.resume_id == resume_id).all()
    jobs = {j.id: j for j in db.query(Job).filter(Job.resume_id == resume_id).all()}
    items = []
    for a in analyses:
        job = jobs.get(a.job_id)
        if not job:
            continue
        items.append(
            {
                "job_id": a.job_id,
                "job_title": job.title,
                "overall_score": a.overall_score,
                "components": {"skills": a.skill_comparison},
                "breakdown": {k: v for k, v in a.breakdown.items() if k != "overall"},
            }
        )
    return rank_jobs(items)
