"""Job description + match analysis endpoints (blocking + SSE streaming)."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.models.entities import Analysis, Job, Resume
from app.schemas.schemas import AnalysisResponse, JobCreateResponse, JobsAnalyzeRequest
from app.services.pipeline import analysis_to_dict, create_job_for_resume, run_match

router = APIRouter(tags=["jobs"])
logger = logging.getLogger("app.api.jobs")


@router.post("/jobs/analyze", response_model=list[AnalysisResponse])
def analyze_jobs(payload: JobsAnalyzeRequest, db: Session = Depends(get_db)) -> list[AnalysisResponse]:
    resume = db.get(Resume, payload.resume_id)
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found. Upload a resume first.")

    if len(payload.jobs) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 job descriptions per request.")

    responses: list[AnalysisResponse] = []
    for job_in in payload.jobs:
        job = create_job_for_resume(db, resume, job_in.title, job_in.company, job_in.description)
        analysis = run_match(db, resume, job)
        responses.append(AnalysisResponse(**analysis_to_dict(analysis, job)))
    return responses


@router.post("/jobs/analyze/stream")
def analyze_jobs_stream(payload: JobsAnalyzeRequest, db: Session = Depends(get_db)) -> StreamingResponse:
    """SSE stream: each job's full analysis is emitted the moment it's ready,
    so the UI can render results one by one instead of blocking on all jobs."""
    resume = db.get(Resume, payload.resume_id)
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found. Upload a resume first.")
    if len(payload.jobs) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 job descriptions per request.")

    total = len(payload.jobs)

    def event_stream():
        done = 0
        for job_in in payload.jobs:
            try:
                job = create_job_for_resume(db, resume, job_in.title, job_in.company, job_in.description)
                analysis = run_match(db, resume, job)
                data = analysis_to_dict(analysis, job)
                done += 1
                yield (
                    "data: "
                    + json.dumps({"type": "result", "index": done - 1, "total": total, "analysis": data})
                    + "\n\n"
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Streaming analysis failed for a job: %s", exc)
                yield (
                    "data: "
                    + json.dumps({"type": "error", "index": done, "detail": "One job analysis failed."})
                    + "\n\n"
                )
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/match", response_model=AnalysisResponse)
def match_resume_job(payload: JobsAnalyzeRequest, db: Session = Depends(get_db)) -> AnalysisResponse:
    """Single-job convenience alias of /jobs/analyze."""
    payload.jobs = payload.jobs[:1]
    results = analyze_jobs(payload, db)
    if not results:
        raise HTTPException(status_code=500, detail="Analysis failed unexpectedly.")
    return results[0]


@router.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(analysis_id: str, db: Session = Depends(get_db)) -> AnalysisResponse:
    a = db.get(Analysis, analysis_id)
    if not a:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    job = db.get(Job, a.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Linked job not found.")
    return AnalysisResponse(**analysis_to_dict(a, job))


@router.get("/jobs", response_model=list[JobCreateResponse])
def list_jobs(resume_id: str, db: Session = Depends(get_db)) -> list[JobCreateResponse]:
    jobs = db.query(Job).filter(Job.resume_id == resume_id).all()
    return [
        JobCreateResponse(
            job_id=j.id,
            title=j.title,
            parsed=j.parsed,
            extraction_method=j.parsed.get("_extraction_method", "heuristic"),
        )
        for j in jobs
    ]
