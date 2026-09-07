"""Resume upload + profile endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.entities import Analysis, Job, Resume
from app.schemas.schemas import ResumeProfileResponse
from app.services.document_parser import DocumentParseError, parse_document
from app.services.extractor import extract_resume
from app.services.skills import normalizer

router = APIRouter(prefix="/resume", tags=["resume"])
logger = logging.getLogger("app.api.resume")


@router.post("/upload", response_model=ResumeProfileResponse)
async def upload_resume(file: UploadFile = File(...), db: Session = Depends(get_db)) -> ResumeProfileResponse:
    filename = file.filename or "resume"
    try:
        data = await file.read()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Could not read the uploaded file.") from exc

    try:
        parsed = parse_document(filename, data)
    except DocumentParseError as exc:
        logger.warning("Rejected upload '%s': %s", filename, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    profile = extract_resume(parsed.text)

    resume = Resume(
        filename=filename,
        file_type=filename.rsplit(".", 1)[-1].lower(),
        file_size_bytes=len(data),
        raw_text=parsed.text,
        parse_method=parsed.method,
        extraction_status="fallback" if profile.get("_extraction_method") == "heuristic" else "ok",
        profile={k: v for k, v in profile.items() if not k.startswith("_")},
        warnings=parsed.warnings,
    )
    profile.pop("_extraction_method", None)
    db.add(resume)
    db.commit()
    db.refresh(resume)

    logger.info("Resume stored id=%s method=%s skills=%d", resume.id, resume.parse_method, len(resume.profile.get("skills", {})))

    return ResumeProfileResponse(
        resume_id=resume.id,
        filename=resume.filename,
        parse_method=resume.parse_method,
        extraction_status=resume.extraction_status,
        profile=resume.profile,
        warnings=resume.warnings,
        created_at=resume.created_at.isoformat(),
    )


@router.get("/{resume_id}", response_model=ResumeProfileResponse)
def get_resume(resume_id: str, db: Session = Depends(get_db)) -> ResumeProfileResponse:
    resume = db.get(Resume, resume_id)
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found.")
    return ResumeProfileResponse(
        resume_id=resume.id,
        filename=resume.filename,
        parse_method=resume.parse_method,
        extraction_status=resume.extraction_status,
        profile=resume.profile,
        warnings=resume.warnings,
        created_at=resume.created_at.isoformat(),
    )


@router.get("/{resume_id}/history")
def resume_history(resume_id: str, db: Session = Depends(get_db)) -> dict:
    resume = db.get(Resume, resume_id)
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found.")
    items = db.query(Analysis).filter(Analysis.resume_id == resume_id).order_by(Analysis.created_at.desc()).all()
    jobs = {j.id: j for j in db.query(Job).filter(Job.resume_id == resume_id).all()}
    return {
        "resume_id": resume_id,
        "analyses": [
            {
                "analysis_id": a.id,
                "job_id": a.job_id,
                "job_title": jobs[a.job_id].title if a.job_id in jobs else "Role",
                "overall_score": a.overall_score,
                "created_at": a.created_at.isoformat(),
            }
            for a in items
        ],
    }
