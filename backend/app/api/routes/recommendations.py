"""Job recommendation ranking endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.entities import Resume
from app.schemas.schemas import RankedJobSchema
from app.services.pipeline import rank_jobs_for_resume

router = APIRouter(prefix="/recommendations", tags=["recommendations"])
logger = logging.getLogger("app.api.recommendations")


@router.get("", response_model=list[RankedJobSchema])
def get_recommendations(resume_id: str, db: Session = Depends(get_db)) -> list[RankedJobSchema]:
    resume = db.get(Resume, resume_id)
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found.")
    ranked = rank_jobs_for_resume(db, resume_id)
    return [RankedJobSchema(**r) for r in ranked]
