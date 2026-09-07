"""Health & diagnostics endpoints."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter
from sqlalchemy import text

from app.core.config import settings
from app.core.database import engine
from app.schemas.schemas import HealthResponse
from app.services.ai.embeddings import get_backend

router = APIRouter(tags=["health"])
logger = logging.getLogger("app.health")

_STARTED_AT = time.time()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    db_ok = True
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        logger.error("Health DB check failed: %s", exc)
        db_ok = False

    backend_name = "unknown"
    try:
        backend_name = get_backend().name
    except Exception as exc:  # noqa: BLE001
        logger.error("Embedding backend check failed: %s", exc)

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        version="1.0.0",
        llm_available=settings.llm_available,
        llm_provider=settings.llm_provider,
        embedding_backend=backend_name,
        database=settings.database_url.split("://")[0] + ("://…" if db_ok else " (unreachable)"),
    )
