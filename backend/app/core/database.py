"""SQLAlchemy database session management."""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings

logger = logging.getLogger("app.db")

_connect_args = {}
if settings.database_url.startswith("sqlite"):
    # Ensure the sqlite parent directory exists before engine creation.
    db_file = settings.database_url.split("///", 1)[-1]
    Path(db_file).parent.mkdir(parents=True, exist_ok=True)
    _connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.database_url,
    connect_args=_connect_args,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def init_db() -> None:
    """Create all tables (idempotent)."""
    from app import models  # noqa: F401  (import registers models on Base)

    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized at %s", settings.database_url)


def get_db() -> Session:
    """FastAPI dependency yielding a scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
