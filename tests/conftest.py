"""Shared pytest fixtures: isolated DB, temp env, sample documents."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

# Isolate data dir BEFORE importing app modules
_TMP = tempfile.mkdtemp(prefix="acip_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP}/test.db"
os.environ["LOG_FILE"] = f"{_TMP}/test.log"
os.environ["EMBEDDING_BACKEND"] = "hashing"  # deterministic, no network, fast
os.environ["LLM_ENABLED"] = "false"


@pytest.fixture(scope="session")
def sample_texts():
    from sample_texts import SAMPLE_JD_BACKEND, SAMPLE_JD_ML, SAMPLE_RESUME

    return {"resume": SAMPLE_RESUME, "jd_ml": SAMPLE_JD_ML, "jd_backend": SAMPLE_JD_BACKEND}


@pytest.fixture()
def db_session():
    from app.core.database import SessionLocal, init_db

    init_db()
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture()
def client(db_session):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def uploaded_resume(client, sample_texts):
    resp = client.post(
        "/api/resume/upload",
        files={"file": ("resume.txt", sample_texts["resume"].encode("utf-8"), "text/plain")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()
