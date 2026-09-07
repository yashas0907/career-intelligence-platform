"""RAG chat endpoints: classic JSON + SSE streaming variant."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.entities import Analysis, ChatMessage, Job, Resume
from app.schemas.schemas import ChatRequest, ChatResponse, ChatSourceSchema
from app.services.pipeline import index_resume_context
from app.services.rag import answer_question, answer_question_stream

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger("app.api.chat")


def _resolve_context(payload: ChatRequest, db: Session) -> tuple[Resume, Analysis | None]:
    """Find the resume (and optional analysis) for the question."""
    analysis: Analysis | None = None
    if payload.analysis_id:
        analysis = db.get(Analysis, payload.analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found.")
        resume = db.get(Resume, analysis.resume_id)
        if not resume:
            raise HTTPException(status_code=404, detail="Resume for analysis not found.")
        return resume, analysis

    # No analysis_id: fall back to the most recent resume — chatting with
    # just a resume (before any job analysis) is a valid user journey.
    resume = db.query(Resume).order_by(Resume.created_at.desc()).first()
    if not resume:
        raise HTTPException(status_code=400, detail="No resume uploaded yet. Upload a resume first.")
    analysis = (
        db.query(Analysis)
        .filter(Analysis.resume_id == resume.id)
        .order_by(Analysis.created_at.desc())
        .first()
    )
    return resume, analysis


def _build_analysis_payload(a: Analysis) -> dict:
    skills = a.skill_comparison or {}
    return {
        "overall_score": a.overall_score,
        "breakdown": a.breakdown,
        "skill_comparison": skills,
    }


def _get_history(db: Session, analysis_id: str) -> list[dict[str, str]]:
    return [
        {"role": m.role, "content": m.content}
        for m in db.query(ChatMessage)
        .filter(ChatMessage.analysis_id == analysis_id)
        .order_by(ChatMessage.id.desc())
        .limit(6)
    ][::-1]


def _store_messages(db: Session, analysis_id: str, question: str, answer: str, sources: list[dict]) -> None:
    db.add(ChatMessage(analysis_id=analysis_id, role="user", content=question))
    db.add(
        ChatMessage(
            analysis_id=analysis_id,
            role="assistant",
            content=answer,
            retrieved_chunks=sources,
        )
    )
    db.commit()


@router.post("", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    resume, analysis = _resolve_context(payload, db)
    store = index_resume_context(db, resume)
    if not store.chunks:
        raise HTTPException(status_code=500, detail="Document index is empty — please re-upload.")

    result = answer_question(
        payload.question,
        store,
        _build_analysis_payload(analysis) if analysis else None,
        _get_history(db, analysis.id) if analysis else [],
    )
    if analysis:
        _store_messages(db, analysis.id, payload.question, result["answer"], result["sources"])

    logger.info("Chat answered resume=%s method=%s sources=%d", resume.id, result["method"], len(result["sources"]))
    return ChatResponse(
        answer=result["answer"],
        sources=[ChatSourceSchema(**s) for s in result["sources"]],
        method=result["method"],
        analysis_id=analysis.id if analysis else None,
    )


@router.post("/stream")
def chat_stream(payload: ChatRequest, db: Session = Depends(get_db)) -> StreamingResponse:
    """SSE stream: meta (sources) -> answer tokens -> done(method)."""
    resume, analysis = _resolve_context(payload, db)
    store = index_resume_context(db, resume)
    if not store.chunks:
        raise HTTPException(status_code=500, detail="Document index is empty — please re-upload.")

    analysis_payload = _build_analysis_payload(analysis) if analysis else None
    question = payload.question

    def event_stream():
        full_answer: list[str] = []
        final_sources: list[dict] = []
        final_method = "none"
        try:
            for event in answer_question_stream(question, store, analysis_payload):
                if event["type"] == "meta":
                    final_sources = event["sources"]
                    yield f"data: {json.dumps({'type': 'meta', 'sources': event['sources']})}\n\n"
                elif event["type"] == "answer":
                    full_answer.append(event["text"])
                    yield f"data: {json.dumps({'type': 'answer', 'text': event['text']})}\n\n"
                elif event["type"] == "done":
                    final_method = event["method"]
                    yield f"data: {json.dumps({'type': 'done', 'method': event['method']})}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Chat stream failed: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'detail': 'The assistant hit an error mid-answer.'})}\n\n"
        finally:
            if analysis and full_answer:
                try:
                    _store_messages(db, analysis.id, question, "".join(full_answer), final_sources)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to persist chat after stream: %s", exc)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/history")
def chat_history(analysis_id: str, db: Session = Depends(get_db)) -> dict:
    msgs = (
        db.query(ChatMessage)
        .filter(ChatMessage.analysis_id == analysis_id)
        .order_by(ChatMessage.id.asc())
        .all()
    )
    return {
        "analysis_id": analysis_id,
        "messages": [
            {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
            for m in msgs
        ],
    }
