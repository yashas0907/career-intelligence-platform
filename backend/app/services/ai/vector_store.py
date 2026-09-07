"""Lightweight in-process vector store for RAG over resume + JD chunks.

Why not a separate vector DB server?
- Documents are small (1 resume + N job descriptions, a few KB each).
- A pure-numpy in-process store keeps the project reproducible (docker-compose
  stays 2 services) while demonstrating the same retrieval concepts:
  chunking, embedding, cosine top-k retrieval, and evidence citation.
- Embeddings are cached on disk next to the SQLite DB so restarts are cheap.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.services.ai.embeddings import cosine_matrix, get_backend

logger = logging.getLogger("app.vstore")

_CHUNK_SPLIT_RE = re.compile(r"\n{2,}|\f")


@dataclass
class Chunk:
    id: str
    doc_id: str
    source: str  # resume | job
    label: str   # e.g. "Job 2 — Machine Learning Intern"
    text: str
    index: int

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_text(text: str, doc_id: str, source: str, label: str, max_chars: int = 700, overlap: int = 0) -> list[Chunk]:
    """Paragraph-aware chunking; falls back to sliding window for giant blobs."""
    paragraphs = [p.strip() for p in _CHUNK_SPLIT_RE.split(text) if p.strip()]
    chunks: list[Chunk] = []
    buf: list[str] = []
    size = 0
    idx = 0

    def flush() -> None:
        nonlocal buf, size, idx
        if buf:
            chunks.append(Chunk(f"{doc_id}:{idx}", doc_id, source, label, "\n".join(buf), idx))
            idx += 1
            buf, size = [], 0

    for para in paragraphs:
        if len(para) > max_chars:  # hard-split long paragraphs
            flush()
            for j in range(0, len(para), max_chars - overlap):
                piece = para[j : j + max_chars - overlap]
                if piece.strip():
                    chunks.append(Chunk(f"{doc_id}:{idx}", doc_id, source, label, piece.strip(), idx))
                    idx += 1
            continue
        if size + len(para) > max_chars:
            flush()
        buf.append(para)
        size += len(para) + 1
    flush()
    return chunks


class VectorStore:
    """In-memory index persisted to disk (JSON + npy), one namespace per resume."""

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None
        self._store_dir = Path(settings.database_url.split("///", 1)[-1]).parent / "vector_store"
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self._store_dir / f"{namespace}.json"
        self._vec_path = self._store_dir / f"{namespace}.npy"
        self._backend_label = get_backend().name

    # -- persistence -------------------------------------------------------
    def save(self) -> None:
        if self._matrix is None or not self.chunks:
            return
        np.save(self._vec_path, self._matrix)
        payload = {
            "backend": self._backend_label,
            "chunks": [c.to_dict() for c in self.chunks],
        }
        self._meta_path.write_text(json.dumps(payload), encoding="utf-8")

    def load(self) -> bool:
        if not (self._meta_path.exists() and self._vec_path.exists()):
            return False
        try:
            payload = json.loads(self._meta_path.read_text(encoding="utf-8"))
            if payload.get("backend") != self._backend_label:
                return False  # embedding space changed -> rebuild
            self.chunks = [Chunk(**c) for c in payload["chunks"]]
            self._matrix = np.load(self._vec_path)
            return bool(self.chunks) and self._matrix.shape[0] == len(self.chunks)
        except Exception as exc:  # noqa: BLE001
            logger.warning("VectorStore load failed for %s: %s", self.namespace, exc)
            return False

    # -- indexing ----------------------------------------------------------
    def index(self, texts: list[str], doc_map: dict[str, tuple[str, str]]) -> None:
        """Embed chunks. doc_map: doc_id -> (source, label)."""
        if not texts:
            return
        self.chunks = []
        backend = get_backend()
        matrix = backend.embed(texts)
        self._matrix = matrix
        self.save()
        logger.info("Indexed %d chunks for %s", len(self.chunks), self.namespace)

    def build_from_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        backend = get_backend()
        self._backend_label = backend.name
        self.chunks = chunks
        self._matrix = backend.embed([c.text for c in chunks])
        self.save()
        logger.info("Indexed %d chunks for %s (backend=%s)", len(chunks), self.namespace, backend.name)

    # -- retrieval ---------------------------------------------------------
    def search(self, query: str, top_k: int = 5, source: str | None = None) -> list[tuple[Chunk, float]]:
        if self._matrix is None or not self.chunks:
            return []
        backend = get_backend()
        q = backend.embed([query])
        sims = cosine_matrix(q, self._matrix)[0]
        order = np.argsort(-sims)
        results: list[tuple[Chunk, float]] = []
        for i in order:
            c = self.chunks[int(i)]
            if source and c.source != source:
                continue
            results.append((c, float(sims[i])))
            if len(results) >= top_k:
                break
        return results
