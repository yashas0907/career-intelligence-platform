"""Embeddings + semantic similarity with a pluggable backend.

Backends (in priority order when `EMBEDDING_BACKEND=auto`):
1. OpenAI `text-embedding-3-small` — if `OPENAI_API_KEY` is set (no local model needed).
2. sentence-transformers `all-MiniLM-L6-v2` — fully offline, if installed.
3. HashingVectorizer fallback — deterministic lexical hashing; keeps the whole
   pipeline runnable with zero external dependencies (degraded semantic quality).

All backends expose the same interface: embed(texts) -> matrix, and similarity is
cosine. Dimension can vary; consumers store embeddings per-backend-label.
"""

from __future__ import annotations

import logging
import re
import zlib
from functools import lru_cache

import numpy as np

from app.core.config import settings

logger = logging.getLogger("app.embeddings")


class EmbeddingBackend:
    name = "base"
    dim = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

class OpenAIEmbeddings(EmbeddingBackend):
    """OpenAI embeddings (paid). Gemini's free tier does NOT expose an
    OpenAI-compatible embeddings endpoint, so when the LLM provider is Gemini
    we prefer the local sentence-transformers backend instead."""

    def __init__(self) -> None:
        from openai import OpenAI

        kwargs: dict = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        self._client = OpenAI(**kwargs)
        self._model = settings.openai_embedding_model
        self.name = f"openai:{self._model}"
        self.dim = 1536 if "3-small" in self._model else 3072 if "3-large" in self._model else 1536

    def embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        # batch to stay under API limits
        for i in range(0, len(texts), 100):
            batch = [t if t.strip() else " " for t in texts[i : i + 100]]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            out.extend(d.embedding for d in resp.data)
        return np.asarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# SentenceTransformers backend
# ---------------------------------------------------------------------------

class STEmbeddings(EmbeddingBackend):
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(settings.st_model)
        self.name = f"st:{settings.st_model}"
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


# ---------------------------------------------------------------------------
# Deterministic hashing fallback (always available)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9#+.]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _bigrams(tokens: list[str]) -> list[str]:
    return [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]


class HashingEmbeddings(EmbeddingBackend):
    """Bag-of-ngrams hashed into a fixed vector, L2-normalized.

    Not truly 'semantic', but deterministic, dependency-free, and its cosine
    similarity correlates strongly with lexical overlap — good enough as a
    graceful-degradation path. We label it clearly so we never over-claim.
    """

    def __init__(self, dim: int = 512) -> None:
        self.name = "hashing:512"
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = _tokens(text)
            feats = toks + _bigrams(toks)
            for f in feats:
                # stable across processes/restarts (builtin hash() is
                # seed-randomized per process and would break persistence)
                h = zlib.crc32(f.encode("utf-8")) % self.dim
                vecs[i, h] += 1.0
            n = np.linalg.norm(vecs[i])
            if n > 0:
                vecs[i] /= n
        return vecs


@lru_cache(maxsize=1)
def get_backend() -> EmbeddingBackend:
    pref = settings.embedding_backend
    # Gemini free tier has no embeddings endpoint -> prefer local ST there.
    if settings.llm_provider == "gemini" and pref == "auto":
        pref = "st"
    if pref == "hashing":
        backend = HashingEmbeddings()
        logger.info("Embeddings backend: %s (forced)", backend.name)
        return backend
    if pref == "openai" or (pref == "auto" and settings.openai_api_key.strip()):
        try:
            backend = OpenAIEmbeddings()
            logger.info("Embeddings backend: %s", backend.name)
            return backend
        except Exception as exc:  # noqa: BLE001
            logger.error("OpenAI embeddings unavailable (%s); falling back", exc)
    if pref in {"st", "auto"}:
        try:
            backend = STEmbeddings()
            logger.info("Embeddings backend: %s", backend.name)
            return backend
        except Exception as exc:  # noqa: BLE001
            logger.warning("sentence-transformers unavailable: %s", exc)
    backend = HashingEmbeddings()
    logger.info("Embeddings backend: %s (lexical fallback)", backend.name)
    return backend


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between each row of a and each row of b."""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity in [0, 1] (clamped from [-1, 1])."""
    backend = get_backend()
    vecs = backend.embed([text_a, text_b])
    sim = float(cosine_matrix(vecs[:1], vecs[1:])[0, 0])
    return max(0.0, min(1.0, (sim + 1) / 2))
